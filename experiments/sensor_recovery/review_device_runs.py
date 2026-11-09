"""Audit supplied UART captures, detect identical content, and retain evidence.

No new MCU measurement is performed. Content equality does not establish
whether files came from the same acquisition or distinct deterministic runs.
Batch repetitions within one capture are not independent device runs.
"""
import argparse
import hashlib
import json
import re
import statistics
from datetime import datetime
from pathlib import Path

from mcu.capture import parse_lines

HERE = Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def verify_session(raw, supplied, parsed, sidecar):
    """Check acquisition evidence, not device authenticity or firmware identity."""
    if sidecar is None:
        if "SESSION" in parsed["metadata"]:
            raise ValueError("A command capture requires its .session.json sidecar")
        return None
    s=json.loads(sidecar.decode("utf-8-sig"))
    if s.get("status")!="validated" or s.get("raw_sha256")!=sha(raw) or s.get("result_sha256")!=sha(supplied) or s.get("raw_bytes")!=len(raw):
        raise ValueError("Sidecar status, byte count or hashes do not match")
    def timestamp(value):
        t=datetime.fromisoformat(value)
        if t.tzinfo is None:raise ValueError("Session timestamp lacks timezone")
        return t
    start,end=timestamp(s["start_utc"]),timestamp(s["end_utc"])
    if end<start:raise ValueError("Session timestamps are reversed")
    token=s.get("session_id","")
    if not re.fullmatch("[0-9a-f]{32}",token):raise ValueError("Invalid session ID")
    if s.get("mode")=="command_v3":
        meta=parsed["metadata"]
        if meta.get("SESSION")!=token or meta.get("TRANSPORT")!="usb-jtag-poll-command-v3" or meta["BOARD"]!="esp32c3" or s.get("usb_phase")!="complete":
            raise ValueError("Session identity or transport differs from device log")
        commands=s["tx_commands"];texts=[x["text"] for x in commands]
        if len(texts)<3 or texts[-2:]!=["Q,"+token,"R,"+token] or any(x!="H" for x in texts[:-2]):
            raise ValueError("Unexpected/repeated start command sequence")
        times=[start,*[timestamp(x["utc"]) for x in commands],timestamp(s["run_accepted_utc"]),end]
        if times!=sorted(times):raise ValueError("Command/acknowledgement timestamps are inconsistent")
        lines=[line.strip() for line in raw.decode("utf-8-sig").splitlines() if line.strip()]
        echo="ECHO,"+token;accepted="ACCEPTED,"+token
        if [x for x in lines if x.startswith("ECHO,")]!=[echo] or [x for x in lines if x.startswith("ACCEPTED,")]!=[accepted]:
            raise ValueError("Missing/mismatched/duplicate device acknowledgement")
        ready=lines.index("READY,SENSOR_USB_V3,esp32c3")
        header=next(i for i,x in enumerate(lines) if x.startswith("SENSOR_BENCH_V1,"))
        if not ready<lines.index(echo)<lines.index(accepted)<header or any(x.startswith(("ERROR,","ALREADY_RAN,")) for x in lines):
            raise ValueError("Invalid device startup sequence")
    elif s.get("mode")!="legacy_auto_start" or "SESSION" in parsed["metadata"]:
        raise ValueError("Unsupported session mode")
    script=(HERE/"mcu/capture.py").read_bytes().replace(b"\r\n",b"\n")
    return dict(session_id=token,mode=s["mode"],start_utc=s["start_utc"],end_utc=s["end_utc"],
                duration_s=(end-start).total_seconds(),sidecar_sha256=sha(sidecar),
                hashes_and_identity_verified=True,
                capture_script_matches_current=s.get("capture_script_sha256") in
                    {sha(script),sha(script.replace(b"\n",b"\r\n"))})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--reference-log", type=Path, help="Previously captured board log, with paired JSON, for timing comparison")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Choose a fresh output directory")
    fixture = json.loads((HERE / "mcu/fixture_manifest.json").read_text())
    expected_fixture = ["synthetic_" + fixture["scenario"],
                        "seed" + str(fixture["seed"]),
                        "unit" + str(fixture["unit"]), "B" + str(fixture["budget"])]
    manifests={None:fixture,"spline_optimized_v1":json.loads((HERE/"mcu/optimized_manifest.json").read_text())}
    def expected_payloads(parsed):
        manifest=manifests[parsed["metadata"].get("SUITE")]
        return {"overhead":0,**{m["name"]:m["payload_bytes"] for m in manifest["models"]}}
    observations, content_fingerprints, submissions, sessions = {}, set(), [], {}
    for log in args.logs:
        raw = log.read_bytes()
        parsed = parse_lines(raw.decode("utf-8-sig").splitlines())
        supplied_json = log.with_suffix(".json")
        supplied = supplied_json.read_bytes()
        if parsed != json.loads(supplied.decode("utf-8-sig")):
            raise ValueError(f"Parsed log differs from supplied JSON: {log.name}")
        if parsed["metadata"]["fixture"] != expected_fixture:
            raise ValueError(f"Unexpected numerical fixture: {log.name}")
        if parsed["payload_bytes"] != expected_payloads(parsed):
            raise ValueError(f"Unexpected payload manifest: {log.name}")
        sidecar_path=log.with_suffix(".session.json")
        sidecar=sidecar_path.read_bytes() if sidecar_path.exists() else None
        session=verify_session(raw,supplied,parsed,sidecar)
        fingerprint = sha(json.dumps(parsed, sort_keys=True, separators=(",", ":")).encode())
        content_fingerprints.add(fingerprint)
        if session:
            token=session["session_id"]
            identity=(fingerprint,json.loads(sidecar))
            if token in sessions and sessions[token]!=identity:
                raise ValueError("One session ID has conflicting measurement contents or acquisition metadata")
            sessions[token]=identity
        submissions.append(dict(log_name=log.name, json_name=supplied_json.name,
                                log_sha256=sha(raw), json_sha256=sha(supplied),
                                parsed_content_sha256=fingerprint,session=session))
        # Deterministic Mega resets can yield identical data with distinct
        # host-recorded acquisitions. Deduplicate identical submissions of one
        # session, not separate identified sessions with identical predictions.
        key=("session",session["session_id"]) if session else ("content",fingerprint)
        if key not in observations:
            observations[key] = dict(fingerprint=fingerprint,raw=raw,supplied=supplied,parsed=parsed,sidecar=sidecar,session=session)

    reference=None
    if args.reference_log:
        reference=parse_lines(args.reference_log.read_text(encoding="utf-8-sig").splitlines())
        if reference!=json.loads(args.reference_log.with_suffix(".json").read_text(encoding="utf-8-sig")) or reference["metadata"]["fixture"]!=expected_fixture or reference["payload_bytes"]!=expected_payloads(reference):
            raise ValueError("Reference capture/JSON/fixture mismatch")

    args.out.mkdir(parents=True)
    capture_dir = args.out / "captures"
    capture_dir.mkdir()
    analysis = dict(evidence_kind="user_supplied_mcu_uart_capture_on_synthetic_fixture",
                    submitted_pairs=len(submissions), unique_capture_contents=len(content_fingerprints),
                    retained_capture_observations=len(observations),
                    independent_session_count=None,
                    identified_session_count=len(sessions),
                    session_identity_status="sidecars_verified" if len(sessions)==len(observations) else "legacy_or_mixed_evidence",
                    submissions=submissions,
                    numerical_fixture_manifest_sha256=sha((HERE / "mcu/fixture_manifest.json").read_bytes()),
                    analysis_script_sha256=sha(Path(__file__).read_bytes()), captures=[])
    lines = ["# User-supplied MCU run review", "",
             f"Received {len(submissions)} log/JSON pairs with {len(content_fingerprints)} unique parsed contents; retained {len(observations)} capture observations.",
             "Each raw log parses completely and reproduces its supplied JSON exactly. "
             "Fixture metadata and model payloads match the checked-in manifest.", "",
             f"Verified session IDs with acquisition sidecars: {len(sessions)}. "
             "C3 command IDs are echoed by the device; legacy auto-start IDs and acquisition times are recorded by the host only. "
             "Recorded session identity distinguishes acquisitions, but does not establish statistical independence, different boards or cold boots. "
             "Legacy files without sidecars have no verified acquisition identity. Equal contents alone do not prove a copied acquisition. "
             "The ten timing batches within a capture measure within-run variation; they are not ten independent boards or reset sessions.", ""]
    for index, item in enumerate(observations.values(), 1):
        fingerprint=item["fingerprint"]
        d = item["parsed"]
        stem = f"{d['metadata']['BOARD']}_capture{index:02d}"
        (capture_dir / (stem + ".log")).write_bytes(item["raw"])
        (capture_dir / (stem + ".json")).write_bytes(item["supplied"])
        if item["sidecar"] is not None:
            (capture_dir/(stem+".session.json")).write_bytes(item["sidecar"])
        models = []
        for name, accuracy in d["accuracy"].items():
            times = [r["total_us"] / r["calls"] for r in d["timings"][name]]
            median = statistics.median(times)
            models.append(dict(method=name, payload_bytes=d["payload_bytes"][name],
                               **accuracy, **d["summary"][name],
                               min_batch_us_per_call=min(times), max_batch_us_per_call=max(times),
                               batch_range_percent=100 * (max(times) - min(times)) / median))
        speeds = {lut: {other: d["summary"][other]["gross_us_per_call"] /
                       d["summary"][lut]["gross_us_per_call"]
                       for other in ("polynomial", "cubic16", "kan16","cubic16_unrolled","cubic16_horner") if other in d["summary"]}
                  for lut in ("uniform32_float32", "uniform32_int16")}
        update_us = statistics.median(r["total_us"] / r["calls"] for r in d["update_cost_only"])
        analysis["captures"].append(dict(id=stem, parsed_content_sha256=fingerprint,
                                          metadata=d["metadata"], models=models,
                                          gross_speedups=speeds, update_cost_gross_us=update_us,session=item["session"]))
        lines += [f"## {stem}", "",
                  f"All {len(models)} prediction checks PASS at the predeclared 0.002 °C maximum host-gap tolerance. "
                  f"The largest observed gap is {max(m['max_host_gap_C'] for m in models):.7f} °C.", "",
                  "Times are medians of ten batches of 2048 calls. Gross times include fixture reads, "
                  "loop/call/checksum overhead, normalization and output conversion; they exclude ADC acquisition and Serial printing. "
                  f"The separate baseline median is {d['overhead_us_per_call']:.6f} µs/call. "
                  "Corrected values are estimates, not exact isolated-kernel times.", "",
                  "| Method | Payload B | Gross µs/call | Corrected µs/call | MAE °C | Max error °C |",
                  "|---|---:|---:|---:|---:|---:|"]
        for m in models:
            lines.append(f"| {m['method']} | {m['payload_bytes']} | {m['gross_us_per_call']:.6f} | "
                         f"{m['overhead_subtracted_us_per_call']:.6f} | {m['mae_C']:.7f} | {m['max_C']:.7f} |")
        lines += ["", "Gross speed ratios (comparator time / LUT time):", ""]
        for lut, ratios in speeds.items():
            lines.append(f"- {lut}: " + "; ".join(f"{k} {v:.3f}×" for k, v in ratios.items()) + ".")
        lines += ["", f"The projected-update cost median is {update_us:.6f} µs/step; all ten update batches retain monotonicity. "
                  "This primitive uses synthetic targets and does not demonstrate calibration convergence or drift adaptation.", ""]
        if reference:
            comparison=[dict(method=name,reference_us_per_call=reference["summary"][name]["gross_us_per_call"],
                             current_us_per_call=d["summary"][name]["gross_us_per_call"],
                             reference_over_current=reference["summary"][name]["gross_us_per_call"]/d["summary"][name]["gross_us_per_call"])
                        for name in d["summary"] if name in reference["summary"]]
            analysis["captures"][-1]["board_comparison"]=dict(reference_metadata=reference["metadata"],
                reference_log_sha256=sha(args.reference_log.read_bytes()),rows=comparison)
            lines += [f"### Comparison with {reference['metadata']['BOARD']}","",
                      "Same numerical fixture and payloads; different boards, clock rates, toolchains and transport. "
                      "Ratios describe these captures, not a frequency-normalized architectural result.","",
                      "| Method | Reference µs/call | Current µs/call | Reference/current |",
                      "|---|---:|---:|---:|"]
            for row in comparison:
                lines.append(f"| {row['method']} | {row['reference_us_per_call']:.6f} | {row['current_us_per_call']:.6f} | {row['reference_over_current']:.3f}× |")
            lines.append("")
    analysis["repeatability"]={}
    for board,suite in sorted({(c["metadata"]["BOARD"],c["metadata"].get("SUITE","legacy8")) for c in analysis["captures"]}):
        captures=[c for c in analysis["captures"] if c["metadata"]["BOARD"]==board and c["metadata"].get("SUITE","legacy8")==suite]
        if len(captures)<2:continue
        rows=[];accuracy_fields=("mae_C","rmse_C","max_C","max_host_gap_C")
        accuracy_sets=[{m["method"]:tuple(m[k] for k in accuracy_fields) for m in c["models"]} for c in captures]
        for method in accuracy_sets[0]:
            values=[next(m["gross_us_per_call"] for m in c["models"] if m["method"]==method) for c in captures]
            rows.append(dict(method=method,session_medians_us=values,min_us=min(values),max_us=max(values),
                             relative_range_percent=100*(max(values)-min(values))/min(values)))
        summary=dict(captures=len(captures),identified_sessions=len({c["session"]["session_id"] for c in captures if c["session"]}),
                     accuracy_fields_identical=all(a==accuracy_sets[0] for a in accuracy_sets[1:]),
                     range_definition="100 * (max(session medians) - min(session medians)) / min(session medians)",
                     max_relative_range_percent=max(r["relative_range_percent"] for r in rows),
                     update_cost_session_medians_us=[c["update_cost_gross_us"] for c in captures],models=rows)
        group=board if suite=="legacy8" else board+"/"+suite
        analysis["repeatability"][group]=summary
        lines += [f"## Between-capture repeatability: {group}","",
                  f"Reviewed {len(captures)} captures; {summary['identified_sessions']} acquisition IDs have verified sidecars. "
                  "This checks repeatability of the supplied captures; it does not estimate between-board or cold-boot variation. "
                  "Timing batches are kept within their sessions rather than pooled as independent acquisitions.","",
                  f"All four reported accuracy fields identical across captures: {summary['accuracy_fields_identical']}. "
                  f"Largest relative range of session medians: {summary['max_relative_range_percent']:.6f}%. "
                  "Relative range is 100 × (maximum − minimum) / minimum; it is not a confidence interval.","",
                  "| Method | Minimum session median, µs/call | Maximum session median, µs/call | Relative range, % |",
                  "|---|---:|---:|---:|"]
        for row in rows:
            lines.append(f"| {row['method']} | {row['min_us']:.6f} | {row['max_us']:.6f} | {row['relative_range_percent']:.6f} |")
        lines.append("")
    c3_sessions={c["session"]["session_id"] for c in analysis["captures"] if c["metadata"]["BOARD"]=="esp32c3" and c["session"]}
    optimized_sessions={c["session"]["session_id"] for c in analysis["captures"] if c["metadata"]["BOARD"]=="esp32c3" and c["metadata"].get("SUITE")=="spline_optimized_v1" and c["session"]}
    optimized_mega_sessions={c["session"]["session_id"] for c in analysis["captures"] if c["metadata"]["BOARD"]=="mega2560" and c["metadata"].get("SUITE")=="spline_optimized_v1" and c["session"]}
    next_step=("- The preliminary two-session C3 capture checkpoint is complete. Next: optimized cubic/Horner deployment comparators, followed by physical NTC measurements. No further repeat of unchanged firmware is required for this checkpoint."
               if len(c3_sessions)>=2 else
               "- Next: complete at least two separately identified ESP32-C3 command sessions with the same fixture; do not repeat Mega runs solely to obtain different numbers. Optimized deployment baselines and physical NTC measurements remain separate stages.")
    if len(optimized_sessions)>=2:
        next_step="- Two optimized-suite C3 command sessions are available. Interpret the unrolled/Horner timing together with payload and accuracy; physical NTC validation remains a separate stage."
    if len(optimized_mega_sessions)>=2:
        next_step="- Two optimized-suite Mega host acquisition records are available, including any byte-identical deterministic outputs. Compare their timing/payload/error tradeoff with C3; matched-storage evaluation and physical NTC validation remain separate stages."
    spline_note=("- This suite includes general de Boor, unrolled and precompiled Horner cubic evaluation. The original scalar KAN edge remains; optimized float32 KAN candidates failed the existing host gate and were excluded from hardware timing."
                 if any(c["metadata"].get("SUITE")=="spline_optimized_v1" for c in analysis["captures"]) else
                 "- Cubic/KAN comparisons are against the checked-in general de Boor evaluator. An optimized piecewise-cubic/Horner deployment baseline is still needed before a general speed-superiority claim. The KAN model is one fixed-grid scalar edge plus SiLU.")
    lines += ["## Interpretation and remaining work", "",
              "- Compare the reported speed ratios together with accuracy and payload: float32 LUT32 stores 144 bytes versus 56 for the selected degree-nine Chebyshev model. A speed gain does not establish an accuracy or memory gain.",
              "- Int16 LUT32 reduces the payload to 88 bytes, but inference reconstructs floating-point values. It is not fixed-point integer inference; quantization need not reduce latency.",
              "- Uint8 and nonuniform results must be judged using both their timing and their error. This fixed fixture does not determine the best method over all sensors and budgets.",
              spline_note,
              "- Physical execution time is distinct from physical sensor accuracy: all ADC/temperature test pairs are synthetic. These measurements do not establish a new NTC accuracy specification.",
              "- The UART protocol does not embed an ELF/source hash. Matching fixture metadata and numerical checks do not cryptographically identify the flashed firmware. Retain source_hashes.json and packages.txt alongside the device runs.",
              next_step, ""]
    (args.out / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf8")
    (args.out / "RESULTS.md").write_text("\n".join(lines), encoding="utf8")
    print(json.dumps(dict(submitted_pairs=len(submissions), unique_capture_contents=len(content_fingerprints),retained_capture_observations=len(observations),
                          validated=True, output=str(args.out))))


if __name__ == "__main__":
    main()
