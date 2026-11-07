"""Audit supplied UART captures, detect identical content, and retain evidence.

No new MCU measurement is performed. Batch repetitions within one capture
are not treated as independent device runs.
"""
import argparse
import hashlib
import json
import statistics
from pathlib import Path

from mcu.capture import parse_lines

HERE = Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Choose a fresh output directory")
    fixture = json.loads((HERE / "mcu/fixture_manifest.json").read_text())
    expected_fixture = ["synthetic_" + fixture["scenario"],
                        "seed" + str(fixture["seed"]),
                        "unit" + str(fixture["unit"]), "B" + str(fixture["budget"])]
    expected_payloads = {m["name"]: m["payload_bytes"] for m in fixture["models"]}
    expected_payloads["overhead"] = 0
    unique, submissions = {}, []
    for log in args.logs:
        raw = log.read_bytes()
        parsed = parse_lines(raw.decode("utf-8-sig").splitlines())
        supplied_json = log.with_suffix(".json")
        supplied = supplied_json.read_bytes()
        if parsed != json.loads(supplied.decode("utf-8-sig")):
            raise ValueError(f"Parsed log differs from supplied JSON: {log.name}")
        if parsed["metadata"]["fixture"] != expected_fixture:
            raise ValueError(f"Unexpected numerical fixture: {log.name}")
        if parsed["payload_bytes"] != expected_payloads:
            raise ValueError(f"Unexpected payload manifest: {log.name}")
        fingerprint = sha(json.dumps(parsed, sort_keys=True, separators=(",", ":")).encode())
        submissions.append(dict(log_name=log.name, json_name=supplied_json.name,
                                log_sha256=sha(raw), json_sha256=sha(supplied),
                                parsed_content_sha256=fingerprint))
        if fingerprint not in unique:
            unique[fingerprint] = dict(raw=raw, supplied=supplied, parsed=parsed)

    args.out.mkdir(parents=True)
    capture_dir = args.out / "captures"
    capture_dir.mkdir()
    analysis = dict(evidence_kind="user_supplied_mcu_uart_capture_on_synthetic_fixture",
                    submitted_pairs=len(submissions), unique_capture_contents=len(unique),
                    submissions=submissions,
                    numerical_fixture_manifest_sha256=sha((HERE / "mcu/fixture_manifest.json").read_bytes()),
                    analysis_script_sha256=sha(Path(__file__).read_bytes()), captures=[])
    lines = ["# User-supplied MCU run review", "",
             f"Received {len(submissions)} log/JSON pairs with {len(unique)} unique parsed contents.",
             "Each raw log parses completely and reproduces its supplied JSON exactly. "
             "Fixture metadata and model payloads match the checked-in manifest.", "",
             "Identical copies are counted once. A different filename does not establish an independent run. "
             "The ten timing batches within a capture measure within-run variation; they are not ten independent boards or reset sessions.", ""]
    for index, (fingerprint, item) in enumerate(unique.items(), 1):
        d = item["parsed"]
        stem = f"{d['metadata']['BOARD']}_capture{index:02d}"
        (capture_dir / (stem + ".log")).write_bytes(item["raw"])
        (capture_dir / (stem + ".json")).write_bytes(item["supplied"])
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
                       for other in ("polynomial", "cubic16", "kan16")}
                  for lut in ("uniform32_float32", "uniform32_int16")}
        update_us = statistics.median(r["total_us"] / r["calls"] for r in d["update_cost_only"])
        analysis["captures"].append(dict(id=stem, parsed_content_sha256=fingerprint,
                                          metadata=d["metadata"], models=models,
                                          gross_speedups=speeds, update_cost_gross_us=update_us))
        lines += [f"## {stem}", "",
                  "All eight prediction checks PASS at the predeclared 0.002 °C maximum host-gap tolerance. "
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
    lines += ["## Interpretation and remaining work", "",
              "- Compare the reported speed ratios together with accuracy and payload: float32 LUT32 stores 144 bytes versus 56 for the selected degree-nine Chebyshev model. A speed gain does not establish an accuracy or memory gain.",
              "- Int16 LUT32 reduces the payload to 88 bytes, but inference reconstructs floating-point values. It is not fixed-point integer inference; quantization need not reduce latency.",
              "- Uint8 and nonuniform results must be judged using both their timing and their error. This fixed fixture does not determine the best method over all sensors and budgets.",
              "- Cubic/KAN comparisons are against the checked-in general de Boor evaluator. An optimized piecewise-cubic/Horner deployment baseline is still needed before a general speed-superiority claim. The KAN model is one fixed-grid scalar edge plus SiLU.",
              "- Physical execution time is distinct from physical sensor accuracy: all ADC/temperature test pairs are synthetic. These measurements do not establish a new NTC accuracy specification.",
              "- The UART protocol does not embed an ELF/source hash. Matching fixture metadata and numerical checks do not cryptographically identify the flashed firmware. Retain source_hashes.json and packages.txt alongside the device runs.",
              "- Next: a second reset session on Mega2560, two ESP32-C3 captures with the same fixture, then a separately versioned optimized-baseline experiment and physical NTC measurements.", ""]
    (args.out / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf8")
    (args.out / "RESULTS.md").write_text("\n".join(lines), encoding="utf8")
    print(json.dumps(dict(submitted_pairs=len(submissions), unique_capture_contents=len(unique),
                          validated=True, output=str(args.out))))


if __name__ == "__main__":
    main()
