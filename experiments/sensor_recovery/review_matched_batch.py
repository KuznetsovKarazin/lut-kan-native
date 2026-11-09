"""Audit an original matched-MCU ZIP without changing its evidence bytes."""
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat
import statistics
import subprocess
import zipfile

from mcu.capture_matched import parse_lines, STATIC, ONLINE, MODES, MEGA_HELLO_ERROR_BUDGET
from mcu.run_matched import source_hashes

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def instant(value):
    result = datetime.fromisoformat(value)
    require(result.tzinfo is not None, "Timestamp lacks timezone")
    return result


def check_capture(files, entry, board, capture_script_sha256):
    name = entry["result"]
    require(re.fullmatch(r"run\d{2}\.json", name), "Invalid capture filename")
    stem = name[:-5]
    raw = files[stem + ".log"]
    data = json.loads(files[name])
    side = json.loads(files[stem + ".session.json"])
    require(parse_lines(raw.decode("utf-8-sig").splitlines()) == data, "Raw/JSON mismatch")
    token = data["metadata"]["SESSION"]
    require(token == entry["session_id"] == side["session_id"], "Session ID mismatch")
    require(data["metadata"]["BOARD"] == side["board_requested"] == board, "Board mismatch")
    require(side["status"] == "validated" and side["usb_phase"] == "complete", "Incomplete session")
    require(side["mode"] == "command_matched_v1", "Wrong capture mode")
    require(side["capture_script_sha256"] == capture_script_sha256, "Capture script hash mismatch")
    require(side["raw_bytes"] == len(raw), "Raw length mismatch")
    for key, content in (("raw_sha256", raw), ("result_sha256", files[name])):
        require(entry[key] == side[key] == digest(content), "Capture hash mismatch: " + key)
    commands = [r["text"] for r in side["tx_commands"]]
    require(1 <= len(commands)-2 <= 15 and commands[-2:] == ["Q," + token, "R," + token]
            and all(c == "H" for c in commands[:-2]), "Wrong commands")
    rejected = [line for line in raw.decode("utf-8-sig").splitlines() if line.startswith("ERROR,")]
    saved_rejections = side.get("pre_ready_rejections", [])
    require(rejected == [r["line"] for r in saved_rejections], "Startup rejection record mismatch")
    if rejected:
        require(board == "mega2560" and side.get("hello_error_budget") == MEGA_HELLO_ERROR_BUDGET,
                "Unexpected startup recovery policy")
        require(all(instant(side["start_utc"]) <= instant(r["utc"]) <= instant(side["tx_commands"][-2]["utc"])
                    for r in saved_rejections), "Startup rejection time outside hello phase")
    times = [instant(x) for x in [side["start_utc"], *[r["utc"] for r in side["tx_commands"]],
                                 side["run_accepted_utc"], side["end_utc"]]]
    require(times == sorted(times) and times[0] < times[-1], "Session time order mismatch")
    console = files[stem + ".console.txt"].decode("utf8")
    require("\n".join(raw.decode("utf-8-sig").splitlines()) in "\n".join(console.splitlines()),
            "Console omits or changes raw capture")
    require("Validated matched capture:" in console, "Console lacks validation result")
    reference = json.loads((HERE / "mcu/matched_host_reference.json").read_text(encoding="utf8"))
    for method, predicted in zip(STATIC, reference["static_expected"]):
        errors = [abs(p - y) for p, y in zip(predicted, reference["truth"])]
        expected = [statistics.mean(errors), math.sqrt(statistics.mean(e * e for e in errors)), max(errors)]
        for key, value in zip(("mae_C", "rmse_C", "max_C"), expected):
            require(abs(data["accuracy"][method][key] - value) <= .002, "Static metric differs from host reference")
    expected_online = {(s["mode"], m): s["reference_final_metrics"][i]
                       for s in reference["streams"] for i, m in enumerate(ONLINE)}
    for row in data["update_accuracy"]:
        for key, value in zip(("mae_C", "rmse_C", "max_C"), expected_online[row["drift"], row["method"]]):
            require(abs(row[key] - value) <= .002, "Final metric differs from host reference")
    return data, side


def timing_stats(rows):
    values = [r["total_us"] / r["calls"] for r in rows]
    return dict(repeats=len(values), median_us=statistics.median(values), min_us=min(values), max_us=max(values))


def audit_archive(path):
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), "Duplicate ZIP entry")
        for item in archive.infolist():
            p = PurePosixPath(item.filename)
            require(not p.is_absolute() and ".." not in p.parts and "\\" not in item.filename
                    and not stat.S_ISLNK(item.external_attr >> 16), "Unsafe ZIP entry")
        require(archive.testzip() is None, "ZIP CRC failure")
        files = {name: archive.read(name) for name in names}
    batch = json.loads(files["batch.json"])
    require(batch["status"] == "validated" and batch["stage"] == "complete", "Batch is not complete")
    require(len(batch["captures"]) == batch["requested_runs"] >= 2, "Wrong capture count")
    require(batch["environment"] == batch["board"] + "_matched", "Wrong environment")
    require(batch["source_before"] == batch["source_after"], "Sources changed during batch")
    commit = batch["git_commit"]
    require(re.fullmatch(r"[0-9a-f]{40}", commit), "Invalid source commit")
    sources = {name.replace("\\", "/"): value for name, value in batch["source_before"].items()}
    require(set(sources) == {p.replace("\\", "/") for p in source_hashes()}, "Missing/extra source records")
    for name, value in sources.items():
        pinned = subprocess.check_output(["git", "show", commit + ":" + name], cwd=ROOT)
        require(digest(pinned.replace(b"\r\n", b"\n")) == value["lf_sha256"], "Pinned source mismatch: " + name)
    images = batch["built_artifacts"]
    needed = {"firmware.elf", "firmware.hex" if batch["board"] == "mega2560" else "firmware.bin"}
    require(needed <= set(images), "Missing firmware images")
    for name, record in images.items():
        require(record["archive_path"] == "firmware/" + name, "Image path mismatch")
        data = files[record["archive_path"]]
        require(len(data) == record["bytes"] and digest(data) == record["sha256"], "Image hash/length mismatch")
    upload = files["upload.log"].decode("utf8")
    require(re.search(re.escape(batch["environment"]) + r"\s+SUCCESS", upload), "Upload lacks success marker")
    require("UnicodeEncodeError" not in files["packages.txt"].decode("utf8"), "Package inventory failed")
    runs = []
    sessions = []
    for index, entry in enumerate(batch["captures"], 1):
        require(entry["result"] == f"run{index:02d}.json", "Nonsequential capture filenames")
        data, side = check_capture(files, entry, batch["board"], sources["experiments/sensor_recovery/mcu/capture_matched.py"]["raw_sha256"])
        require(instant(batch["start_utc"]) <= instant(side["start_utc"]) < instant(side["end_utc"])
                <= instant(batch["end_utc"]), "Capture outside batch interval")
        require(side["port"] == batch["port"], "Capture port differs from batch")
        require(side["session_id"] not in {s["session_id"] for s in sessions}, "Repeated session ID")
        if sessions:
            require(instant(sessions[-1]["end_utc"]) <= instant(side["start_utc"]), "Overlapping sessions")
        runs.append(data)
        sessions.append(side)
    expected_run_files = {f"run{i:02d}{suffix}" for i in range(1, len(runs) + 1)
                          for suffix in (".json", ".log", ".session.json", ".console.txt")}
    require({n for n in names if n.startswith("run")} == expected_run_files, "Unexpected capture files")
    static = [dict(method=m, payload_bytes=144,
                   sessions=[dict(session_id=s["session_id"], **timing_stats([t for t in r["timings"] if t["method"] == m]),
                                  **r["accuracy"][m]) for r, s in zip(runs, sessions)]) for m in STATIC]
    online = [dict(drift=d, method=m, **runs[0]["update_storage"][m],
                   sessions=[dict(session_id=s["session_id"], **timing_stats([t for t in r["update_timings"] if t["method"] == m and t["drift"] == d]),
                                  **next(t for t in r["update_accuracy"] if t["method"] == m and t["drift"] == d)) for r, s in zip(runs, sessions)])
              for d in MODES for m in ONLINE]
    overhead = [dict(session_id=s["session_id"], inference=timing_stats([t for t in r["timings"] if t["method"] == "overhead"]),
                     update={d: timing_stats([t for t in r["update_overhead"] if t["drift"] == d]) for d in MODES}) for r, s in zip(runs, sessions)]
    report = dict(status="validated", archive=path.name, archive_bytes=path.stat().st_size, archive_sha256=digest(path.read_bytes()),
                  source_commit=commit, board=batch["board"], start_utc=batch["start_utc"], end_utc=batch["end_utc"],
                  source_checks=sources, image_checks=images, source_before_equals_after=True,
                  files={n: dict(bytes=len(b), sha256=digest(b)) for n, b in files.items()},
                  sessions=[dict(session_id=s["session_id"], start_utc=s["start_utc"], end_utc=s["end_utc"],
                                 pre_ready_rejection_count=len(s.get("pre_ready_rejections", [])),
                                 max_prediction_gap_C=max(v["max_host_gap_C"] for v in r["accuracy"].values()),
                                 max_checkpoint_gap_C=max(v["max_gap_C"] for v in r["state_checks"]),
                                 max_timed_state_gap_C=max(v["final_state_gap_C"] for v in r["update_timings"]),
                                 counts={k: len(r[k]) for k in ("accuracy", "timings", "state_checks", "update_accuracy", "update_timings", "update_overhead")})
                            for r, s in zip(runs, sessions)], static=static, online=online, overhead=overhead,
                  notes="Gross medians within each acquisition; same-board sessions are not independent boards. Synthetic labelled fixture; no physical sensor accuracy.")
    return report, files


def save_review(archive, out):
    report, files = audit_archive(archive)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    report["redactions"] = []
    for name, raw in files.items():
        if name.startswith("firmware/"):
            continue  # Actual images remain in the immutable submitted ZIP.
        target = name
        if name in ("batch.json", "upload.log") or name.endswith(".console.txt"):
            text = raw.decode("utf8")
            text, mac = re.subn(r"(?:[0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}", "[DEVICE_MAC_REDACTED]", text)
            text, user = re.subn(r"C:\\+Users\\+[^\\\s\"]+", lambda m: m[0].replace(m[0].split("\\")[-1], "[USER]"), text)
            raw = text.encode("utf8")
            if mac or user:
                p = Path(name)
                target = p.stem + ".redacted" + p.suffix
                report["redactions"].append(dict(original=name, saved=target, sha256=digest(raw), mac_count=mac, user_path_count=user))
        (out / target).write_bytes(raw)
    (out / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = save_review(args.archive, args.out)
    print(f"Validated {len(result['sessions'])} {result['board']} sessions; saved review to {args.out}")
