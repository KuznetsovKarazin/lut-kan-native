"""Reproduce this batch's archive/source/session audit from the original ZIP.

Usage: python verify_batch.py /path/to/mega_opt_20261108_012910_658.zip
Prints JSON; does not alter the archive, stored captures or firmware.
"""
import json
import re
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[4]
HERE = ROOT / "experiments/sensor_recovery"
sys.path.insert(0, str(HERE))
from mcu.capture import parse_lines
from review_device_runs import sha, verify_session

SOURCE_COMMIT = "348c0e589cc44ff36a2abbfa916e99aac2a0158a"


def timestamp(value):
    t = datetime.fromisoformat(value)
    if t.tzinfo is None:
        raise ValueError("Timestamp lacks timezone")
    return t


def check(condition, message):
    if not condition:
        raise ValueError(message)


def audit(archive):
    with zipfile.ZipFile(archive) as z:
        check(z.testzip() is None, "ZIP CRC failure")
        check(len(z.namelist()) == len(set(z.namelist())), "Duplicate ZIP entries")
        expected = {"batch.json", "upload.log"} | {
            f"run{i:02d}{suffix}" for i in (1, 2)
            for suffix in (".json", ".log", ".session.json", ".console.txt")
        }
        check(set(z.namelist()) == expected, "Unexpected or missing ZIP entries")
        b = json.loads(z.read("batch.json"))
        check(b["status"] == "validated" and b["stage"] == "complete", "Incomplete batch")
        check(b["board"] == "mega2560" and b["environment"] == "mega2560_opt", "Wrong board/environment")
        check(b["requested_runs"] == len(b["captures"]) == 2, "Wrong capture count")
        check(b["git_commit"] == SOURCE_COMMIT, "Unexpected source commit")
        check(b["source_before"] == b["source_after"], "Sources changed during batch")
        source_matches = {}
        for name, hashes in b["source_before"].items():
            name = name.replace("\\", "/")
            path = PurePosixPath(name)
            check(not path.is_absolute() and ".." not in path.parts, "Unsafe source path")
            raw = subprocess.check_output(["git", "show", SOURCE_COMMIT + ":" + name], cwd=ROOT)
            lf = raw.replace(b"\r\n", b"\n")
            check(sha(lf) == hashes["lf_sha256"], "Normalized source hash mismatch: " + name)
            check(hashes["raw_sha256"] in {sha(lf), sha(lf.replace(b"\n", b"\r\n"))}, "Raw source hash mismatch: " + name)
            source_matches[name] = hashes["lf_sha256"]
        expected_sources = {"experiments/sensor_recovery/mcu/" + p for p in (
            "platformio.ini", "capture.py", "run_suite.py", "fixture_manifest.json", "optimized_manifest.json",
            "include/command_protocol.h", "include/fixture.h", "include/kernels.h", "include/models.h",
            "include/optimized_coefficients.h", "include/optimized_spline.h", "include/usb_bench_serial.h", "src/main.cpp")}
        check(set(source_matches) == expected_sources, "Incomplete source inventory")
        manifest = json.loads(subprocess.check_output([
            "git", "show", SOURCE_COMMIT + ":experiments/sensor_recovery/mcu/optimized_manifest.json"], cwd=ROOT))
        payloads = {"overhead": 0, **{m["name"]: m["payload_bytes"] for m in manifest["models"]}}
        sessions = []
        last_end = timestamp(b["start_utc"])
        for i, record in enumerate(b["captures"], 1):
            stem = f"run{i:02d}"
            check(record["result"] == stem + ".json", "Unexpected capture order")
            raw, supplied = z.read(stem + ".log"), z.read(stem + ".json")
            parsed = parse_lines(raw.decode("utf-8-sig").splitlines())
            check(parsed == json.loads(supplied), "Log/JSON mismatch")
            check(parsed["metadata"]["SUITE"] == "spline_optimized_v1" and parsed["metadata"]["BOARD"] == b["board"], "Wrong capture suite/board")
            check(parsed["metadata"]["fixture"] == ["synthetic_ntc_frontend", "seed20261108", "unit0", "B75"], "Wrong fixture")
            check(parsed["payload_bytes"] == payloads, "Payload mismatch")
            sidecar = z.read(stem + ".session.json")
            s = verify_session(raw, supplied, parsed, sidecar)
            check(s["capture_script_matches_current"], "Unexpected capture script")
            check(s["mode"] == "legacy_auto_start" and json.loads(sidecar)["tx_commands"] == [], "Unexpected Mega transport")
            check(json.loads(sidecar)["port"] == b["port"], "Session/batch port mismatch")
            check(record["session_id"] == s["session_id"] and record["raw_sha256"] == sha(raw) and record["result_sha256"] == sha(supplied), "Batch capture record mismatch")
            check(json.loads(sidecar)["capture_script_sha256"] == b["source_before"]["experiments\\sensor_recovery\\mcu\\capture.py"]["raw_sha256"], "Sidecar/source identity mismatch")
            check(last_end <= timestamp(s["start_utc"]) <= timestamp(s["end_utc"]) <= timestamp(b["end_utc"]), "Overlapping or out-of-batch sessions")
            last_end = timestamp(s["end_utc"])
            console = z.read(stem + ".console.txt").decode("utf8").replace("\r\n", "\n")
            check(raw.decode("utf8").replace("\r\n", "\n") in console, "Console does not contain raw capture")
            sessions.append(s)
        check(len({s["session_id"] for s in sessions}) == 2, "Repeated acquisition ID")
        upload = z.read("upload.log").decode("utf8")
        check("[SUCCESS]" in upload and "1 succeeded" in upload and "bytes of flash verified" in upload, "Upload did not complete")
        check("Processing mega2560_opt" in upload and "Using manually specified: " + b["port"] in upload, "Upload environment/port mismatch")
        flash_written = int(re.search(r"avrdude: (\d+) bytes of flash written", upload)[1])
        flash_verified = int(re.search(r"avrdude: (\d+) bytes of flash verified", upload)[1])
        check(flash_written == flash_verified, "Written/verified flash size differs")
        check(set(b["built_artifacts"]) == {"firmware.elf", "firmware.hex"}, "Missing AVR image metadata")
        for artifact in b["built_artifacts"].values():
            check(artifact["bytes"] > 0 and re.fullmatch(r"[0-9a-f]{64}", artifact["sha256"]), "Malformed artifact metadata")
        ram = int(re.search(r"RAM:.*used (\d+) bytes", upload)[1])
        flash = int(re.search(r"Flash:.*used (\d+) bytes", upload)[1])
        check(flash == flash_written, "Linked/uploaded flash size differs")
        return dict(
            archive_name=archive.name, archive_bytes=archive.stat().st_size, archive_sha256=sha(archive.read_bytes()),
            archive_crc_and_members_valid=True, batch_complete=True,
            source_commit=SOURCE_COMMIT, verified_source_files=source_matches,
            sources_unchanged_during_batch=True, sequential_sessions_within_batch=True,
            sessions=sessions, reported_build_artifacts=b["built_artifacts"],
            image_hashes_independently_recomputed=False,
            image_hash_limitation="The archive records the host image hashes but does not contain the firmware images; the device protocol does not attest them.",
            upload_success_and_flash_verification_checked=True,
            host_recorded_session_ids_not_echoed_by_device=True,
            identical_raw_capture_contents=z.read("run01.log") == z.read("run02.log"),
            identical_result_json_contents=z.read("run01.json") == z.read("run02.json"),
            windows_full_harness=dict(flash_bytes=flash, static_ram_bytes=ram),
            archive_member_sha256={n: sha(z.read(n)) for n in sorted(expected)},
            audit_script_sha256=sha(Path(__file__).read_bytes()),
            capture_parser_sha256=sha((HERE / "mcu/capture.py").read_bytes()),
            session_verifier_sha256=sha((HERE / "review_device_runs.py").read_bytes()))


if __name__ == "__main__":
    print(json.dumps(audit(Path(sys.argv[1])), indent=2))
