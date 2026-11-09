"""Validate real saved capture evidence and reject self-consistent alterations."""
import copy
import json
from pathlib import Path

import pytest

from mcu.capture_matched import parse_lines
from review_matched_batch import check_capture, digest

SAVED = Path(__file__).resolve().parent / "results/c3_matched_20261108"


def saved_capture(board="esp32c3", run=1):
    directory = SAVED if board == "esp32c3" else SAVED.parent / "mega_matched_20261108"
    batch = json.loads((directory / "batch.redacted.json").read_bytes())
    files = {p.name: p.read_bytes() for p in directory.glob(f"run{run:02d}.*")}
    sources = {p.replace("\\", "/"): v for p, v in batch["source_before"].items()}
    script = sources["experiments/sensor_recovery/mcu/capture_matched.py"]["raw_sha256"]
    return files, copy.deepcopy(batch["captures"][run-1]), script


@pytest.mark.parametrize("board", ["esp32c3", "mega2560"])
@pytest.mark.parametrize("run", [1, 2])
def test_saved_device_capture_is_reparsed_and_matches_host_reference(board, run):
    files, entry, script = saved_capture(board, run)
    result, session = check_capture(files, entry, board, script)
    assert len(result["update_timings"]) == 120
    assert session["session_id"] == entry["session_id"]
    assert len(session.get("pre_ready_rejections", [])) == (1 if board == "mega2560" else 0)


@pytest.mark.parametrize("verb", ["H", "Q", "R"])
def test_audit_allows_repeated_hello_but_not_probe_or_run_commands(verb):
    files, entry, script = saved_capture()
    side = json.loads(files["run01.session.json"])
    repeated = copy.deepcopy(side["tx_commands"][0])
    repeated["text"] = verb if verb == "H" else verb + "," + side["session_id"]
    side["tx_commands"].insert(1, repeated)
    files["run01.session.json"] = json.dumps(side).encode("utf8")
    if verb == "H":
        check_capture(files, entry, "esp32c3", script)
    else:
        with pytest.raises(ValueError, match="Wrong commands"):
            check_capture(files, entry, "esp32c3", script)


@pytest.mark.parametrize("fault", ["side_id", "script_hash", "time_order", "console", "result_hash", "false_metric"])
def test_reject_changed_evidence(fault):
    files, entry, script = saved_capture()
    side = json.loads(files["run01.session.json"])
    if fault == "side_id":
        side["session_id"] = "0" * 32
    elif fault == "script_hash":
        side["capture_script_sha256"] = "0" * 64
    elif fault == "time_order":
        side["end_utc"] = side["start_utc"]
    elif fault == "console":
        files["run01.console.txt"] = b"Truncated console\n"
    elif fault == "result_hash":
        side["result_sha256"] = "0" * 64
    else:
        # Forge a passing accuracy row and recompute all associated hashes/JSON.
        # The independent reference comparison must still reject its wrong MAE.
        old = files["run01.log"].decode("utf8")
        lines = old.splitlines()
        for i, line in enumerate(lines):
            if line.startswith("ACCURACY,lut32_float32,"):
                row = line.split(","); row[2] = "1.0000000"; lines[i] = ",".join(row)
        new = "\r\n".join(lines) + "\r\n"
        files["run01.log"] = new.encode("utf8")
        files["run01.json"] = json.dumps(parse_lines(lines)).encode("utf8")
        files["run01.console.txt"] = files["run01.console.txt"].replace(old.encode("utf8"), new.encode("utf8"))
        side["raw_bytes"] = len(files["run01.log"])
        for key, name in [("raw_sha256", "run01.log"), ("result_sha256", "run01.json")]:
            side[key] = entry[key] = digest(files[name])
    files["run01.session.json"] = json.dumps(side).encode("utf8")
    message = "Static metric differs" if fault == "false_metric" else None
    with pytest.raises(ValueError, match=message):
        check_capture(files, entry, "esp32c3", script)
