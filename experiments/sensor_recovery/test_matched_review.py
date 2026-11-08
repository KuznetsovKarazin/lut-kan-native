"""Validate real saved capture evidence and reject self-consistent alterations."""
import copy
import json
from pathlib import Path

import pytest

from mcu.capture_matched import parse_lines
from review_matched_batch import check_capture, digest

SAVED = Path(__file__).resolve().parent / "results/c3_matched_20261108"


def saved_capture():
    batch = json.loads((SAVED / "batch.redacted.json").read_bytes())
    files = {p.name: p.read_bytes() for p in SAVED.glob("run01.*")}
    sources = {p.replace("\\", "/"): v for p, v in batch["source_before"].items()}
    script = sources["experiments/sensor_recovery/mcu/capture_matched.py"]["raw_sha256"]
    return files, copy.deepcopy(batch["captures"][0]), script


def test_saved_device_capture_is_reparsed_and_matches_host_reference():
    files, entry, script = saved_capture()
    result, session = check_capture(files, entry, "esp32c3", script)
    assert len(result["update_timings"]) == 120
    assert session["session_id"] == "646fff2706d34a228f77cc5fc0b4348d"


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
