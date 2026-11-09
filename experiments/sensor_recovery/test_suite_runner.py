"""Batch sequencing and failure stops using temporary artificial outputs only."""
import json
from pathlib import Path

import pytest

from mcu.run_suite import capture_runs,sha


def fake_success(command,log):
    target=Path(command[command.index("--out")+1])
    data=dict(metadata=dict(SUITE="spline_optimized_v1",BOARD="esp32c3"),accuracy={str(i):{} for i in range(10)})
    target.write_text(json.dumps(data));target.with_suffix(".log").write_bytes(b"TEST_INPUT_ONLY\n")
    session=dict(status="validated",session_id=target.stem,raw_sha256=sha(target.with_suffix(".log")),result_sha256=sha(target))
    target.with_suffix(".session.json").write_text(json.dumps(session))
    log.write_text("MOCK ONLY\n")


def test_two_runs_are_sequential_and_use_command_mode(tmp_path):
    calls=[];meta=dict(captures=[])
    def run(command,log):
        assert len(meta["captures"])==len(calls)
        calls.append(command);fake_success(command,log)
    capture_runs("esp32c3","COM_TEST",tmp_path,2,meta,runner=run)
    assert len(meta["captures"])==2
    assert all("--start-command" in c for c in calls)
    assert [c["result"] for c in meta["captures"]]==["run01.json","run02.json"]


def test_failed_first_capture_does_not_start_second(tmp_path):
    calls=[];meta=dict(captures=[])
    def run(command,log):calls.append(command);raise RuntimeError("MOCK disconnect")
    with pytest.raises(RuntimeError,match="disconnect"):
        capture_runs("esp32c3","COM_TEST",tmp_path,2,meta,runner=run)
    assert len(calls)==1 and not meta["captures"]


def test_wrong_suite_stops_batch(tmp_path):
    calls=[];meta=dict(captures=[])
    def run(command,log):
        calls.append(command);fake_success(command,log)
        target=Path(command[command.index("--out")+1]);data=json.loads(target.read_text());data["metadata"]["SUITE"]="wrong";target.write_text(json.dumps(data))
    with pytest.raises(ValueError,match="ten-model"):
        capture_runs("esp32c3","COM_TEST",tmp_path,2,meta,runner=run)
    assert len(calls)==1 and not meta["captures"]
