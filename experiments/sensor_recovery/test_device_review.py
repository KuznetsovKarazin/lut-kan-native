"""Tamper checks for capture provenance; all fixtures are artificial."""
import json

import pytest

from mcu.capture import parse_lines
from review_device_runs import sha, verify_session
from test_capture import protocol_fixture


def evidence():
    token="a"*32
    lines=protocol_fixture()
    lines[1:3]=["BOARD,esp32c3","CPU_HZ,160000000"]
    lines[1:1]=["SESSION,"+token,"TRANSPORT,usb-jtag-poll-command-v3"]
    lines[:0]=["READY,SENSOR_USB_V3,esp32c3","ECHO,"+token,"ACCEPTED,"+token]
    raw=("\r\n".join(lines)+"\r\n").encode()
    parsed=parse_lines(lines);supplied=json.dumps(parsed).encode()
    s=dict(session_id=token,mode="command_v3",status="validated",usb_phase="complete",
           raw_sha256=sha(raw),result_sha256=sha(supplied),raw_bytes=len(raw),
           start_utc="2026-11-08T00:00:00+00:00",end_utc="2026-11-08T00:00:10+00:00",
           run_accepted_utc="2026-11-08T00:00:04+00:00",
           tx_commands=[dict(text=text,utc=f"2026-11-08T00:00:0{i+1}+00:00")
                        for i,text in enumerate(["H","Q,"+token,"R,"+token])])
    return raw,supplied,parsed,s


def test_complete_session_and_legacy_without_sidecar():
    raw,supplied,parsed,s=evidence()
    result=verify_session(raw,supplied,parsed,json.dumps(s).encode())
    assert result["hashes_and_identity_verified"] and result["duration_s"]==10
    assert result["capture_script_matches_current"] is False
    legacy=parse_lines(protocol_fixture())
    assert verify_session(b"",b"",legacy,None) is None


@pytest.mark.parametrize("change",["raw_hash","result_hash","token","repeat_run","device_ack","timestamps","missing_sidecar"])
def test_reject_inconsistent_acquisition_evidence(change):
    raw,supplied,parsed,s=evidence()
    if change=="raw_hash":s["raw_sha256"]="0"*64
    elif change=="result_hash":s["result_sha256"]="0"*64
    elif change=="token":s["session_id"]="b"*32
    elif change=="repeat_run":s["tx_commands"].append(s["tx_commands"][-1])
    elif change=="device_ack":
        raw=raw.replace(b"ACCEPTED,"+b"a"*32,b"ACCEPTED,"+b"b"*32)
        s["raw_sha256"]=sha(raw)
    elif change=="timestamps":s["run_accepted_utc"]="2026-11-08T00:00:00+00:00"
    sidecar=None if change=="missing_sidecar" else json.dumps(s).encode()
    with pytest.raises(ValueError):verify_session(raw,supplied,parsed,sidecar)
