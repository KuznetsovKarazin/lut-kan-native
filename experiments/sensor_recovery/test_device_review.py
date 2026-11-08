"""Tamper checks for capture provenance; all fixtures are artificial."""
import json
import sys

import pytest

from mcu.capture import parse_lines
from review_device_runs import sha, verify_session
from test_capture import protocol_fixture
import review_device_runs as review


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


@pytest.mark.parametrize("identity,retained,identified",[("distinct",2,2),("same",1,1),("same_reformatted",1,1),("absent",1,0)])
def test_identical_legacy_data_preserves_distinct_host_acquisitions(tmp_path,monkeypatch,identity,retained,identified):
    """Artificial timing records; separate sidecars must survive deduplication."""
    manifest=json.loads((review.HERE/"mcu/fixture_manifest.json").read_text())
    payloads={"overhead":0,**{m["name"]:m["payload_bytes"] for m in manifest["models"]}}
    lines=protocol_fixture()
    lines[0]=f"SENSOR_BENCH_V1,synthetic_{manifest['scenario']},seed{manifest['seed']},unit{manifest['unit']},B{manifest['budget']}"
    lines=[f"PAYLOAD,{x.split(',')[1]},{payloads[x.split(',')[1]]}" if x.startswith("PAYLOAD,") else x for x in lines]
    raw=("\n".join(lines)+"\n").encode();supplied=json.dumps(parse_lines(lines)).encode()
    logs=[]
    for i in (1,2):
        log=tmp_path/f"artificial_{i}.log";log.write_bytes(raw);log.with_suffix(".json").write_bytes(supplied);logs.append(str(log))
        if identity!="absent":
            s=dict(session_id=str(i if identity=="distinct" else 1)*32,mode="legacy_auto_start",status="validated",
                   start_utc="2026-11-08T00:00:00+00:00",end_utc="2026-11-08T00:00:01+00:00",
                   raw_sha256=sha(raw),result_sha256=sha(supplied),raw_bytes=len(raw))
            log.with_suffix(".session.json").write_text(json.dumps(s,indent=2 if identity=="same_reformatted" and i==2 else None))
    out=tmp_path/"review"
    monkeypatch.setattr(sys,"argv",["review",*logs,"--out",str(out)])
    review.main();result=json.loads((out/"analysis.json").read_text())
    assert result["unique_capture_contents"]==1
    assert result["retained_capture_observations"]==len(result["captures"])==retained
    assert result["identified_session_count"]==identified
    assert len(list((out/"captures").glob("*.session.json")))==identified
    if identity=="distinct":
        assert result["repeatability"]["mega2560"]["captures"]==2
        assert result["repeatability"]["mega2560"]["max_relative_range_percent"]==0
        assert {c["session"]["session_id"] for c in result["captures"]}=={"1"*32,"2"*32}


def test_same_id_with_conflicting_sidecar_is_rejected(tmp_path,monkeypatch):
    manifest=json.loads((review.HERE/"mcu/fixture_manifest.json").read_text())
    payloads={"overhead":0,**{m["name"]:m["payload_bytes"] for m in manifest["models"]}}
    lines=protocol_fixture();lines[0]=f"SENSOR_BENCH_V1,synthetic_{manifest['scenario']},seed{manifest['seed']},unit{manifest['unit']},B{manifest['budget']}"
    lines=[f"PAYLOAD,{x.split(',')[1]},{payloads[x.split(',')[1]]}" if x.startswith("PAYLOAD,") else x for x in lines]
    raw=("\n".join(lines)+"\n").encode();supplied=json.dumps(parse_lines(lines)).encode();logs=[]
    for i in (1,2):
        log=tmp_path/f"artificial_{i}.log";log.write_bytes(raw);log.with_suffix(".json").write_bytes(supplied);logs.append(str(log))
        s=dict(session_id="1"*32,mode="legacy_auto_start",status="validated",raw_sha256=sha(raw),result_sha256=sha(supplied),raw_bytes=len(raw),
               start_utc=f"2026-11-08T00:00:0{i}+00:00",end_utc="2026-11-08T00:00:03+00:00")
        log.with_suffix(".session.json").write_text(json.dumps(s))
    monkeypatch.setattr(sys,"argv",["review",*logs,"--out",str(tmp_path/"review")])
    with pytest.raises(ValueError,match="acquisition metadata"):review.main()
