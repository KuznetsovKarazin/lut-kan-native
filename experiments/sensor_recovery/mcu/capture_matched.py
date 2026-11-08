"""Capture the matched suite with device-acknowledged IDs on both boards."""
import argparse,csv,hashlib,json,math,re,statistics,uuid
from pathlib import Path
try:
    from .capture import capture_stream,open_serial,utc_now
except ImportError:
    from capture import capture_stream,open_serial,utc_now

HERE=Path(__file__).resolve().parent
STATIC=("lut32_float32","lut60_int16","cubic32_unrolled","horner8")
ONLINE=("lut_nlms","lut_projected_nlms","cubic_nlms","cubic_projected_nlms")
MODES=("none","step","ramp")
CHECKS=(0,127,128,160,192,256,320,384)


def parse_lines(lines):
    meta={};payload={};storage={};accuracy={};timing={};states={};ua={};ut={};overhead={}
    started=complete=False;prefix=[]
    def numbers(values,nonnegative=True):
        v=list(map(float,values))
        if not all(math.isfinite(x) and (x>=0 or not nonnegative) for x in v):raise ValueError("Nonfinite/invalid numeric field")
        return v
    def unique(d,key,value):
        if key in d:raise ValueError("Duplicate record: "+str(key))
        d[key]=value
    def size(row,n):
        if len(row)!=n:raise ValueError("Malformed "+row[0]+" record")
    def timed(row,index,calls):
        rep,count,us=map(int,row[index:index+3]);value=numbers([row[index+3]],False)[0]
        if rep not in range(10) or count!=calls or us<=0:raise ValueError("Invalid timing dimensions")
        return dict(repeat=rep,calls=count,total_us=us,checksum=value)
    for line in lines:
        row=next(csv.reader([line.strip()]));
        if not row:continue
        tag=row[0]
        if tag=="SENSOR_BENCH_V1":
            if started or complete:raise ValueError("Multiple benchmark starts/reset")
            if row[1:]!=["synthetic_ntc_frontend","seed2026110801","unit0","B75"]:raise ValueError("Wrong fixture header")
            started=True;continue
        if not started:
            prefix.append(row);continue
        if complete:raise ValueError("Trailing records after END")
        if tag=="END_SENSOR_BENCH":size(row,1);complete=True;continue
        if tag in ("SUITE","SESSION","BOARD","TRANSPORT","CPU_HZ","FIXTURE_SHA256","PRECISION"):
            size(row,2);unique(meta,tag,row[1])
        elif tag=="NOTE":continue
        elif tag=="PAYLOAD":size(row,3);unique(payload,row[1],int(row[2]))
        elif tag=="UPDATE_STORAGE":size(row,4);unique(storage,row[1],dict(persistent_bytes=int(row[2]),workspace_bytes=int(row[3])))
        elif tag=="ACCURACY":
            size(row,7);v=numbers(row[2:6])
            if v[-1]>.002 or row[-1]!="PASS":raise ValueError("Static parity failure")
            unique(accuracy,row[1],dict(zip(("mae_C","rmse_C","max_C","max_host_gap_C"),v)))
        elif tag=="TIME":
            size(row,6);item=timed(row,2,2048);unique(timing,(row[1],item["repeat"]),item)
        elif tag=="STATE":
            size(row,7);gap=numbers([row[4]])[0]
            if row[1] not in MODES or row[2] not in ONLINE:raise ValueError("Unknown state method/mode")
            order="MONOTONE" if "projected" in row[2] else "UNCONSTRAINED"
            if gap>.002 or row[5]!=order or row[6]!="PASS":raise ValueError("Online state parity/order failure")
            unique(states,(row[1],row[2],int(row[3])),dict(max_gap_C=gap,order=row[5]))
        elif tag=="UPDATE_ACCURACY":
            size(row,8);v=numbers(row[3:7])
            if v[-1]>.002 or row[-1]!="PASS":raise ValueError("Online final metric parity failure")
            unique(ua,(row[1],row[2]),dict(zip(("mae_C","rmse_C","max_C","max_host_metric_gap_C"),v)))
        elif tag=="UPDATE_TIME":
            size(row,9);item=timed(row,3,384);gap=numbers([row[7]])[0]
            if gap>.002 or row[8]!="PASS":raise ValueError("Timed trajectory failed final state gate")
            item["final_state_gap_C"]=gap;unique(ut,(row[1],row[2],item["repeat"]),item)
        elif tag=="UPDATE_OVERHEAD":
            size(row,6);item=timed(row,2,384);unique(overhead,(row[1],item["repeat"]),item)
        else:raise ValueError("Unknown benchmark record: "+tag)
    manifest=json.loads((HERE/"matched_manifest.json").read_text())
    expected_meta={"SUITE","SESSION","BOARD","TRANSPORT","CPU_HZ","FIXTURE_SHA256","PRECISION"}
    if not started or not complete or set(meta)!=expected_meta:raise ValueError("Incomplete matched benchmark")
    board=meta["BOARD"]
    hz={"mega2560":"16000000","esp32c3":"160000000"}
    transport={"mega2560":"uart-command-matched-v1","esp32c3":"usb-jtag-poll-command-matched-v1"}
    if board not in hz or meta["CPU_HZ"]!=hz[board] or meta["TRANSPORT"]!=transport[board]:raise ValueError("Board/clock/transport mismatch")
    if meta["SUITE"]!="matched_adaptation_v1" or meta["PRECISION"]!="float32" or meta["FIXTURE_SHA256"]!=manifest["fixture_sha256"]:raise ValueError("Suite/fixture/precision mismatch")
    if not re.fullmatch("[0-9a-f]{32}",meta["SESSION"]):raise ValueError("Invalid session ID")
    for tag,value in [("READY","SENSOR_MATCHED_V1,"+board),("ECHO",meta["SESSION"]),("ACCEPTED",meta["SESSION"])]:
        found=[r for r in prefix if r[0]==tag]
        if not found or any(",".join(r[1:])!=value for r in found) or (tag!="READY" and len(found)!=1):raise ValueError("Missing/mismatched "+tag)
    if payload!={n:144 for n in STATIC}:raise ValueError("Wrong static payloads")
    if storage!={n:dict(persistent_bytes=148,workspace_bytes=160 if "projected" in n else 0) for n in ONLINE}:raise ValueError("Wrong update storage")
    requirements=[(accuracy,set(STATIC)),(timing,{(m,r) for m in ("overhead",*STATIC) for r in range(10)}),
        (states,{(d,m,t) for d in MODES for m in ONLINE for t in CHECKS}),
        (ua,{(d,m) for d in MODES for m in ONLINE}),
        (ut,{(d,m,r) for d in MODES for m in ONLINE for r in range(10)}),
        (overhead,{(d,r) for d in MODES for r in range(10)})]
    if any(set(data)!=keys for data,keys in requirements):raise ValueError("Incomplete/extra method, mode, checkpoint or repeat")
    trows=[dict(method=k[0],**v) for k,v in sorted(timing.items())]
    urows=[dict(drift=k[0],method=k[1],**v) for k,v in sorted(ut.items())]
    return dict(metadata=meta,payload_bytes=payload,update_storage=storage,accuracy=accuracy,timings=trows,
        state_checks=[dict(drift=k[0],method=k[1],step=k[2],**v) for k,v in sorted(states.items())],
        update_accuracy=[dict(drift=k[0],method=k[1],**v) for k,v in sorted(ua.items())],update_timings=urows,
        update_overhead=[dict(drift=k[0],**v) for k,v in sorted(overhead.items())],
        inference_median_us={m:statistics.median(v["total_us"]/2048 for k,v in timing.items() if k[0]==m) for m in ("overhead",*STATIC)},
        update_median_us=[dict(drift=d,method=m,gross_us_per_update=statistics.median(v["total_us"]/384 for k,v in ut.items() if k[:2]==(d,m))) for d in MODES for m in ONLINE],
        notes="Real device timing only when produced from actual UART capture; synthetic numerical fixture; gross timing includes reads/checksum; no ADC acquisition or measured peak stack.")


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--board",choices=["mega2560","esp32c3"],required=True)
    p.add_argument("--port");p.add_argument("--out",type=Path,required=True);p.add_argument("--parse",type=Path)
    p.add_argument("--timeout",type=float,default=600);a=p.parse_args()
    a.out.parent.mkdir(parents=True,exist_ok=True)
    raw=a.out.with_suffix(".log");side=a.out.with_suffix(".session.json")
    if a.out.exists() or (not a.parse and (raw.exists() or side.exists())):p.error("Choose a new output name")
    if a.parse:
        result=parse_lines(a.parse.read_text(encoding="utf-8-sig").splitlines())
        if result["metadata"]["BOARD"]!=a.board:raise ValueError("Unexpected board")
        a.out.write_text(json.dumps(result,indent=2)+"\n");return
    if not a.port or a.timeout<=0:p.error("Supply a port and a positive timeout")
    token=uuid.uuid4().hex
    session=dict(session_id=token,board_requested=a.board,start_utc=utc_now(),port=a.port,
                 mode="command_matched_v1",status="capturing",tx_commands=[],capture_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    side.write_text(json.dumps(session,indent=2)+"\n");error=None
    try:
        with raw.open("xb") as output,open_serial(a.port,True) as serial:
            print("Checking device handshake and session ID. Do not press RESET or BOOT.",flush=True)
            lines=capture_stream(serial,output,True,token,a.timeout,session,emit=lambda s:print(s,flush=True),
                                 ready_line="READY,SENSOR_MATCHED_V1,"+a.board)
        result=parse_lines(lines)
        if result["metadata"]["BOARD"]!=a.board or result["metadata"]["SESSION"]!=token:raise ValueError("Board/session mismatch")
        a.out.write_text(json.dumps(result,indent=2)+"\n")
        session.update(status="validated",result_sha256=hashlib.sha256(a.out.read_bytes()).hexdigest())
    except (Exception,KeyboardInterrupt) as exc:
        error=exc;session.update(status="failed",error=str(exc),error_type=type(exc).__name__)
    finally:
        session["end_utc"]=utc_now()
        if raw.exists():session.update(raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),raw_bytes=raw.stat().st_size)
        side.write_text(json.dumps(session,indent=2)+"\n")
    if error:raise SystemExit("Capture stopped; raw log and sidecar retained: "+str(error))
    print("Validated matched capture:",a.out,flush=True)


if __name__=="__main__":main()
