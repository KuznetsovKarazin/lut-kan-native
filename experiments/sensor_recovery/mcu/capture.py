"""Capture a real UART run; retain raw output and validate its completeness."""
import argparse,csv,hashlib,json,math,statistics,time,uuid
from datetime import datetime,timezone
from pathlib import Path

def parse_lines(lines):
    timings={};accuracy={};metadata={};complete=False;started=False;updates=[];payloads={}
    for line in lines:
        row=next(csv.reader([line.strip()]))
        if not row:continue
        if row[0]=="SENSOR_BENCH_V1":
            if started:raise ValueError("Multiple benchmark starts: possible board reset")
            started=True;metadata["fixture"]=row[1:]
        elif row[0]=="END_SENSOR_BENCH":
            if not started or complete:raise ValueError("Unexpected benchmark end")
            complete=True
        elif row[0] in ("BOARD","CPU_HZ","FREE_MEMORY_ESTIMATE","SESSION","TRANSPORT"):
            if row[0] in metadata:raise ValueError("Duplicate metadata: possible board reset")
            metadata[row[0]]=row[1]
        elif row[0]=="PAYLOAD":
            if row[1] in payloads:raise ValueError("Duplicate payload row")
            payloads[row[1]]=int(row[2])
        elif row[0]=="ACCURACY":
            if row[1] in accuracy:raise ValueError("Duplicate accuracy row: possible board reset")
            vals=list(map(float,row[2:6]))
            if len(vals)!=4 or not all(math.isfinite(v) and v>=0 for v in vals) or vals[3]>.002 or row[6]!="PASS":raise ValueError("Firmware parity failed")
            accuracy[row[1]]=dict(zip(("mae_C","rmse_C","max_C","max_host_gap_C"),vals))
        elif row[0]=="TIME":
            item=dict(repeat=int(row[2]),calls=int(row[3]),total_us=int(row[4]),checksum=float(row[5]))
            if item["calls"]!=2048 or item["total_us"]<=0 or not math.isfinite(item["checksum"]):raise ValueError("Invalid timing row")
            timings.setdefault(row[1],[]).append(item)
        elif row[0]=="UPDATE_COST_ONLY":
            if row[4]!="MONOTONE":raise ValueError("Online update lost monotonicity")
            item=dict(repeat=int(row[1]),calls=int(row[2]),total_us=int(row[3]))
            if item["calls"]!=2048 or item["total_us"]<=0:raise ValueError("Invalid update timing")
            updates.append(item)
    expected={"physics","polynomial","cubic16","kan16","uniform32_float32","uniform32_int16","uniform32_uint8","nonuniform16_int16"}
    if not(started and complete) or set(accuracy)!=expected or set(timings)!=expected|{"overhead"}:raise ValueError("Incomplete benchmark: use all-model firmware and capture from its header to END")
    if set(payloads)!=expected|{"overhead"} or any(v<0 for v in payloads.values()):raise ValueError("Missing/invalid payload metadata")
    board=metadata.get("BOARD");expected_hz={"mega2560":16000000,"esp32c3":160000000}
    if board not in expected_hz or int(metadata.get("CPU_HZ",0))!=expected_hz[board] or "FREE_MEMORY_ESTIMATE" not in metadata:raise ValueError("Missing or inconsistent board metadata")
    for name,rs in timings.items():
        if len(rs)!=10 or {r["repeat"] for r in rs}!=set(range(10)):raise ValueError(f"Incomplete/duplicate repeats: {name}")
    if len(updates)!=10 or {r["repeat"] for r in updates}!=set(range(10)):raise ValueError("Incomplete update-cost repeats")
    med={n:statistics.median(r["total_us"]/r["calls"] for r in rs) for n,rs in timings.items()}
    summary={n:dict(gross_us_per_call=v,overhead_subtracted_us_per_call=v-med["overhead"]) for n,v in med.items() if n!="overhead"}
    return dict(metadata=metadata,payload_bytes=payloads,accuracy=accuracy,timings=timings,summary=summary,
        overhead_us_per_call=med["overhead"],update_cost_only=updates,
        notes="Actual UART timing; numerical synthetic fixture, not physical-sensor validation. Negative corrected timings would indicate unresolved measurement overhead.")

def utc_now():
    return datetime.now(timezone.utc).isoformat()


def open_serial(port, command_mode, serial_factory=None):
    if serial_factory is None:
        import serial
        serial_factory=serial.Serial
    stream=serial_factory(port=None,baudrate=115200,timeout=.2,write_timeout=2,
                          xonxoff=False,rtscts=False,dsrdtr=False)
    if command_mode:
        # Set BEFORE opening; never pulse a reset while capturing native USB.
        stream.dtr=False;stream.rts=False
    stream.port=port
    try:
        stream.open()
    except Exception:
        stream.close()
        raise
    return stream


def capture_stream(stream,raw,command_mode,session_id,timeout,session,
                   emit=print,clock=time.monotonic):
    """Retain partial reads until newline; never clear unread device output."""
    lines=[];pending=b"";started=False;requested=False
    deadline=clock()+timeout;hello_deadline=min(deadline,clock()+15)
    last_hello=clock()-1;ack_deadline=deadline

    def send(data):
        if stream.write(data)!=len(data):raise IOError("Partial command write")
        session["tx_commands"].append(dict(utc=utc_now(),text=data.decode().strip()))

    while clock()<deadline:
        if command_mode and not requested:
            if clock()>=hello_deadline:
                raise TimeoutError("No SENSOR_USB_V2 handshake. Flash the updated C3 firmware; do not press RESET while capture is open.")
            if clock()-last_hello>=1:
                send(b"H\n");last_hello=clock()
        if requested and not started and clock()>=ack_deadline:
            raise TimeoutError("C3 accepted the handshake but no benchmark header arrived")
        chunk=stream.read(256)
        if not chunk:continue
        raw.write(chunk);raw.flush();pending+=chunk
        while b"\n" in pending:
            frame,pending=pending.split(b"\n",1)
            if len(frame)>4096:raise ValueError("Oversized serial frame")
            line=frame.decode("utf8",errors="replace").strip()
            if not line:continue
            emit(line);lines.append(line)
            if command_mode and line=="READY,SENSOR_USB_V2,esp32c3" and not requested:
                send(("R,"+session_id+"\n").encode());requested=True;ack_deadline=clock()+15
            if line.startswith("SENSOR_BENCH_V1"):
                if command_mode and not requested:
                    raise ValueError("Unexpected auto-starting firmware. Reflash the command-driven C3 build.")
                if started:raise ValueError("Board restarted during measurement")
                started=True
            if line.startswith("ALREADY_RAN,"):
                raise ValueError("Device rejected a repeated session ID")
            if line=="END_SENSOR_BENCH":
                return lines
        if len(pending)>4096:raise ValueError("Unterminated serial frame exceeds 4096 bytes")
    raise TimeoutError(f"Capture incomplete after {timeout:g} seconds; pending {len(pending)} bytes. Raw bytes have been retained.")


def main():
    p=argparse.ArgumentParser();p.add_argument("--port");p.add_argument("--out",type=Path,required=True)
    p.add_argument("--parse",type=Path);p.add_argument("--timeout",type=float,default=240)
    p.add_argument("--start-command",action="store_true",help="Updated ESP32-C3: handshake and request a run without RESET")
    args=p.parse_args();args.out.parent.mkdir(parents=True,exist_ok=True)
    if args.out.exists():p.error("Output already exists")
    if args.timeout<=0:p.error("Timeout must be positive")
    raw_path=args.out.with_suffix(".log");session_path=args.out.with_suffix(".session.json")
    if not args.parse and (raw_path.exists() or session_path.exists()):p.error("Raw log or session file already exists; choose a new output name")
    if args.parse:
        result=parse_lines(args.parse.read_text(encoding="utf-8-sig").splitlines())
        with args.out.open("x",encoding="utf8") as f:f.write(json.dumps(result,indent=2)+"\n")
        print("Validated saved device log:",args.out)
        return
    if not args.port:p.error("Supply --port or --parse")
    session_id=uuid.uuid4().hex
    session=dict(session_id=session_id,start_utc=utc_now(),port=args.port,
                 mode="command_v2" if args.start_command else "legacy_auto_start",
                 capture_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 status="capturing",tx_commands=[])
    with session_path.open("x",encoding="utf8") as f:f.write(json.dumps(session,indent=2)+"\n")
    error=None
    try:
        with raw_path.open("xb") as raw, open_serial(args.port,args.start_command) as stream:
            if args.start_command:
                print("Serial open. C3 command mode: establishing connection. Do NOT press RESET or BOOT.",flush=True)
            else:
                print("Serial open. Legacy Mega capture: if no header appears in 5 s, press RESET once. For updated C3 use --start-command.",flush=True)
            lines=capture_stream(stream,raw,args.start_command,session_id,args.timeout,session,
                                 emit=lambda s:print(s,flush=True))
        result=parse_lines(lines)
        if args.start_command:
            if result["metadata"].get("SESSION")!=session_id or result["metadata"].get("TRANSPORT")!="esp-idf-usb-jtag-command-v2":
                raise ValueError("Device session/transport differs from requested run")
            if result["metadata"]["BOARD"]!="esp32c3":raise ValueError("Unexpected board")
        with args.out.open("x",encoding="utf8") as f:f.write(json.dumps(result,indent=2)+"\n")
        session["status"]="validated"
        session["result_sha256"]=hashlib.sha256(args.out.read_bytes()).hexdigest()
    except Exception as exc:
        error=exc;session["status"]="failed";session["error_type"]=type(exc).__name__;session["error"]=str(exc)
    finally:
        session["end_utc"]=utc_now()
        if raw_path.exists():
            session["raw_sha256"]=hashlib.sha256(raw_path.read_bytes()).hexdigest()
            session["raw_bytes"]=raw_path.stat().st_size
        session_path.write_text(json.dumps(session,indent=2)+"\n",encoding="utf8")
    if error:
        print(f"Capture failed: {error}\nSaved raw log: {raw_path}\nSaved session: {session_path}")
        print("If USB disconnected, close monitors, unplug/replug C3, check its port, then retry with a NEW output name. Do not press RESET during capture.")
        raise SystemExit(1)
    print("Validated actual device capture:",args.out)
    print("Session identity and timestamps:",session_path)

if __name__=="__main__":main()
