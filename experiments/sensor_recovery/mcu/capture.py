"""Capture a real UART run; retain raw output and validate its completeness."""
import argparse,csv,json,math,statistics,time
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
        elif row[0] in ("BOARD","CPU_HZ","FREE_MEMORY_ESTIMATE"):
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
    if not(started and complete) or set(accuracy)!=expected or set(timings)!=expected|{"overhead"}:raise ValueError("Incomplete benchmark: use all-model firmware and capture from reset to END")
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

def main():
    p=argparse.ArgumentParser();p.add_argument("--port");p.add_argument("--out",type=Path,required=True)
    p.add_argument("--parse",type=Path);p.add_argument("--timeout",type=float,default=240)
    args=p.parse_args();args.out.parent.mkdir(parents=True,exist_ok=True)
    if args.out.exists():p.error("Output already exists")
    if not args.parse and args.out.with_suffix(".log").exists():p.error("Raw log already exists; choose a new output name")
    if args.parse:lines=args.parse.read_text(encoding="utf8").splitlines()
    else:
        if not args.port:p.error("Supply --port or --parse")
        import serial
        lines=[];end=time.monotonic()+args.timeout
        with serial.Serial(args.port,115200,timeout=1) as stream, args.out.with_suffix(".log").open("x",encoding="utf8") as raw:
            print("Serial open. Mega may start automatically. If no SENSOR_BENCH_V1 appears within 5 s, press RESET once. Do not reset after the benchmark starts; wait for END_SENSOR_BENCH.",flush=True)
            while time.monotonic()<end:
                line=stream.readline().decode("utf8",errors="replace").strip()
                if not line:continue
                print(line,flush=True);lines.append(line);raw.write(line+"\n");raw.flush()
                if line=="END_SENSOR_BENCH":break
    result=parse_lines(lines);args.out.write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")
    print("Validated actual device capture:",args.out)

if __name__=="__main__":main()
