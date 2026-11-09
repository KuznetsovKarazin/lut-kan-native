"""Strict capture of the matched fixture including polynomial and physical baselines."""
import argparse,csv,hashlib,json,math,re,statistics,uuid
from pathlib import Path
try:
 from .capture import capture_stream,open_serial,utc_now
except ImportError:
 from capture import capture_stream,open_serial,utc_now
HERE=Path(__file__).resolve().parent
METHODS=('lut32_float32','lut60_int16','cubic32_unrolled','horner8','polynomial','physics','polynomial_horner')
MEGA_HELLO_DELAY_S=2.;MEGA_HELLO_ERROR_BUDGET=3

def parse_lines(lines):
 meta={};payload={};accuracy={};timing={};prefix=[];started=False;complete=False
 def unique(d,k,v):
  if k in d:raise ValueError('Duplicate record '+str(k))
  d[k]=v
 for line in lines:
  row=next(csv.reader([line.strip()]))
  if not row:continue
  tag=row[0]
  if complete:raise ValueError('Trailing output')
  if tag=='SENSOR_BENCH_V1':
   if started or row[1:]!=['synthetic_ntc_frontend','seed2026110801','unit0','B75']:raise ValueError('Wrong/repeated fixture')
   started=True;continue
  if not started:prefix.append(row);continue
  if tag=='END_SENSOR_BENCH':
   if len(row)!=1:raise ValueError('Malformed END')
   complete=True
  elif tag in ('SUITE','SESSION','BOARD','CPU_HZ','FIXTURE_SHA256','PRECISION'):
   if len(row)!=2:raise ValueError('Malformed metadata')
   unique(meta,tag,row[1])
  elif tag=='PAYLOAD':
   if len(row)!=3:raise ValueError('Malformed payload')
   unique(payload,row[1],int(row[2]))
  elif tag=='ACCURACY':
   if len(row)!=7:raise ValueError('Malformed accuracy')
   v=list(map(float,row[2:6]))
   if not all(math.isfinite(x) and x>=0 for x in v) or v[3]>.002 or row[6]!='PASS':raise ValueError('Parity failure')
   unique(accuracy,row[1],dict(zip(('mae_C','rmse_C','max_C','max_host_gap_C'),v)))
  elif tag=='TIME':
   if len(row)!=6:raise ValueError('Malformed timing')
   rep,calls,us=map(int,row[2:5]);checksum=float(row[5])
   if rep not in range(10) or calls!=2048 or us<=0 or not math.isfinite(checksum):raise ValueError('Invalid timing')
   unique(timing,(row[1],rep),dict(method=row[1],repeat=rep,calls=calls,total_us=us,checksum=checksum))
  else:raise ValueError('Unknown record '+tag)
 manifest=json.loads((HERE/'revision_manifest.json').read_text());board=meta.get('BOARD')
 hz=dict(mega2560='16000000',esp32c3='160000000',esp32s3='240000000')
 if not started or not complete or set(meta)!={'SUITE','SESSION','BOARD','CPU_HZ','FIXTURE_SHA256','PRECISION'}:raise ValueError('Incomplete capture')
 if board not in hz or meta['CPU_HZ']!=hz[board] or meta['SUITE']!='matched_references_v2' or meta['PRECISION']!='float32' or meta['FIXTURE_SHA256']!=manifest['fixture_sha256']:raise ValueError('Identity mismatch')
 if not re.fullmatch('[0-9a-f]{32}',meta['SESSION']):raise ValueError('Invalid session')
 for tag,value in [('READY','SENSOR_REFERENCES_V2,'+board),('ECHO',meta['SESSION']),('ACCEPTED',meta['SESSION'])]:
  found=[r for r in prefix if r[0]==tag]
  if not found or any(','.join(r[1:])!=value for r in found) or (tag!='READY' and len(found)!=1):raise ValueError('Handshake mismatch '+tag)
 positions=[next(i for i,r in enumerate(prefix) if r[0]==tag) for tag in ('READY','ECHO','ACCEPTED')]
 if positions!=sorted(positions):raise ValueError('Handshake order')
 errors=[(i,r) for i,r in enumerate(prefix) if r[0]=='ERROR']
 if errors and (board!='mega2560' or len(errors)>3 or any(i>=positions[0] or r!=['ERROR','INVALID_COMMAND'] for i,r in errors)):raise ValueError('Command rejection')
 if payload!=manifest['payload_bytes'] or set(accuracy)!=set(METHODS) or set(timing)!={(m,i) for m in ('overhead',*METHODS) for i in range(10)}:raise ValueError('Incomplete/extra results')
 return dict(metadata=meta,payload_bytes=payload,accuracy=accuracy,timings=list(timing.values()),inference_median_us={m:statistics.median(v['total_us']/2048 for (name,_),v in timing.items() if name==m) for m in ('overhead',*METHODS)},notes='Synthetic inputs on physical MCU only if acquired from actual device; gross timing, no sensor/energy/WCET claim.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--board",choices=["mega2560","esp32c3","esp32s3"],required=True)
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
                 mode="command_references_v2",status="capturing",tx_commands=[],capture_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    side.write_text(json.dumps(session,indent=2)+"\n");error=None
    try:
        with raw.open("xb") as output,open_serial(a.port,True) as serial:
            print("Checking device handshake and session ID. Do not press RESET or BOOT.",flush=True)
            lines=capture_stream(serial,output,True,token,a.timeout,session,emit=lambda s:print(s,flush=True),
                                 ready_line="READY,SENSOR_REFERENCES_V2,"+a.board,
                                 hello_delay_s=MEGA_HELLO_DELAY_S if a.board=="mega2560" else 0.0,
                                 hello_error_budget=MEGA_HELLO_ERROR_BUDGET if a.board=="mega2560" else 0)
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
    print("Validated baseline capture:",a.out,flush=True)


if __name__=="__main__":main()
