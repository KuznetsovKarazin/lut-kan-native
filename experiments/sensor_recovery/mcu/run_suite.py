"""Upload the ten-model suite once, capture sequential runs, and package evidence."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile
from datetime import datetime,timezone
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]


def utc():return datetime.now(timezone.utc).isoformat()
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes():
    paths=[HERE/"platformio.ini",HERE/"capture.py",Path(__file__),HERE/"fixture_manifest.json",HERE/"optimized_manifest.json",
           *sorted((HERE/"include").glob("*.h")),*sorted((HERE/"src").glob("*.cpp"))]
    return {str(p.relative_to(ROOT)):dict(raw_sha256=sha(p),lf_sha256=hashlib.sha256(p.read_bytes().replace(b"\r\n",b"\n")).hexdigest()) for p in paths}


def console_print(text,end="\n"):
    """Keep a legacy console from aborting an otherwise valid capture."""
    try:
        print(text,end=end,flush=True)
    except UnicodeEncodeError:
        encoding=getattr(sys.stdout,"encoding",None) or "ascii"
        print(text.encode(encoding,errors="backslashreplace").decode(encoding),end=end,flush=True)


def run_logged(command,path):
    console_print("Running: "+subprocess.list2cmdline(command))
    # Redirected Python stdout otherwise uses the Windows locale (e.g. CP1251).
    # PlatformIO prints Unicode dependency-tree characters; configure the child,
    # not just this parent's decoder. Its Python subprocesses inherit the setting.
    env=os.environ.copy()
    env.update(PYTHONIOENCODING="utf-8",PYTHONUTF8="1")
    with path.open("x",encoding="utf8") as log:
        with subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                              text=True,encoding="utf8",errors="replace",env=env) as process:
            for line in process.stdout:
                log.write(line);log.flush();console_print(line,end="")
            code=process.wait()
    if code:raise RuntimeError(f"Command exited {code}; see {path.name}")


def wait_for_port(port,timeout=20):
    from serial.tools import list_ports
    deadline=time.monotonic()+timeout;stable=None
    print("Waiting for the serial port to remain present; do not press RESET or BOOT.",flush=True)
    while time.monotonic()<deadline:
        ports=list(list_ports.comports())
        match=next((p for p in ports if p.device.casefold()==port.casefold()),None)
        if match:
            if stable is None:stable=time.monotonic()
            if time.monotonic()-stable>=1:
                serial_id=getattr(match,"serial_number",None)
                return dict(port=match.device,vid=match.vid,pid=match.pid,
                            usb_serial_sha256=hashlib.sha256(serial_id.encode()).hexdigest() if serial_id else None)
        else:stable=None
        time.sleep(.2)
    raise TimeoutError(f"{port} did not become available. Check 'py -m platformio device list'; a changed COM number needs a new invocation.")


def capture_runs(board,port,out,runs,metadata,runner=run_logged):
    for i in range(1,runs+1):
        metadata["stage"]=f"capture_{i}"
        target=out/f"run{i:02d}.json"
        command=[sys.executable,str(HERE/"capture.py"),"--port",port,"--out",str(target)]
        if board=="esp32c3":command.append("--start-command")
        runner(command,out/f"run{i:02d}.console.txt")
        data=json.loads(target.read_text(encoding="utf8"))
        session=json.loads(target.with_suffix(".session.json").read_text(encoding="utf8"))
        if data["metadata"].get("SUITE")!="spline_optimized_v1" or data["metadata"].get("BOARD")!=board or len(data["accuracy"])!=10:
            raise ValueError("Capture is not the requested ten-model board/suite")
        raw=target.with_suffix(".log")
        if session.get("status")!="validated" or session.get("raw_sha256")!=sha(raw) or session.get("result_sha256")!=sha(target):
            raise ValueError("Capture sidecar/hashes are inconsistent")
        if any(r["session_id"]==session["session_id"] for r in metadata["captures"]):
            raise ValueError("Repeated acquisition ID")
        metadata["captures"].append(dict(result=target.name,session_id=session["session_id"],raw_sha256=sha(raw),result_sha256=sha(target)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--board",choices=["esp32c3","mega2560"],required=True)
    p.add_argument("--port",required=True);p.add_argument("--out",type=Path,required=True)
    p.add_argument("--runs",type=int,default=2)
    args=p.parse_args();out=args.out.resolve();archive=out.with_name(out.name+".zip")
    if not 1<=args.runs<=10:p.error("--runs must be between 1 and 10")
    if out.exists() or archive.exists():p.error("Output directory/archive exists; choose a new name")
    before=source_hashes();out.mkdir(parents=True)
    metadata=dict(start_utc=utc(),board=args.board,port=args.port,environment=args.board+"_opt",
                  requested_runs=args.runs,status="running",stage="upload",captures=[],source_before=before,
                  python_version=sys.version)
    try:
        git=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,capture_output=True,text=True)
        metadata["git_commit"]=git.stdout.strip() if git.returncode==0 else None
    except OSError:metadata["git_commit"]=None
    error=None
    try:
        command=[sys.executable,"-m","platformio","run","-d",str(HERE),"-e",metadata["environment"],"-t","upload","--upload-port",args.port]
        metadata["upload_command"]=command;run_logged(command,out/"upload.log")
        build=HERE/".pio/build"/metadata["environment"]
        metadata["built_artifacts"]={name:dict(bytes=(build/name).stat().st_size,sha256=sha(build/name)) for name in ("firmware.bin","firmware.elf") if (build/name).exists()}
        # AVR produces .hex rather than .bin.
        if (build/"firmware.hex").exists():metadata["built_artifacts"]["firmware.hex"]=dict(bytes=(build/"firmware.hex").stat().st_size,sha256=sha(build/"firmware.hex"))
        if not metadata["built_artifacts"]:raise ValueError("No built firmware artifact found")
        metadata["stage"]="port_wait";metadata["usb_device"]=wait_for_port(args.port)
        if source_hashes()!=before:raise ValueError("Sources changed during upload; capture stopped")
        capture_runs(args.board,args.port,out,args.runs,metadata)
        if source_hashes()!=before:raise ValueError("Sources changed during capture")
        metadata["status"]="validated";metadata["stage"]="complete"
    except (Exception,KeyboardInterrupt) as exc:
        error=exc;metadata["status"]="failed";metadata["error"]=str(exc);metadata["error_type"]=type(exc).__name__
    finally:
        metadata["end_utc"]=utc();metadata["source_after"]=source_hashes()
        (out/"batch.json").write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf8")
        with zipfile.ZipFile(archive,"x",compression=zipfile.ZIP_DEFLATED) as z:
            for path in sorted(out.iterdir()):
                if path.is_file():z.write(path,path.name)
    print("Evidence archive:",archive,flush=True)
    if error:
        print("Batch stopped:",error,flush=True);raise SystemExit(1)
    print(f"Validated {len(metadata['captures'])} complete ten-model captures. Board may be disconnected.",flush=True)


if __name__=="__main__":main()
