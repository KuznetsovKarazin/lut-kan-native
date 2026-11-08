"""Upload once and acquire identified matched-storage/adaptation runs."""
import argparse,hashlib,json,shutil,subprocess,sys,zipfile
from pathlib import Path
try:
    from .run_suite import run_logged,wait_for_port,utc,sha
    from .capture_references import parse_lines
except ImportError:
    from run_suite import run_logged,wait_for_port,utc,sha
    from capture_references import parse_lines

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]


def source_hashes():
    paths=[HERE/"platformio.ini",HERE/"requirements.txt",HERE/"capture.py",HERE/"capture_references.py",
           Path(__file__),HERE/"run_suite.py",HERE/"revision_manifest.json",
           *sorted((HERE/"include").glob("*.h")),*sorted((HERE/"src").glob("*.cpp"))]
    return {str(p.relative_to(ROOT)):dict(raw_sha256=sha(p),lf_sha256=hashlib.sha256(p.read_bytes().replace(b"\r\n",b"\n")).hexdigest()) for p in paths}


def capture_runs(board,port,out,runs,metadata,runner=run_logged):
    for i in range(1,runs+1):
        metadata["stage"]=f"capture_{i}"
        target=out/f"run{i:02d}.json"
        runner([sys.executable,str(HERE/"capture_references.py"),"--board",board,"--port",port,"--out",str(target)],
               out/f"run{i:02d}.console.txt")
        result=json.loads(target.read_text(encoding="utf8"))
        session=json.loads(target.with_suffix(".session.json").read_text(encoding="utf8"))
        raw=target.with_suffix(".log")
        reparsed=parse_lines(raw.read_text(encoding="utf-8-sig").splitlines())
        if result!=reparsed:raise ValueError("Raw log does not reproduce result")
        if result["metadata"]["BOARD"]!=board or result["metadata"]["SESSION"]!=session["session_id"]:raise ValueError("Device/sidecar board or session mismatch")
        if session.get("status")!="validated" or session.get("raw_sha256")!=sha(raw) or session.get("result_sha256")!=sha(target):raise ValueError("Invalid capture sidecar/hash")
        if any(r["session_id"]==session["session_id"] for r in metadata["captures"]):raise ValueError("Repeated acquisition ID")
        metadata["captures"].append(dict(result=target.name,session_id=session["session_id"],raw_sha256=sha(raw),result_sha256=sha(target)))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--board",choices=["esp32c3","mega2560","esp32s3"],required=True)
    p.add_argument("--port",required=True);p.add_argument("--out",type=Path,required=True);p.add_argument("--runs",type=int,default=2)
    a=p.parse_args();out=a.out.resolve();archive=out.with_name(out.name+".zip")
    if not 1<=a.runs<=10:p.error("--runs must be between 1 and 10")
    if out.exists() or archive.exists():p.error("Choose a new output directory/archive")
    before=source_hashes();out.mkdir(parents=True)
    metadata=dict(start_utc=utc(),board=a.board,port=a.port,environment=a.board+"_references",
        requested_runs=a.runs,status="running",stage="upload",captures=[],source_before=before,python_version=sys.version)
    error=None
    try:
        git=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,capture_output=True,text=True)
        metadata["git_commit"]=git.stdout.strip() if git.returncode==0 else None
        command=[sys.executable,"-m","platformio","run","-d",str(HERE),"-e",metadata["environment"],"-t","upload","--upload-port",a.port]
        metadata["upload_command"]=command;run_logged(command,out/"upload.log")
        build=HERE/".pio/build"/metadata["environment"]
        (out/"firmware").mkdir();metadata["built_artifacts"]={}
        for name in ("firmware.elf","firmware.bin","firmware.hex","bootloader.bin","partitions.bin"):
            source=build/name
            if source.exists():
                target=out/"firmware"/name;shutil.copyfile(source,target)
                metadata["built_artifacts"][name]=dict(bytes=target.stat().st_size,sha256=sha(target),archive_path="firmware/"+name)
        needed={"firmware.elf","firmware.hex" if a.board=="mega2560" else "firmware.bin"}
        if not needed<=set(metadata["built_artifacts"]):raise ValueError("Missing required firmware images")
        metadata["stage"]="package_inventory"
        run_logged([sys.executable,"-m","platformio","pkg","list","-d",str(HERE),"-e",metadata["environment"]],out/"packages.txt")
        metadata["stage"]="port_wait";metadata["usb_device"]=wait_for_port(a.port)
        if source_hashes()!=before:raise ValueError("Sources changed during upload")
        capture_runs(a.board,a.port,out,a.runs,metadata)
        if source_hashes()!=before:raise ValueError("Sources changed during capture")
        metadata.update(status="validated",stage="complete")
    except (Exception,KeyboardInterrupt) as exc:
        error=exc;metadata.update(status="failed",error=str(exc),error_type=type(exc).__name__)
    finally:
        metadata["end_utc"]=utc();metadata["source_after"]=source_hashes()
        (out/"batch.json").write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf8")
        with zipfile.ZipFile(archive,"x",compression=zipfile.ZIP_DEFLATED) as z:
            for path in sorted(out.rglob("*")):
                if path.is_file():z.write(path,path.relative_to(out).as_posix())
    print("Evidence archive:",archive,flush=True)
    if error:raise SystemExit("Batch stopped; send the complete archive: "+str(error))
    print(f"Validated {len(metadata['captures'])} complete baseline captures. Board may be disconnected.",flush=True)


if __name__=="__main__":main()
