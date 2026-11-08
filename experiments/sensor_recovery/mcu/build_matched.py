"""Audit full and individual matched builds; no device is connected or timed."""
import argparse,configparser,hashlib,json,os,re,subprocess,sys,tempfile
from pathlib import Path
from build_audit import sizes

HERE=Path(__file__).resolve().parent
FLASH_ARRAYS=("matched_adc","matched_truth","matched_expected","matched_stream_adc",
              "matched_stream_reference","matched_final_adc","matched_reference_states","matched_reference_metrics",
              "matched_initial_lut","matched_initial_cubic")


def avr_flash_check(elf):
    nm=Path.home()/".platformio/packages/toolchain-atmelavr/bin/avr-nm"
    r=subprocess.run([str(nm),"-S","--size-sort",str(elf)],capture_output=True,text=True,check=True)
    rows=[]
    for line in r.stdout.splitlines():
        v=line.split()
        if len(v)==4 and any(name in v[3] for name in FLASH_ARRAYS):
            start,size=int(v[0],16),int(v[1],16)
            if start+size>65536:raise ValueError("AVR fixture exceeds near-flash addressing: "+line)
            rows.append(dict(symbol=v[3],address=start,size_bytes=size,end=start+size))
    if len(rows)!=len(FLASH_ARRAYS):raise ValueError("Could not verify every AVR fixture array")
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--out",type=Path,required=True)
    p.add_argument("--boards",nargs="+",default=["mega2560_matched","esp32c3_matched"])
    p.add_argument("--models",nargs="+",type=int,default=[0,1,2,3,4,11,12,13,14])
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);rows=[]
    env=os.environ.copy();env["PLATFORMIO_SETTING_ENABLE_TELEMETRY"]="No"
    for board in a.boards:
        if board not in ("mega2560_matched","esp32c3_matched"):p.error("Use matched environments")
        for model in a.models+[-1]:
            config=configparser.ConfigParser(interpolation=None);config.read(HERE/"platformio.ini")
            if model>=0:config["env:"+board]["build_flags"]+=" -DMATCHED_FOOTPRINT -DMATCHED_METHOD_ID="+str(model)
            with tempfile.TemporaryDirectory() as tmp:
                ini=Path(tmp)/"platformio.ini"
                with ini.open("w") as f:config.write(f)
                command=[sys.executable,"-m","platformio","run","-d",str(HERE),"--project-conf",str(ini),"-e",board]
                run=subprocess.run(command,capture_output=True,text=True,env=env)
            log=run.stdout+run.stderr;(a.out/f"{board}_{model}.log").write_text(log)
            if run.returncode:raise RuntimeError("Build failed: "+board+"/"+str(model))
            if re.search(r"\bwarning:",log,re.I):raise RuntimeError("Compiler warning: "+board+"/"+str(model))
            build=HERE/".pio/build"/board;elf=build/"firmware.elf"
            row=dict(board=board,model_id=model,mode="full_benchmark" if model<0 else "individual_deployment",
                     **sizes(log),artifacts={f.name:dict(bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in sorted(build.glob("firmware.*")) if f.suffix in (".elf",".bin",".hex")})
            if model<0 and board.startswith("mega"):row["near_flash_arrays"]=avr_flash_check(elf)
            rows.append(row);print(json.dumps({k:row[k] for k in ("board","model_id","ram_bytes","flash_bytes")}),flush=True)
            (a.out/"builds.json").write_text(json.dumps(rows,indent=2)+"\n")
    packages=subprocess.run([sys.executable,"-m","platformio","pkg","list","-d",str(HERE)],capture_output=True,text=True,check=True)
    (a.out/"packages.txt").write_text(packages.stdout)
    paths=[HERE/"platformio.ini",Path(__file__),HERE/"requirements.txt",*sorted((HERE/"include").glob("*.h")),*sorted((HERE/"src").glob("*.cpp"))]
    (a.out/"source_hashes.json").write_text(json.dumps({str(f.relative_to(HERE)):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths},indent=2)+"\n")
    print("Complete. Static sizes/ELF checks only; no measured MCU timing or peak stack.")


if __name__=="__main__":main()
