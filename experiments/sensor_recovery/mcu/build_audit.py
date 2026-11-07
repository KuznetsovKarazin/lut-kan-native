"""Cross-compile whole benchmark and individual footprint-only deployments."""
import argparse,configparser,hashlib,json,os,re,subprocess,sys,tempfile
from pathlib import Path
HERE=Path(__file__).resolve().parent

def sizes(text):
    result={}
    for name in ("RAM","Flash"):
        m=re.search(name+r":\s+\[[^\]]*\].*?\(used (\d+) bytes",text)
        if not m:raise ValueError("Missing size report: "+name)
        result[name.lower()+"_bytes"]=int(m.group(1))
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True)
    p.add_argument("--boards",nargs="+",default=["mega2560","esp32c3"])
    p.add_argument("--models",nargs="+",type=int,default=list(range(9)))
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True);rows=[]
    env=os.environ.copy();env["PLATFORMIO_SETTING_ENABLE_TELEMETRY"]="No"
    for board in args.boards:
        for model in args.models+[-1]:
            config=configparser.ConfigParser(interpolation=None);config.read(HERE/"platformio.ini")
            flags=f"-DBENCH_MODEL={model}"+(" -DFOOTPRINT_ONLY" if model>=0 else "")
            config["env"]["build_flags"]=flags
            with tempfile.TemporaryDirectory() as tmp:
                ini=Path(tmp)/"platformio.ini"
                with ini.open("w") as f:config.write(f)
                command=[sys.executable,"-m","platformio","run","--project-dir",str(HERE),"--project-conf",str(ini),"-e",board]
                run=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=env)
            log=args.out/f"{board}_model{model}.log";log.write_text(run.stdout)
            if run.returncode:raise RuntimeError(f"Compilation failed; see {log}")
            elf=HERE/".pio/build"/board/"firmware.elf"
            elf_data=elf.read_bytes()
            if elf_data[:4]!=b"\x7fELF":raise RuntimeError(f"Invalid ELF header: {elf}")
            row=dict(board=board,model=model,mode="full_benchmark" if model==-1 else "footprint_only",**sizes(run.stdout),elf_sha256=hashlib.sha256(elf_data).hexdigest())
            rows.append(row);print(json.dumps(row),flush=True)
            (args.out/"builds.json").write_text(json.dumps(rows,indent=2)+"\n")
    versions=subprocess.run([sys.executable,"-m","platformio","pkg","list","--project-dir",str(HERE)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    (args.out/"packages.txt").write_text(versions.stdout)
    sources=[HERE/"platformio.ini",HERE/"requirements.txt",HERE/"build_audit.py",*sorted((HERE/"include").glob("*.h")),*sorted((HERE/"src").glob("*.cpp"))]
    (args.out/"source_hashes.json").write_text(json.dumps({str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},indent=2)+"\n")
    print("Static build audit complete. No physical-board timing was performed.")

if __name__=="__main__":main()
