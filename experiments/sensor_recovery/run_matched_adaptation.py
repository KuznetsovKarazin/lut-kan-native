"""Execute the predeclared reconstruction in separately saved stages."""
import argparse
from contextlib import contextmanager
from datetime import datetime,timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import numpy as np
import scipy
from calibration import make_sensor,calibration_case,error_metrics,canonical_hash
from matched_adaptation import (CAPS,METHODS,ETA_GRID,FORGET_GRID,static_models,
    model_record,serialized_payload,reference_record,generate_stream,
    initial_models,trajectory,stream_record)

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SCENARIOS=("ntc","ntc_frontend","ldr")
BUDGETS=(30,75)
MODES=("none","step","ramp")


def dump(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")


@contextmanager
def gz_writer(path):
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="",fileobj=raw,mode="wb",mtime=0,compresslevel=6) as gz:
            with io.TextIOWrapper(gz,encoding="utf-8",newline="\n") as text:
                yield text


def emit(f,data):
    f.write(json.dumps(data,separators=(",",":"),allow_nan=False)+"\n")


def identification(kind,budget,unit,repeat,mode=None):
    d=dict(scenario=kind,budget=budget,unit=unit,repeat=repeat)
    if mode is not None:d["drift"]=mode
    d["case_id"]="_".join(map(str,d.values()))
    return d


def cases(units,repeats,seed):
    for kind in SCENARIOS:
        for budget in BUDGETS:
            for unit in range(units):
                sensor=make_sensor(kind,unit,seed)
                for repeat in range(repeats):
                    yield identification(kind,budget,unit,repeat),calibration_case(sensor,budget,repeat,seed)


def progress(label,count,total,started):
    if count%8==0 or count==total:
        print(f"{label}: {count}/{total}, {time.time()-started:.1f}s",flush=True)


def run_static(out,units,repeats,seed):
    count=rows=0;started=time.time();total=6*units*repeats
    with gz_writer(out/"cases.jsonl.gz") as refs,gz_writer(out/"raw_results.jsonl.gz") as raw:
        for ident,case in cases(units,repeats,seed):
            emit(refs,dict(ident,**reference_record(case)))
            for cap,name,model in static_models(case):
                payload=serialized_payload(model)
                if len(payload)!=model.payload_bytes() or (cap and len(payload)!=cap):
                    raise AssertionError("Payload accounting mismatch")
                ideal=error_metrics(model.predict(case["x_ideal"]),case["y_test"],case["transform"].y_half)
                noisy=error_metrics(model.predict(case["x_noisy"]),case["y_test"],case["transform"].y_half)
                emit(raw,dict(ident,cap_bytes=cap,method=name,model=model_record(model),
                    payload_bytes=len(payload),payload_sha256=hashlib.sha256(payload).hexdigest(),
                    used_references=ident["budget"],physical_unit=case["sensor"].quantity,
                    reference_sha256=case["reference_sha256"],
                    test_clipped_fraction=float(np.mean(abs(case["x_noisy"])>1)),
                    **{"noisy_"+k:v for k,v in noisy.items()},
                    **{"ideal_"+k:v for k,v in ideal.items()}))
                rows+=1
            count+=1;progress("static",count,total,started)
    return dict(cases=count,result_rows=rows)


def initial_record(models):
    return {k:model_record(v) for k,v in models.items()}


def candidate_grid(method):
    return FORGET_GRID if method.endswith("rls") else ETA_GRID


def run_development(out,units,repeats,seed):
    count=rows=0;started=time.time();total=18*units*repeats
    scores={m:{str(p):[] for p in candidate_grid(m)} for m in METHODS if not m.startswith("frozen")}
    failures={m:{str(p):0 for p in candidate_grid(m)} for m in scores}
    with gz_writer(out/"streams.jsonl.gz") as refs,gz_writer(out/"candidate_results.jsonl.gz") as raw:
        for ident,case in cases(units,repeats,seed):
            models=initial_models(case)
            for mode in MODES:
                sid=identification(ident["scenario"],ident["budget"],ident["unit"],ident["repeat"],mode)
                stream=generate_stream(case,ident["repeat"],ident["budget"],mode,seed)
                record=dict(sid,initial_references=reference_record(case),initial_models=initial_record(models),
                            stream=stream_record(stream))
                sh=canonical_hash(record);emit(refs,dict(record,stream_sha256=sh))
                for method in scores:
                    for parameter in candidate_grid(method):
                        result=trajectory(models,stream,method,parameter,retain=False)
                        key=str(parameter)
                        if result["failure"]:failures[method][key]+=1
                        else:scores[method][key].append(result["integrated_normalized_mse"])
                        emit(raw,dict(sid,stream_sha256=sh,**result));rows+=1
                count+=1;progress("development",count,total,started)
    selected={};details={}
    for method,grid in scores.items():
        details[method]=[]
        for p,values in grid.items():
            eligible=not failures[method][p] and len(values)==count
            details[method].append(dict(parameter=float(p),n_complete=len(values),
                failures=failures[method][p],eligible=eligible,
                objective=float(np.mean(values)) if values else None))
        eligible=[d for d in details[method] if d["eligible"]]
        if not eligible:raise RuntimeError("No valid development candidate: "+method)
        selected[method]=min(eligible,key=lambda d:d["objective"])["parameter"]
    selection=dict(development_seed=seed,development_units=units,development_repeats=repeats,
        development_streams=count,objective="mean integrated normalized heldout MSE over development streams",
        parameters=selected,candidates=details,confirmation_data_used=False)
    dump(out/"selected_parameters.json",selection)
    return dict(streams=count,trajectories=rows,selected_parameters=selected)


def run_confirmation(out,units,repeats,seed,selection_path):
    selection=json.loads(selection_path.read_text())
    if selection["development_seed"]==seed:raise ValueError("Development and confirmation seeds must differ")
    parameters=selection["parameters"]
    if set(parameters)!={m for m in METHODS if not m.startswith("frozen")}:
        raise ValueError("Incomplete or unexpected method selection")
    for method,p in parameters.items():
        if p not in candidate_grid(method):raise ValueError("Unregistered candidate")
    dump(out/"selected_parameters.json",selection)
    count=rows=failures=0;started=time.time();total=18*units*repeats
    with gz_writer(out/"streams.jsonl.gz") as refs,gz_writer(out/"raw_trajectories.jsonl.gz") as raw:
        for ident,case in cases(units,repeats,seed):
            models=initial_models(case)
            for mode in MODES:
                sid=identification(ident["scenario"],ident["budget"],ident["unit"],ident["repeat"],mode)
                stream=generate_stream(case,ident["repeat"],ident["budget"],mode,seed)
                record=dict(sid,initial_references=reference_record(case),initial_models=initial_record(models),
                            stream=stream_record(stream))
                sh=canonical_hash(record);emit(refs,dict(record,stream_sha256=sh))
                for method in METHODS:
                    result=trajectory(models,stream,method,parameters.get(method,0.),retain=True)
                    emit(raw,dict(sid,stream_sha256=sh,initial_reference_sha256=case["reference_sha256"],
                        available_reference_budget=ident["budget"]+384,
                        physical_unit=case["sensor"].quantity,**result))
                    failures+=bool(result["failure"]);rows+=1
                count+=1;progress("confirmation",count,total,started)
    return dict(streams=count,trajectories=rows,failed_trajectories=failures,
                selection_sha256=hashlib.sha256(selection_path.read_bytes()).hexdigest())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("static","development","confirmation"))
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--units",type=int)
    p.add_argument("--repeats",type=int)
    p.add_argument("--seed",type=int)
    p.add_argument("--selection",type=Path)
    a=p.parse_args()
    development=a.stage=="development"
    units=a.units if a.units is not None else (4 if development else 8)
    repeats=a.repeats if a.repeats is not None else (2 if development else 3)
    seed=a.seed if a.seed is not None else (2026110800 if development else 2026110801)
    if min(units,repeats)<1:p.error("Positive unit/repetition counts required")
    if a.stage=="confirmation" and a.selection is None:p.error("--selection required")
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.mkdir(exist_ok=False)
    head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    names=("calibration.py","robust.py","matched_adaptation.py","run_matched_adaptation.py",
           "MATCHED_ADAPTATION_PROTOCOL.md")
    manifest=dict(stage=a.stage,units=units,repeats=repeats,seed=seed,
        start_utc=datetime.now(timezone.utc).isoformat(),base_git_head=head,
        working_tree_dirty=bool(subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip()),
        command=[sys.executable,*sys.argv],python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
        threads={k:os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS")},
        source_sha256={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in names},
        provenance="new reconstruction run; synthetic parametric measurements",
        physical_sensor_measurements=False,new_mcu_measurements=False)
    dump(a.out/"started.json",manifest)
    started=time.time()
    if a.stage=="static":counts=run_static(a.out,units,repeats,seed)
    elif a.stage=="development":counts=run_development(a.out,units,repeats,seed)
    else:counts=run_confirmation(a.out,units,repeats,seed,a.selection)
    result_hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(a.out.iterdir()) if f.is_file()}
    manifest.update(counts,elapsed_seconds=time.time()-started,complete=True,
                    end_utc=datetime.now(timezone.utc).isoformat(),result_sha256=result_hashes)
    dump(a.out/"run_info.json",manifest)
    print(json.dumps(counts,allow_nan=False),flush=True)


if __name__=="__main__":main()
