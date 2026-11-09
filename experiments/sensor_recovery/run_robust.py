"""Confirmation sweep with new units; known failures retained separately."""
import argparse,gzip,hashlib,json,platform,time
from pathlib import Path
import numpy as np
import scipy
from calibration import make_sensor,calibration_case,fit_physics,fit_selected,error_metrics
from robust import fit_shape,encode_shape
from run_pilot import aggregate,dump_json

HERE=Path(__file__).resolve().parent

def one_case(scenario,unit,repeat,budget,lengths,seed):
    c=calibration_case(make_sensor(scenario,unit,seed),budget,repeat,seed)
    tr=c["transform"]; x,y=c["x"],c["y"]; t,v=c["train_idx"],c["val_idx"]
    args=x[t],y[t],x[v],y[v],tr,scenario
    rows=[]
    def add(name,m,l=0,fmt="float32"):
        ideal=error_metrics(m.predict(c["x_ideal"]),c["y_test"],tr.y_half)
        noisy=error_metrics(m.predict(c["x_noisy"]),c["y_test"],tr.y_half)
        grid=np.linspace(-1,1,2049)
        rows.append(dict(scenario=scenario,unit=unit,repeat=repeat,budget=budget,
            method=name,length=l,encoding=fmt,payload_bytes=m.payload_bytes(),
            reference_sha256=c["reference_sha256"],hyper=m.hyper,
            positive_steps=int(np.sum(np.diff(m.predict(grid))>1e-7)),
            **{"ideal_"+k:val for k,val in ideal.items()},
            **{"noisy_"+k:val for k,val in noisy.items()}))
    add("physics",fit_physics(x,y,tr,scenario))
    add("polynomial",fit_selected("poly",*args))
    for l in lengths:
        for family,mono,irregular in (("direct_lut",False,False),("monotone_lut",True,False),
                                      ("nonuniform_monotone",True,True)):
            m,knots=fit_shape(*args,l,mono,irregular)
            for fmt in ("float32","int16"):
                add(family,encode_shape(m,knots,fmt,irregular),l,fmt)
        add("cubic_spline",fit_selected("cubic",*args,n_coeff=l),l)
        add("kan_edge",fit_selected("kan_edge",*args,n_coeff=l),l)
    return rows


def paired(rows):
    out=[]
    for scenario in sorted({r["scenario"] for r in rows}):
      for b in (30,75):
       for l in (8,16,32,64):
        for method in ("monotone_lut","nonuniform_monotone"):
         a={ (r["unit"],r["repeat"]):r for r in rows if r["scenario"]==scenario and r["budget"]==b and r["length"]==l and r["encoding"]=="int16" and r["method"]=="direct_lut"}
         z={ (r["unit"],r["repeat"]):r for r in rows if r["scenario"]==scenario and r["budget"]==b and r["length"]==l and r["encoding"]=="int16" and r["method"]==method}
         if not a: continue
         units=sorted({i[0] for i in a})
         d=np.array([np.mean([z[i]["noisy_mae"]-a[i]["noisy_mae"] for i in a if i[0]==u]) for u in units])
         draws=np.random.default_rng(431).choice(d,(2000,len(d)),replace=True).mean(axis=1)
         out.append(dict(scenario=scenario,budget=b,length=l,method=method,
             mean_mae_difference_vs_uniform=float(d.mean()),lo=float(np.quantile(draws,.025)),hi=float(np.quantile(draws,.975)),
             equal_payload=method=="monotone_lut"))
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True)
    p.add_argument("--units",type=int,default=8);p.add_argument("--repeats",type=int,default=3)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/"run_info.json").exists():p.error("Choose a fresh output directory")
    rows=[];start=time.time();count=0
    for scenario in ("ntc","ntc_frontend","ldr"):
      for budget in (30,75):
       for unit in range(args.units):
        for repeat in range(args.repeats):
         rows.extend(one_case(scenario,unit,repeat,budget,(8,16,32,64),20261108));count+=1
         if count%8==0:print(f"{count} confirmation cases, {time.time()-start:.1f}s",flush=True)
    with gzip.open(args.out/"raw_results.jsonl.gz","wt",encoding="utf8") as f:
        for row in rows:f.write(json.dumps(row,allow_nan=False)+"\n")
    summary=aggregate(rows)
    for r in summary:
        group=[v for v in rows if all(v[k]==r[k] for k in ("scenario","budget","method","length","encoding"))]
        r["cases_with_monotonicity_violations"]=sum(v["positive_steps"]>0 for v in group)
    dump_json(args.out/"summary.json",summary);dump_json(args.out/"paired.json",paired(rows))
    regression=[]
    for s,b,l,u,r in (("ntc",30,32,5,0),("ldr",30,64,2,2)):
        regression.extend(one_case(s,u,r,b,(l,),20261107))
    dump_json(args.out/"known_failure_regressions.json",regression)
    sources={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.iterdir() if p.suffix==".py" or p.name=="FOLLOWUP_PROTOCOL.md"}
    dump_json(args.out/"run_info.json",dict(seed=20261108,cases=count,evaluations=len(rows),
        units=args.units,repeats=args.repeats,python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
        elapsed_seconds=time.time()-start,source_sha256=sources,physical_measurements=False))
    print(f"Done: {count} cases / {len(rows)} evaluations",flush=True)

if __name__=="__main__":main()
