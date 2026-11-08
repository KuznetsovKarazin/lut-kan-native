"""Paired second-review refinements: block OLS and extended decay grid."""
from pathlib import Path
import argparse,hashlib,json,platform,subprocess,time,gzip
import numpy as np
import scipy
import revision_analysis as r
HERE=Path(__file__).resolve().parent
TAUS=(32,128,512,1024,2048)

def affine_trajectory(case,initial,stream,family,accumulator='float32'):
    model=initial[family];c=model.coefficients;n=len(c);tr=stream['transform']
    a=r.design(family,stream['x'],n)
    tests={t:r.design(family,v['x'],n) for t,v in stream['tests'].items()}
    dtype=np.float32 if accumulator=='float32' else np.float64
    sums=np.zeros(4,dtype=dtype);gain,offset=np.float32(1),np.float32(0);count=0;degenerate=0
    def metric(t):return r.error_metrics((tests[t]@c)*gain+offset,stream['tests'][t]['truth'],tr.y_half)
    snapshots=[dict(t=0,before=metric(0),after=metric(0),gain=float(gain),offset=float(offset))]
    for t,(b,y) in enumerate(zip(a,stream['y']),1):
        before=metric(t) if t in tests else None
        p=dtype(b@c);y=dtype(y)
        sums+=np.array([p,y,p*p,p*y],dtype=dtype);count+=1
        if count==32:
            sp,sy,spp,spy=sums;den=dtype(count)*spp-sp*sp
            if den>dtype(1e-8):
                g=(dtype(count)*spy-sp*sy)/den
                off=(sy-g*sp)/dtype(count)
                gain,offset=np.float32(g),np.float32(off)
            else:degenerate+=1
            sums.fill(0);count=0
        if t in tests:snapshots.append(dict(t=t,before=before,after=metric(t),gain=float(gain),offset=float(offset)))
    tt=np.array([s['t'] for s in snapshots]);mae=np.array([s['after']['mae'] for s in snapshots]);mse=np.array([s['after']['rmse']**2/tr.y_half**2 for s in snapshots])
    return dict(method=family+'_affine32_'+accumulator,final_mae=float(mae[-1]),integrated_mae=float(np.trapezoid(mae,tt)/384),integrated_normalized_mse=float(np.trapezoid(mse,tt)/384),snapshots=snapshots,completed_steps=384,failure=None,degenerate_blocks=degenerate,accumulator_bytes=4*np.dtype(dtype).itemsize,correction_bytes=8,counter_not_included=True)

def run(stage,out,selection=None):
    out.mkdir(parents=True,exist_ok=False);start=time.time();dev=stage=='development'
    seed=r.DEV_SEED if dev else r.SEED;units=4 if dev else 8
    paths=['refinement_analysis.py','REFINEMENT_PROTOCOL.md','revision_analysis.py','matched_adaptation.py','calibration.py','robust.py']
    meta=dict(stage=stage,seed=seed,units=units,repeats=2,python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,source_sha256={p:hashlib.sha256((HERE/p).read_bytes()).hexdigest() for p in paths},base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),scope='post-review paired reanalysis of existing seeds')
    r.dump(out/'started.json',meta);scores={};groups={};curves={};count=0;paths_count=0
    chosen=None if dev else json.loads(selection.read_text())['selected']
    with gzip.open(out/'records.jsonl.gz','wt') as f:
        for sc in r.SCENARIOS:
            for B in (30,75):
                for unit in range(units):
                    for rep in range(2):
                        case=r.calibration_case(r.make_sensor(sc,unit,seed),B,rep,seed);initial=r.initial_models(case)
                        for drift in ('none','step','ramp'):
                            stream=r.generate_stream(case,rep,B,drift,seed);runs=[]
                            if dev:
                                for fam in ('lut','cubic'):
                                    for tau in TAUS:
                                        v=r.custom_trajectory(case,initial,stream,fam+'_decayed',tau);runs.append(v)
                                        scores.setdefault(fam,{}).setdefault(str(tau),[]).append(v['integrated_normalized_mse'])
                            else:
                                for fam in ('lut','cubic'):
                                    v=r.custom_trajectory(case,initial,stream,fam+'_decayed',chosen[fam]);v['method']=fam+'_decayed_extended';runs.append(v)
                                    for precision in ('float32','float64'):runs.append(affine_trajectory(case,initial,stream,fam,precision))
                                for v in runs:
                                    key=(sc,B,drift,v['method']);groups.setdefault(key,[]).append(dict(unit=unit,repeat=rep,final_mae=v['final_mae'],integrated_mae=v['integrated_mae']))
                                    for snap in v['snapshots']:curves.setdefault(key+(snap['t'],),[]).append(snap['after']['mae'])
                            r.write_row(f,dict(scenario=sc,budget=B,unit=unit,repeat=rep,drift=drift,runs=runs))
                            count+=1;paths_count+=len(runs)
                print(stage,sc,B,count,flush=True)
    if dev:
        means={fam:{k:float(np.mean(v)) for k,v in d.items()} for fam,d in scores.items()}
        r.dump(out/'selection.json',dict(selected={fam:int(min(d,key=d.get)) for fam,d in means.items()},objective='integrated_normalized_MSE',scores=means))
    else:
        result=[]
        for key,v in sorted(groups.items()):
            row=dict(zip(('scenario','budget','drift','method'),key))
            for metric in ('final_mae','integrated_mae'):
                per=[float(np.mean([x[metric] for x in v if x['unit']==u])) for u in range(units)]
                row[metric]=dict(mean=float(np.mean(per)),per_unit=per)
            result.append(row)
        r.dump(out/'summary.json',result)
        r.dump(out/'curves.json',[dict(zip(('scenario','budget','drift','method','t'),key),mean_mae=float(np.mean(v))) for key,v in sorted(curves.items())])
    meta.update(complete=True,streams=count,trajectories=paths_count,elapsed_seconds=time.time()-start,result_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()})
    r.dump(out/'run_info.json',meta);print(json.dumps(meta),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['development','confirmation']);p.add_argument('--out',type=Path,required=True);p.add_argument('--selection',type=Path);a=p.parse_args();run(a.stage,a.out,a.selection)
