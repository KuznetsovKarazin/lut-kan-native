"""Post-review audit, sensitivity and causal recalibration baselines.

The frozen protocol is REVIEW_REVISION_PROTOCOL.md. No physical acquisition or
MCU execution time is synthesized. Stages write new directories exclusively.
"""
from pathlib import Path
from dataclasses import dataclass
import argparse
import gzip
import hashlib
import json
import platform
import subprocess
import time
import numpy as np
import scipy
from calibration import (Sensor, Model, LAMBDAS, DEGREES, make_sensor,
    calibration_case, fit_selected, fit_physics, regularized_fit, error_metrics)
from matched_adaptation import (CAPS, CHECKPOINTS, METHODS, design, initial_models,
    generate_stream, trajectory, OnlineState, project_decreasing, model_record,
    reference_record, stream_record)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEED = 2026110811
DEV_SEED = 2026110810
SCENARIOS = ('ntc','ntc_frontend','ldr')
SETTINGS = {'nominal':{}, 'noise_low':{'noise':.25}, 'noise_high':{'noise':4.},
    'tolerance_half':{'spread':.5}, 'tolerance_double':{'spread':2.},
    'amplitude_low':{'amplitude':1.}, 'amplitude_high':{'amplitude':12.},
    'frequency_low':{'frequency':1.}, 'frequency_high':{'frequency':9.}}
TAUS = (32,128,512)
FORGET = (.90,.95,.98,.995,1.)


def dump(path, value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def write_row(f, row):
    f.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')


def rows(path):
    with gzip.open(path,'rt') as f:
        for line in f: yield json.loads(line)


@dataclass
class StressSensor(Sensor):
    noise: float = 1.
    amplitude: float = 4.
    frequency: float = 3.

    def adc(self,q):
        # Parent arithmetic remains unchanged for nominal settings.
        if self.kind != 'ntc_frontend': return super().adc(q)
        q=np.asarray(q,float);p=self.parameters
        r=p['r0']*np.exp(p['beta']*(1/(q+273.15)-1/298.15))
        a=r/(10000+r)
        return a+self.amplitude/1023*np.sin(2*np.pi*self.frequency*a+p['phase'])

    def observe(self,q,rng,averages=16):
        c=1023*self.adc(q)
        return np.clip(np.rint(c[:,None]+rng.normal(0,self.noise,(len(c),averages))),0,1023).mean(axis=1)/1023


def sensor_for(kind,unit,seed,setting):
    opts=SETTINGS[setting];base=make_sensor(kind,unit,seed);p=base.parameters.copy()
    scale=opts.get('spread',1.)
    p['r0']=10000+(p['r0']-10000)*scale
    key,center=('gamma',.7) if kind=='ldr' else ('beta',3950.)
    p[key]=center+(p[key]-center)*scale
    return StressSensor(kind,unit,p,opts.get('noise',1.),opts.get('amplitude',4.),opts.get('frequency',3.))


def oracle_inverse(sensor,adc):
    """Known-generator inverse; evaluator only, clipped to operating range."""
    a=np.asarray(adc,float);lo=np.full_like(a,sensor.q_range[0]);hi=np.full_like(a,sensor.q_range[1])
    for _ in range(55):
        mid=(lo+hi)/2;above=sensor.adc(mid)>a
        lo=np.where(above,mid,lo);hi=np.where(above,hi,mid)
    return (lo+hi)/2


def selected_cv(case,family,n=32):
    """Fivefold interleaved CV, fixed candidate grid, then refit on all B."""
    x,y=case['x'],case['y'];fold=np.arange(len(x))%5
    best=None
    for candidate in (DEGREES if family=='poly' else LAMBDAS):
        size=candidate+1 if family=='poly' else n
        a=design(family,x,size);scores=[]
        for k in range(5):
            train=fold!=k;valid=~train
            c=regularized_fit(a[train],y[train],0 if family=='poly' else candidate).astype(np.float32)
            scores.extend((a[valid]@c-y[valid])**2)
        score=float(np.mean(scores))
        if best is None or score<best[0]:best=(score,candidate)
    score,parameter=best;n=parameter+1 if family=='poly' else n
    c=regularized_fit(design(family,x,n),y,0 if family=='poly' else parameter).astype(np.float32)
    hyper=dict(selection='interleaved_5fold',cv_mse=score)
    hyper.update(dict(degree=int(parameter)) if family=='poly' else dict(n_coeff=n,lambda_2=parameter))
    return Model(family,c,case['transform'],case['sensor'].kind,hyper)


def old_fit(case,family,n=32):
    i,j=case['train_idx'],case['val_idx'];x,y=case['x'],case['y']
    return fit_selected(family,x[i],y[i],x[j],y[j],case['transform'],case['sensor'].kind,n)


def run_static(out,units,repeats):
    count=0
    with gzip.open(out/'records.jsonl.gz','wt') as f:
        for setting in SETTINGS:
            kinds=('ntc_frontend',) if setting.startswith(('amplitude','frequency')) else SCENARIOS
            for kind in kinds:
                for budget in (30,75):
                    for unit in range(units):
                        for rep in range(repeats):
                            sensor=sensor_for(kind,unit,SEED,setting)
                            case=calibration_case(sensor,budget,rep,SEED);tr=case['transform']
                            models=[]
                            for cap in CAPS if setting=='nominal' else (144,):
                                for family in ('lut','cubic'):
                                    n=(cap-16)//4
                                    models.append((cap,family+'_cv',selected_cv(case,family,n)))
                                    if setting=='nominal':models.append((cap,family+'_holdout',old_fit(case,family,n)))
                            models.append((0,'poly_cv',selected_cv(case,'poly')))
                            models.append((0,'physics',fit_physics(case['x'],case['y'],tr,kind)))
                            if setting=='nominal':models.append((0,'poly_holdout',old_fit(case,'poly')))
                            oracle=tr.ny(oracle_inverse(sensor,tr.dx(case['x_noisy'])))
                            output=[]
                            for cap,name,m in models:
                                pi,pn=m.predict(case['x_ideal']),m.predict(case['x_noisy'])
                                output.append(dict(cap_bytes=cap,method=name,model=model_record(m),
                                    ideal=error_metrics(pi,case['y_test'],tr.y_half),
                                    noisy=error_metrics(pn,case['y_test'],tr.y_half),
                                    prediction_ideal=pi.tolist(),prediction_noisy=pn.tolist()))
                            write_row(f,dict(setting=setting,scenario=kind,budget=budget,unit=unit,repeat=rep,
                                reference=reference_record(case),models=output,
                                oracle=error_metrics(oracle,case['y_test'],tr.y_half)))
                            count+=1
            print('static',setting,count,flush=True)
    return dict(cases=count)


def custom_trajectory(case,initial,stream,method,parameter=None):
    family='cubic' if method.startswith('cubic') else 'lut';model=initial[family]
    c=model.coefficients.copy();n=len(c);tr=stream['transform']
    a=design(family,stream['x'],n)
    test={t:design(family,v['x'],n) for t,v in stream['tests'].items()}
    affine='two_point' in method;batch='batch' in method;decayed='decayed' in method
    gain,offset=1.,0.;small=large=None
    state=OnlineState.initialize(model,family+'_nlms',.5)
    def prediction(t):
        y=test[t]@c
        return y*gain+offset if affine else y
    def measure(t):return error_metrics(prediction(t),stream['tests'][t]['truth'],tr.y_half)
    snapshots=[dict(t=0,before=measure(0),after=measure(0),coefficients=c.tolist())]
    for t,(b,label) in enumerate(zip(a,stream['y']),1):
        before=measure(t) if t in test else None
        if decayed:
            state.parameter=float(np.float32(.5/(1+t/parameter)));state.step(b,label);c=state.coefficients.copy()
        elif affine:
            p=float(b@model.coefficients)
            if small is None or p<small[0]:small=(p,float(label))
            if large is None or p>large[0]:large=(p,float(label))
            if t%32==0:
                if large[0]-small[0]>1e-8:
                    gain=float(np.float32((large[1]-small[1])/(large[0]-small[0])))
                    offset=float(np.float32(small[1]-gain*small[0]))
                small=large=None
        elif batch and t in test:
            xx=np.r_[case['x'],stream['x'][:t]];yy=np.r_[case['y'],stream['y'][:t]]
            if method.endswith('window'):xx,yy=xx[-64:],yy[-64:]
            c=regularized_fit(design(family,xx,n),yy,model.hyper['lambda_2']).astype(np.float32)
        else:
            if not batch:raise ValueError(method)
        if t in test:
            snapshots.append(dict(t=t,before=before,after=measure(t),coefficients=c.tolist(),gain=gain,offset=offset))
    times=np.array([s['t'] for s in snapshots]);maes=np.array([s['after']['mae'] for s in snapshots]);mses=np.array([s['after']['rmse']**2/tr.y_half**2 for s in snapshots])
    return dict(method=method,parameter=parameter,final_mae=float(maes[-1]),final_rmse=snapshots[-1]['after']['rmse'],
        integrated_mae=float(np.trapezoid(maes,times)/384),integrated_normalized_mse=float(np.trapezoid(mses,times)/384),
        peak_grid_max_abs=max(s[k]['max_abs'] for s in snapshots for k in ('before','after')),
        snapshots=snapshots,failure=None,completed_steps=384,
        retained_reference_pairs=(min(64,len(case['x'])+384) if method.endswith('window') else len(case['x'])+384) if batch else 0,
        inference_payload_bytes=model.payload_bytes())


def run_online(out,units,repeats,development,selection_path=None):
    seed=DEV_SEED if development else SEED
    selection=None if development else json.loads(selection_path.read_text())['selected']
    scores={};count=0;trajectories=0
    with gzip.open(out/'records.jsonl.gz','wt') as f:
        for kind in SCENARIOS:
            for budget in (30,75):
                for unit in range(units):
                    for rep in range(repeats):
                        case=calibration_case(make_sensor(kind,unit,seed),budget,rep,seed)
                        initial=initial_models(case)
                        for drift in ('none','step','ramp'):
                            stream=generate_stream(case,rep,budget,drift,seed);runs=[]
                            if development:
                                candidates=[(fam+'_decayed',tau) for fam in ('lut','cubic') for tau in TAUS]
                                candidates += [(fam+'_rls_extended',value) for fam in ('cubic','poly') for value in FORGET]
                            else:
                                candidates=[(m,.98 if m.endswith('rls') else .5) for m in METHODS]
                                candidates += list(selection.items())
                                candidates += [(fam+'_'+suffix,None) for fam in ('lut','cubic') for suffix in ('two_point','batch_all','batch_window')]
                            for method,param in candidates:
                                if any(suffix in method for suffix in ('decayed','two_point','batch')):
                                    run=custom_trajectory(case,initial,stream,method,param)
                                else:
                                    original=method.replace('_extended','')
                                    run=trajectory(initial,stream,original,param,retain=False);run['method']=method
                                runs.append(run);trajectories+=1
                                if development:scores.setdefault(method,{}).setdefault(str(param),[]).append(None if run['failure'] else run['integrated_normalized_mse'])
                            write_row(f,dict(scenario=kind,budget=budget,unit=unit,repeat=rep,drift=drift,
                                initial_models={k:model_record(v) for k,v in initial.items()},
                                stream=stream_record(stream),runs=runs))
                            count+=1
                print('development' if development else 'confirmation',kind,budget,count,flush=True)
    if development:
        means={m:{p:(None if any(x is None for x in v) else float(np.mean(v))) for p,v in grid.items()} for m,grid in scores.items()}
        failures={m:{p:sum(x is None for x in v) for p,v in grid.items()} for m,grid in scores.items()}
        selected={m:float(min((p for p in grid if grid[p] is not None),key=grid.get)) for m,grid in means.items()}
        dump(out/'selected_parameters.json',dict(selected=selected,development_scores=means,failed_trajectories=failures,seed=seed,criterion='checkpoint_integrated_normalized_MSE; any numerical failure makes a candidate inadmissible'))
    return dict(streams=count,method_trajectories=trajectories,seed=seed)


def audit_original(out):
    source=HERE/'results/matched_adaptation_20261108'
    static=json.loads((source/'static_summary.json').read_text());online=json.loads((source/'adaptation_summary.json').read_text())
    cells=[];projection=[]
    for sc in SCENARIOS:
        for budget in (30,75):
            subset=[r for r in static if r['scenario']==sc and r['budget']==budget]
            lut=next(r for r in subset if r['method']=='lut_float32' and r['cap_bytes']==80)
            cubic=next(r for r in subset if r['method']=='cubic_compact' and r['cap_bytes']==80)
            bestlut=min((r for r in subset if 'lut' in r['method']),key=lambda r:r['noisy_mae']['mean'])
            poly=next(r for r in subset if r['method']=='polynomial')
            cells.append(dict(scenario=sc,budget=budget,cubic80_lower_percent=100*(1-cubic['noisy_mae']['mean']/lut['noisy_mae']['mean']),
                best_lut_mae=bestlut['noisy_mae']['mean'],poly_mae=poly['noisy_mae']['mean'],
                best_equal_cap={str(cap):min((r for r in subset if r['cap_bytes']==cap),key=lambda r:r['noisy_mae']['mean'])['method'] for cap in CAPS}))
            for unit in range(8):
                for rep in range(3):
                    case=calibration_case(make_sensor(sc,unit,2026110801),budget,rep,2026110801);init=initial_models(case)
                    for drift in ('none','step','ramp'):
                        stream=generate_stream(case,rep,budget,drift,2026110801)
                        for family in ('lut','cubic'):
                            c=init[family].coefficients.copy();pc=project_decreasing(c);init_changed=bool(np.any(pc!=c));c=pc
                            activity=0;max_correction=0.;a=design(family,stream['x'],len(c))
                            for b,label in zip(a,stream['y']):
                                d=c.astype(float)+.5*(label-b@c)*b/(b@b+1e-6)
                                p=project_decreasing(d);unprojected=d.astype(np.float32)
                                activity+=bool(np.any(p!=unprojected));max_correction=max(max_correction,float(np.max(abs(p.astype(float)-unprojected)))*case['transform'].y_half);c=p
                            projection.append(dict(scenario=sc,budget=budget,unit=unit,repeat=rep,drift=drift,family=family,
                                changed_initial_state=init_changed,changed_updates=activity,max_correction_physical=max_correction))
            print('audit',sc,budget,flush=True)
    dump(out/'audit.json',dict(static=cells,projection=projection,
        no_drift75=[r for r in online if r['budget']==75 and r['drift']=='none'],
        note='Each original projected state was replayed; equality of rounded means alone was not used as proof.'))
    return dict(original_streams=432,projected_trajectories=864,updates_checked=864*384)


def summarize(root):
    groups={};curves={};counts={}
    for stage in ('static','confirmation'):
        for row in rows(root/stage/'records.jsonl.gz'):
            runs=row['models'] if stage=='static' else row['runs']
            for r in runs:
                if r.get('failure'):raise ValueError('Confirmation includes numerical failures; report separately instead of aggregating truncated paths')
                key=(stage,row.get('setting','nominal'),row['scenario'],row['budget'],row.get('drift','none'),r.get('cap_bytes',0),r['method'])
                metrics={'noisy_mae':r['noisy']['mae'],'ideal_mae':r['ideal']['mae'],'max_error':r['noisy']['max_abs'],'oracle_mae':row['oracle']['mae']} if stage=='static' else {k:r[k] for k in ('final_mae','integrated_mae','peak_grid_max_abs')}
                groups.setdefault(key,[]).append(dict(unit=row['unit'],repeat=row['repeat'],**metrics))
                if stage=='confirmation':
                    for snap in r['snapshots']:
                        ck=key+(snap['t'],);curves.setdefault(ck,[]).append(snap['after']['mae'])
                counts[stage]=counts.get(stage,0)+1
    result=[]
    names=('stage','setting','scenario','budget','drift','cap_bytes','method')
    for key,values in sorted(groups.items()):
        data=dict(zip(names,key));units=sorted({v['unit'] for v in values})
        for metric in values[0]:
            if metric in ('unit','repeat'):continue
            per=[float(np.mean([v[metric] for v in values if v['unit']==u])) for u in units]
            data[metric]=dict(mean=float(np.mean(per)),per_unit=per,maximum_observed=float(max(v[metric] for v in values)))
        data.update(n_units=len(units),n_cases=len(values));result.append(data)
    dump(root/'summary.json',result)
    dump(root/'curves.json',[dict(zip(names+('t',),k),mean_mae=float(np.mean(v))) for k,v in sorted(curves.items())])
    return counts


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('audit','static','development','confirmation','summarize'));parser.add_argument('--out',type=Path,required=True);parser.add_argument('--selection',type=Path);parser.add_argument('--units',type=int);parser.add_argument('--repeats',type=int);a=parser.parse_args()
    if a.stage=='summarize':print(summarize(a.out));return
    a.out.mkdir(parents=True,exist_ok=False);start=time.time()
    units=a.units or (4 if a.stage=='development' else 8);repeats=a.repeats or 2
    info=dict(stage=a.stage,base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,units=units,repeats=repeats,
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),HERE/'REVIEW_REVISION_PROTOCOL.md',HERE/'calibration.py',HERE/'matched_adaptation.py']})
    dump(a.out/'started.json',info)
    if a.stage=='audit':result=audit_original(a.out)
    elif a.stage=='static':result=run_static(a.out,units,repeats)
    else:result=run_online(a.out,units,repeats,a.stage=='development',a.selection)
    info.update(result,elapsed_seconds=time.time()-start,complete=True);info['result_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in a.out.iterdir() if p.is_file()}
    dump(a.out/'run_info.json',info);print(json.dumps(info),flush=True)


if __name__=='__main__':main()
