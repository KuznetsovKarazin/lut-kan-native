"""Matched inference storage and causal online calibration; no MCU claims."""
from dataclasses import dataclass
import hashlib
import numpy as np
from scipy.interpolate import BSpline, PPoly
from scipy.optimize import isotonic_regression
from calibration import (Model, EncodedLUT, Transform, LAMBDAS, linear_basis,
    cubic_basis, spline_knots, regularized_fit, fit_selected, fit_physics,
    make_sensor, calibration_case, error_metrics)
from robust import solve_shape

CAPS = (80, 144, 224)
CHECKPOINTS = (0,32,64,96,127,128,129,132,144,160,192,224,256,320,384)
METHODS = ("frozen_lut", "frozen_cubic", "frozen_poly", "lut_nlms",
           "lut_projected_nlms", "cubic_nlms", "cubic_projected_nlms",
           "cubic_rls", "poly_rls")
ETA_GRID = (.05, .2, .5, 1.)
FORGET_GRID = (.98, .995, 1.)


@dataclass
class HornerModel:
    pieces: np.ndarray
    transform: Transform
    hyper: dict

    @classmethod
    def from_coefficients(cls, coefficients, transform, hyper=None):
        c = np.asarray(coefficients, np.float32)
        if len(c) < 4 or not np.all(np.isfinite(c)):
            raise ValueError("A finite cubic spline needs at least four coefficients")
        pp = PPoly.from_spline(BSpline(spline_knots(len(c)), c.astype(float), 3))
        ix = np.flatnonzero(np.diff(pp.x) > 0)
        widths = np.diff(pp.x)[ix]
        local = pp.c[:,ix] * widths[None,:] ** np.arange(3,-1,-1)[:,None]
        return cls(np.ascontiguousarray(local.T, dtype=np.float32), transform,
                   dict(hyper or {}, n_source_coefficients=len(c)))

    def predict(self, x):
        x = np.clip(np.asarray(x,float).reshape(-1), -1, 1)
        p = (x + 1) * (len(self.pieces)/2)
        ix = np.minimum(p.astype(int), len(self.pieces)-1)
        u = p - ix
        a,b,c,d = self.pieces[ix].astype(float).T
        return ((a*u+b)*u+c)*u+d

    def payload_bytes(self):
        return int(self.pieces.nbytes + 16)

    def boundary_gaps(self):
        if len(self.pieces) == 1:
            return dict(value=0., derivative=0.)
        c = self.pieces.astype(float)
        value = np.max(abs(c[:-1].sum(axis=1)-c[1:,3]))
        derivative = np.max(abs(3*c[:-1,0]+2*c[:-1,1]+c[:-1,2]-c[1:,2]))
        return dict(value=float(value), derivative=float(derivative*len(c)/2))


def fit_horner(x, y, xv, yv, tr, kind, pieces):
    n = pieces + 3
    a = cubic_basis(x,n)
    candidates=[]
    for lam in LAMBDAS:
        c=regularized_fit(a,y,lam).astype(np.float32)
        m=HornerModel.from_coefficients(c,tr)
        candidates.append((float(np.mean((m.predict(xv)-yv)**2)),lam))
    score,lam=min(candidates,key=lambda v:v[0])
    c=regularized_fit(cubic_basis(np.r_[x,xv],n),np.r_[y,yv],lam)
    return HornerModel.from_coefficients(c,tr,dict(lambda_2=lam,val_mse=score))


def fit_encoded_lut(x,y,xv,yv,tr,kind,length,encoding,monotone):
    a,av=linear_basis(x,length),linear_basis(xv,length)
    d=np.diff(np.eye(length),n=2,axis=0)
    best=None
    for lam in LAMBDAS:
        c=solve_shape(a,y,d,lam,monotone).astype(np.float32)
        enc=EncodedLUT.from_model(Model("lut",c,tr,kind),encoding)
        score=float(np.mean((av@enc.decoded()-yv)**2))
        if best is None or score<best[0]:best=(score,lam)
    c=solve_shape(linear_basis(np.r_[x,xv],length),np.r_[y,yv],d,best[1],monotone)
    m=Model("lut",c.astype(np.float32),tr,kind,
            dict(lambda_2=best[1],val_mse=best[0],monotone=monotone,
                 validation_encoding=encoding))
    return EncodedLUT.from_model(m,encoding)


def model_record(model):
    if isinstance(model,EncodedLUT):
        return model.to_dict()
    if isinstance(model,HornerModel):
        return dict(family="horner",pieces=model.pieces.tolist(),
                    transform=vars(model.transform),hyper=model.hyper,
                    payload_bytes=model.payload_bytes(),boundary_gaps=model.boundary_gaps())
    return dict(family=model.family,coefficients=model.coefficients.tolist(),
                transform=vars(model.transform),hyper=model.hyper,
                payload_bytes=model.payload_bytes())


def serialized_payload(model):
    meta=np.asarray(list(vars(model.transform).values()),dtype="<f4").tobytes()
    if isinstance(model,EncodedLUT):
        dtype={"float32":"<f4","int16":"<i2","uint8":"u1"}[model.encoding]
        raw=model.values.astype(dtype).tobytes()
        if model.encoding!="float32":
            raw+=np.asarray([model.scale,model.offset],dtype="<f4").tobytes()
    elif isinstance(model,HornerModel):raw=model.pieces.astype("<f4").tobytes()
    else:
        raw=model.coefficients.astype("<f4").tobytes()
        if model.family=="physics":raw+=np.asarray([10000],dtype="<f4").tobytes()
    return raw+meta


def static_models(case):
    it,iv=case["train_idx"],case["val_idx"]
    x,y,tr=case["x"],case["y"],case["transform"]
    kind=case["sensor"].kind
    args=x[it],y[it],x[iv],y[iv],tr,kind
    for cap in CAPS:
        n=(cap-16)//4
        for encoding,length in (("float32",n),("int16",(cap-24)//2)):
            for mono in (False,True):
                name=("monotone_lut_" if mono else "lut_")+encoding
                yield cap,name,fit_encoded_lut(*args,length,encoding,mono)
        yield cap,"cubic_compact",fit_selected("cubic",*args,n_coeff=n)
        yield cap,"cubic_horner",fit_horner(*args,(cap-16)//16)
        yield cap,"kan_compact",fit_selected("kan_edge",*args,n_coeff=n-1)
    yield 0,"polynomial",fit_selected("poly",*args)
    yield 0,"physics",fit_physics(x,y,tr,kind)


def reference_record(case):
    return dict(parameters=case["sensor"].parameters,transform=vars(case["transform"]),
        x=case["x"].tolist(),y=case["y"].tolist(),
        train_idx=case["train_idx"].tolist(),val_idx=case["val_idx"].tolist(),
        x_ideal=case["x_ideal"].tolist(),x_noisy=case["x_noisy"].tolist(),
        y_test=case["y_test"].tolist(),reference_sha256=case["reference_sha256"])


def drift_level(mode,t):
    t=np.asarray(t)
    if mode=="none":return np.zeros_like(t,dtype=float)
    if mode=="step":return (t>=128).astype(float)
    if mode=="ramp":return np.clip((t-128)/128,0,1)
    raise ValueError(mode)


def drifted_adc(sensor,q,level):
    a=sensor.adc(q)
    return np.clip(a+level*(2+2*np.sin(2*np.pi*a))/1023,0,1)


def adc_observation(adc,rng):
    codes=np.asarray(adc)*1023
    return np.clip(np.rint(codes[:,None]+rng.normal(0,1,(len(codes),16))),0,1023).mean(axis=1)/1023


def generate_stream(case,repeat,budget,mode,seed,steps=384):
    sensor,tr=case["sensor"],case["transform"]
    unit=sensor.unit
    def rng(tag):
        # The same update/test noise is paired across drift schedules.
        return np.random.default_rng(np.random.SeedSequence([seed,unit,repeat,budget,tag]))
    r=rng(711);q=[]
    for _ in range((steps+31)//32):
        block=(np.arange(32)+r.uniform(0,1,32))/32
        r.shuffle(block)
        q.extend(block.tolist())
    q=sensor.q_range[0]+np.asarray(q[:steps])*np.ptp(sensor.q_range)
    levels=drift_level(mode,np.arange(1,steps+1))
    adc=adc_observation(drifted_adc(sensor,q,levels),rng(712))
    sigma=.005 if sensor.kind=="ldr" else .02
    ref=q+rng(713).normal(0,sigma,steps)
    checkpoints=[t for t in CHECKPOINTS if t<=steps]
    if steps not in checkpoints:checkpoints.append(steps)
    qtest=np.linspace(*sensor.q_range,403)[1:-1]
    tests={}
    for t in checkpoints:
        rtest=np.random.default_rng(np.random.SeedSequence([seed,unit,repeat,budget,714,t]))
        a=drifted_adc(sensor,qtest,drift_level(mode,t))
        tests[t]=dict(x=tr.nx(adc_observation(a,rtest)),truth=tr.ny(qtest))
    return dict(x=tr.nx(adc),y=tr.ny(ref),q_truth=q,drift_levels=levels,
                checkpoints=checkpoints,tests=tests,transform=tr,mode=mode)


def initial_models(case):
    it,iv=case["train_idx"],case["val_idx"]
    x,y=case["x"],case["y"]
    args=x[it],y[it],x[iv],y[iv],case["transform"],case["sensor"].kind
    return {k:fit_selected(k,*args,n_coeff=32) for k in ("lut","cubic","poly")}


def family_of(method):
    if "cubic" in method:return "cubic"
    if "poly" in method:return "poly"
    return "lut"


def design(family,x,n):
    if family=="lut":return linear_basis(x,n)
    if family=="cubic":return cubic_basis(x,n)
    return np.polynomial.chebyshev.chebvander(x,n-1)


def project_decreasing(c):
    return isotonic_regression(np.asarray(c,float),increasing=False).x.astype(np.float32)


@dataclass
class OnlineState:
    coefficients: np.ndarray
    method: str
    parameter: float
    covariance: np.ndarray | None = None

    @classmethod
    def initialize(cls,model,method,parameter=0.):
        c=np.asarray(model.coefficients,np.float32).copy()
        if "projected" in method:c=project_decreasing(c)
        p=np.eye(len(c),dtype=np.float32)*100 if method.endswith("rls") else None
        return cls(c,method,float(np.float32(parameter)),p)

    def step(self,basis,label):
        b=np.asarray(basis,float);c=self.coefficients.astype(float)
        pred=float(b@c);err=float(label-pred)
        if self.method.startswith("frozen"):return pred
        if self.method.endswith("rls"):
            p=self.covariance.astype(float)
            pb=p@b;den=self.parameter+b@pb
            if not np.isfinite(den) or den<=0:raise FloatingPointError("Nonpositive RLS denominator")
            gain=pb/den
            c+=gain*err
            p=(p-np.outer(gain,pb))/self.parameter
            self.covariance=((p+p.T)/2).astype(np.float32)
            if not np.all(np.isfinite(self.covariance)):
                raise FloatingPointError("Non-finite RLS covariance")
        else:
            c+=self.parameter*err*b/(b@b+1e-6)
        self.coefficients=project_decreasing(c) if "projected" in self.method else c.astype(np.float32)
        if not np.all(np.isfinite(self.coefficients)):
            raise FloatingPointError("Non-finite coefficients")
        return pred

    def persistent_bytes(self):
        return int(16+self.coefficients.nbytes+(0 if self.method.startswith("frozen") else 4)
                   +(0 if self.covariance is None else self.covariance.nbytes))

    def snapshot(self):
        return dict(coefficients=self.coefficients.tolist(),
                    covariance=None if self.covariance is None else self.covariance.tolist())


def trajectory(initial,stream,method,parameter=0.,retain=True):
    family=family_of(method);model=initial[family]
    state=OnlineState.initialize(model,method,parameter)
    a=design(family,stream["x"],len(state.coefficients))
    tests={t:design(family,v["x"],len(state.coefficients)) for t,v in stream["tests"].items()}
    snapshots=[];prequential=[];failure=None
    tr=stream["transform"]

    def metrics(t):
        v=stream["tests"][t]
        return error_metrics(tests[t]@state.coefficients,v["truth"],tr.y_half)

    m=metrics(0)
    snapshots.append(dict(t=0,before=m,after=m,state=state.snapshot() if retain else None))
    try:
        for i,(b,label) in enumerate(zip(a,stream["y"]),1):
            before=metrics(i) if i in tests else None
            prediction=state.step(b,label)
            if retain:
                prequential.append(dict(prediction_normalized=prediction,
                    reference_error=float((prediction-label)*tr.y_half)))
            if i in tests:
                snap=dict(t=i,before=before,after=metrics(i))
                if retain:
                    snap.update(state=state.snapshot(),
                        coefficient_order_violations=(int(np.sum(np.diff(state.coefficients)>0))
                                                     if family!="poly" else None),
                        test_clipped_fraction=float(np.mean(abs(stream["tests"][i]["x"])>1)))
                if state.covariance is not None:
                    ev=np.linalg.eigvalsh(state.covariance.astype(float))
                    snap["covariance_min_eigenvalue"]=float(ev[0])
                    if ev[0]<-1e-5*max(1.,float(ev[-1])):
                        raise FloatingPointError("Indefinite RLS covariance")
                snapshots.append(snap)
    except (FloatingPointError,ValueError) as exc:
        failure=dict(step=i,error=str(exc))
    times=np.array([s["t"] for s in snapshots])
    maes=np.array([s["after"]["mae"] for s in snapshots])
    mses=np.array([s["after"]["rmse"]**2/tr.y_half**2 for s in snapshots])
    duration=max(1,int(times[-1]))
    result=dict(method=method,parameter=parameter,failure=failure,
        final_mae=float(maes[-1]),final_rmse=float(snapshots[-1]["after"]["rmse"]),
        integrated_mae=float(np.trapezoid(maes,times)/duration),
        integrated_normalized_mse=float(np.trapezoid(mses,times)/duration),
        peak_grid_max_abs=float(max(s[stage]["max_abs"] for s in snapshots for stage in ("before","after"))),
        inference_payload_bytes=16+state.coefficients.nbytes,
        persistent_state_bytes=state.persistent_bytes(),
        completed_steps=int(times[-1]),snapshots=snapshots,
        online_reference_clipped_fraction=float(np.mean(abs(stream["x"])>1)),
        consumed_update_labels=0 if method.startswith("frozen") else (i if failure is None else i-1),
        host_reference_arithmetic="float64 intermediate operations; float32 stored state",
        scratch_and_stack_bytes_measured=False)
    if retain:result["prequential"]=prequential
    return result


def stream_record(stream):
    return dict(x=stream["x"].tolist(),y=stream["y"].tolist(),
        q_truth_for_evaluation_only=stream["q_truth"].tolist(),
        drift_levels=stream["drift_levels"].tolist(),mode=stream["mode"],
        transform=vars(stream["transform"]),checkpoints=stream["checkpoints"],
        tests={str(t):{k:a.tolist() for k,a in v.items()} for t,v in stream["tests"].items()})


def paired_unit_difference(rows,a,b,group_keys,metric):
    groups={}
    for r in rows:
        if r["method"] in (a,b):
            key=tuple(r[k] for k in group_keys)
            groups.setdefault(key,{}).setdefault(r["method"],{})[(r["unit"],r["repeat"])]=r
    out=[]
    for key,methods in sorted(groups.items()):
        if set(methods)!={a,b}:continue
        if methods[a].keys()!=methods[b].keys():raise ValueError("Unpaired case inventory")
        ids=sorted(methods[a]);units=sorted({u for u,_ in ids})
        ua=[];ub=[]
        for unit in units:
            ui=[i for i in ids if i[0]==unit]
            ua.append(np.mean([methods[a][i][metric] for i in ui]))
            ub.append(np.mean([methods[b][i][metric] for i in ui]))
        ua,ub=np.asarray(ua),np.asarray(ub)
        ix=np.random.default_rng(2026110803).integers(0,len(units),(2000,len(units)))
        ci=np.quantile((ua[ix]-ub[ix]).mean(axis=1),[.025,.975])
        out.append(dict(zip(group_keys,key),method_a=a,method_b=b,metric=metric,
            n_units=len(units),mean_a=float(ua.mean()),mean_b=float(ub.mean()),
            mean_a_minus_b=float((ua-ub).mean()),ci95_lo=float(ci[0]),ci95_hi=float(ci[1])))
    return out
