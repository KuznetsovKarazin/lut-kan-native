"""Shape-constrained LUTs and an explicitly accounted nonuniform baseline."""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import lsq_linear
from calibration import LAMBDAS, Model, EncodedLUT, linear_basis


def irregular_basis(x, knots):
    knots=np.asarray(knots,float)
    if np.any(np.diff(knots)<=0):
        raise ValueError("Knots must be strictly increasing")
    x=np.clip(np.asarray(x,float).reshape(-1),knots[0],knots[-1])
    lo=np.clip(np.searchsorted(knots,x,side="right")-1,0,len(knots)-2)
    w=(x-knots[lo])/(knots[lo+1]-knots[lo])
    a=np.zeros((len(x),len(knots)))
    a[np.arange(len(x)),lo]=1-w;a[np.arange(len(x)),lo+1]=w
    return a


def slope_penalty(knots):
    h=np.diff(knots)
    d=np.zeros((len(knots)-2,len(knots)))
    for j in range(len(d)):
        d[j,j:j+3]=[2*h[j+1]/(h[j]+h[j+1]),-2,2*h[j]/(h[j]+h[j+1])]
    return d


def solve_shape(a,y,penalty,lam,monotone):
    design=a/np.sqrt(len(y));target=np.asarray(y)/np.sqrt(len(y))
    if lam:
        design=np.vstack([design,np.sqrt(lam/len(penalty))*penalty])
        target=np.r_[target,np.zeros(len(penalty))]
    if not monotone:
        return np.linalg.lstsq(design,target,rcond=1e-12)[0]
    n=a.shape[1]
    # c[i] = z[0] - sum(z[1:i+1]), with z[1:] >= 0.
    mapping=-np.tril(np.ones((n,n)));mapping[:,0]=1
    fit=lsq_linear(design@mapping,target,
        bounds=(np.r_[-np.inf,np.zeros(n-1)],np.full(n,np.inf)),
        method="bvls",tol=1e-11,max_iter=500)
    if not fit.success:
        raise RuntimeError(f"Monotone solve did not converge: {fit.message}")
    c=mapping@fit.x
    if np.max(np.diff(c))>1e-9:
        raise RuntimeError("Monotonicity constraint violated")
    return c


@dataclass
class ShapeLUT:
    table: EncodedLUT
    knots: np.ndarray
    nonuniform: bool

    @property
    def hyper(self): return self.table.hyper
    def predict(self,x):
        a=irregular_basis(x,self.knots) if self.nonuniform else linear_basis(x,len(self.knots))
        return a@self.table.decoded()
    def payload_bytes(self):
        return self.table.payload_bytes()+(self.knots.nbytes if self.nonuniform else 0)


def fit_shape(x_train,y_train,x_val,y_val,transform,kind,length,monotone=True,nonuniform=False):
    if nonuniform:
        knots=np.quantile(x_train,np.linspace(0,1,length)).astype(np.float32)
        knots[0],knots[-1]=-1,1
        if np.any(np.diff(knots)<=0): raise ValueError("Degenerate quantile grid")
        basis=lambda x:irregular_basis(x,knots)
        penalty=slope_penalty(knots.astype(float))
    else:
        knots=np.linspace(-1,1,length,dtype=np.float32)
        basis=lambda x:linear_basis(x,length)
        penalty=np.diff(np.eye(length),n=2,axis=0)
    a,av=basis(x_train),basis(x_val)
    best=None
    for lam in LAMBDAS:
        c=solve_shape(a,y_train,penalty,lam,monotone).astype(np.float32)
        score=float(np.mean((av@c-y_val)**2))
        if best is None or score<best[0]: best=(score,lam)
    x,y=np.r_[x_train,x_val],np.r_[y_train,y_val]
    c=solve_shape(basis(x),y,penalty,best[1],monotone).astype(np.float32)
    model=Model("lut",c,transform,kind,dict(lambda_2=best[1],val_mse=best[0],
        monotone=monotone,nonuniform=nonuniform,grid_source="train_inputs_only"))
    return model,knots


def encode_shape(model,knots,encoding,nonuniform=False):
    return ShapeLUT(EncodedLUT.from_model(model,encoding),knots,nonuniform)
