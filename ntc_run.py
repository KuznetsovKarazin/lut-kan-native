import sys,numpy as np,json,time
sys.path.insert(0,'src')
from lut_native.sensors import make_ntc_units,ntc_calibration_data,ntc_factory_baseline,fit_poly_baseline
from lut_native.core import lut_forward_numpy
from lut_native.training import train_lut_edge,TrainConfig
from scipy.stats import wilcoxon

K,L=1,32; units=make_ntc_units(8,seed=42); lu,po,fa=[],[],[]
for i,u in enumerate(units):
    d=ntc_calibration_data(u,50,25,200,seed=42+i)
    fac=ntc_factory_baseline(u,d)
    pb=None
    for dg in [2,3,4,5]:
        p=fit_poly_baseline(d['x_cal'],d['y_cal'],dg,d['x_test'],d['y_test'],d['denorm_y'])
        if pb is None or p['mae_phys']<pb['mae_phys']: pb=p
    ms=[]
    for s in range(5):
        r=train_lut_edge(np.zeros((K,L),np.float32),d['x_cal'],d['y_cal'],d['x_val'],d['y_val'],d['x_test'],d['y_test'],-1.0,1.0,TrainConfig(lambda_2=1.0,lr=0.01,epochs=400,batch_size=32,eval_every_epochs=25,seed=s))
        yp=lut_forward_numpy(d['x_test'],r.lut_best)
        ms.append(float(np.mean(np.abs(d['denorm_y'](yp)-d['denorm_y'](d['y_test'])))))
    lu.append(float(np.mean(ms))); po.append(pb['mae_phys']); fa.append(fac['mae_phys'])
    print(f'  NTC S{i+1} B={u.B:.0f}: poly={pb["mae_phys"]:.4f} lut={np.mean(ms):.4f} fac={fac["mae_phys"]:.4f}')
try: _,p=wilcoxon(lu,po,alternative='less')
except: p=float('nan')
rng=np.random.RandomState(0)
ratios=[po_/lu_ for po_,lu_ in zip(po,lu)]
boots=[np.mean(rng.choice(ratios,8,replace=True)) for _ in range(2000)]
ci=(float(np.percentile(boots,2.5)),float(np.percentile(boots,97.5)))
print(f'NTC: poly={np.mean(po):.4f} lut={np.mean(lu):.4f} ratio={np.mean(po)/np.mean(lu):.2f}x CI=[{ci[0]:.2f},{ci[1]:.2f}] p={p:.4f}')
print(f'vs factory: {np.mean(fa):.4f} -> {np.mean(lu):.4f} ({np.mean(fa)/np.mean(lu):.2f}x)')
res={'ntc':dict(poly=round(float(np.mean(po)),4),lut=round(float(np.mean(lu)),4),ratio=round(float(np.mean(po)/np.mean(lu)),3),ci=[round(ci[0],3),round(ci[1],3)],p=round(float(p),4),factory=round(float(np.mean(fa)),4),factory_ratio=round(float(np.mean(fa)/np.mean(lu)),3))}
import json; json.dump(res,open('results/sensor_calib/ntc_ci.json','w'),indent=2)
