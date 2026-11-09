"""Reproduce numerical-precision checks; no synthetic hardware timing."""
import argparse,ctypes,gzip,json,subprocess,tempfile
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent

def audit(root):
 rows=json.loads((root/'confirmation/summary.json').read_text())
 lookup={(r['scenario'],r['budget'],r['drift'],r['method']):r for r in rows}
 gaps={}
 for r in rows:
  if r['method'].endswith('_float32'):
   other=lookup[(r['scenario'],r['budget'],r['drift'],r['method'].replace('float32','float64'))]
   gaps[r['scenario']]=max(gaps.get(r['scenario'],0),abs(r['final_mae']['mean']-other['final_mae']['mean']))
 degenerate=0;count=0
 with gzip.open(root/'confirmation/records.jsonl.gz','rt') as f:
  for line in f:
   for r in json.loads(line)['runs']:
    degenerate+=r.get('degenerate_blocks',0);count+=1
 result={'degenerate_blocks':degenerate,'max_group_final_mae_gap_by_scenario':gaps,'units':{'ntc':'degC','ntc_frontend':'degC','ldr':'log10 lux'},'trajectories':count}
 (root/'ols_audit.json').write_text(json.dumps(result,indent=2)+'\n')
 data=json.loads((HERE/'mcu/revision_reference.json').read_text());tr=data['transform']
 with tempfile.TemporaryDirectory() as tmp:
  libfile=Path(tmp)/'reference.so'
  subprocess.run(['g++','-std=c++11','-O2','-fno-fast-math','-ffp-contract=off','-shared','-fPIC','-I',str(HERE/'mcu/include'),str(HERE/'mcu/revision_host_bridge.cpp'),'-o',str(libfile)],check=True)
  lib=ctypes.CDLL(str(libfile));lib.revision_predict.argtypes=[ctypes.c_int,ctypes.c_float];lib.revision_predict.restype=ctypes.c_float
  x=np.linspace(-1,1,10001);adc=(x*tr['x_half']+tr['x_mid']).astype(np.float32)
  xx=(adc.astype(float)-tr['x_mid'])/tr['x_half'];expected=np.polynomial.chebyshev.chebval(xx,data['polynomial_coefficients'])*tr['y_half']+tr['y_mid']
  pred={m:np.array([lib.revision_predict(i,a) for a in adc]) for m,i in [('clenshaw',0),('horner',2)]}
  gaps={m:float(np.max(abs(v-expected))) for m,v in pred.items()}
  fixture_gaps={m:float(np.max(abs(np.array([lib.revision_predict(i,a) for a in data['adc']])-data['expected'][1 if i==1 else 0]))) for m,i in [('clenshaw',0),('physics',1),('horner',2)]}
  assert max([*gaps.values(),*fixture_gaps.values()])<.002
  result={'kind':'host_float32_parity_not_mcu_timing','fixture_sha256':json.loads((HERE/'mcu/revision_manifest.json').read_text())['fixture_sha256'],'dense_grid_count':len(x),'normalized_range':[-1,1],'max_gap_C':gaps,'max_between_kernels_C':float(np.max(abs(pred['clenshaw']-pred['horner']))),'test_observation_gaps_C':fixture_gaps}
  (root/'host_parity.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--results',type=Path,default=HERE/'results/refinement_20261108');a=p.parse_args();audit(a.results)
