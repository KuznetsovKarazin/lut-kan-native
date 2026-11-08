"""Add small fitted references to the unchanged matched synthetic fixture."""
import hashlib,json
from pathlib import Path
import numpy as np
from calibration import make_sensor,calibration_case,fit_physics,fit_selected
from export_matched_mcu import array,f32,SEED
HERE=Path(__file__).resolve().parent

def export():
 case=calibration_case(make_sensor('ntc_frontend',0,SEED),75,0,SEED);tr=case['transform'];i,j=case['train_idx'],case['val_idx']
 poly=fit_selected('poly',case['x'][i],case['y'][i],case['x'][j],case['y'][j],tr,'ntc_frontend')
 power=np.polynomial.chebyshev.cheb2poly(poly.coefficients.astype(float)).astype(np.float32)
 physics=fit_physics(case['x'],case['y'],tr,'ntc_frontend')
 out=HERE/'mcu';p=out/'include/revision_fixture.h'
 preds=np.array([m.predict(case['x_noisy'])*tr.y_half+tr.y_mid for m in (poly,physics)])
 h=['// Generated revision references; same seed/unit/B/test grid as matched fixture.','#pragma once','#include "matched_fixture.h"',f'#define REVISION_POLY_N {len(poly.coefficients)}',array('revision_poly_c',poly.coefficients).replace(' MATCHED_FLASH',''),array('revision_poly_horner_c',power).replace(' MATCHED_FLASH',''),array('revision_physics_c',physics.coefficients).replace(' MATCHED_FLASH',''),array('revision_expected',preds)]
 p.write_text('\n'.join(h)+'\n')
 modeldata=dict(polynomial_coefficients=poly.coefficients.tolist(),polynomial_horner_coefficients=power.tolist(),physics_coefficients=physics.coefficients.tolist(),transform=vars(tr),payload_bytes=[poly.payload_bytes(),physics.payload_bytes()],expected=preds.tolist(),adc=np.asarray(tr.dx(case['x_noisy']),np.float32).tolist())
 (out/'revision_reference.json').write_text(json.dumps(modeldata,indent=2)+'\n')
 digest=hashlib.sha256((out/'include/matched_fixture.h').read_bytes()+p.read_bytes()).hexdigest()
 (out/'include/revision_identity.h').write_text('#pragma once\n#define REVISION_FIXTURE_SHA256 "'+digest+'"\n')
 (out/'revision_manifest.json').write_text(json.dumps(dict(fixture_sha256=digest,source_seed=SEED,scenario='ntc_frontend',unit=0,repeat=0,budget=75,suite='matched_references_v2',parity_gate_C=.002,polynomial_horner_payload_bytes=16+4*len(power),payload_bytes=dict(lut32_float32=144,lut60_int16=144,cubic32_unrolled=144,horner8=144,polynomial=poly.payload_bytes(),physics=physics.payload_bytes(),polynomial_horner=16+4*len(power))),indent=2)+'\n')
 print('Revision fixture',digest,'payloads',modeldata['payload_bytes'])
if __name__=='__main__':export()
