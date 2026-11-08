#pragma once
#include "revision_fixture.h"
namespace Revision {
static inline float polynomial(float adc) {
 const Matched::Transform &tr=matched_lut_float32.tr;
 // No clipping: retain the fitted Chebyshev model's defined extrapolation.
 float x=(adc-tr.x_mid)/tr.x_half,b1=0,b2=0;
 for(int j=REVISION_POLY_N-1;j>=1;--j){float b=2*x*b1-b2+revision_poly_c[j];b2=b1;b1=b;}
 return (x*b1-b2+revision_poly_c[0])*tr.y_half+tr.y_mid;
}
static inline float physics(float adc) {
 adc=fminf(fmaxf(adc,1e-7f),1.f-1e-7f);
 float z=logf(10000.f*adc/(1.f-adc));
 return 1.f/(revision_physics_c[0]+revision_physics_c[1]*z+revision_physics_c[2]*z*z*z)-273.15f;
}
// The existing full PAVA path remains unchanged for historical measurements.
static inline bool guarded_project(float *c,Matched::PavaWorkspace &w){
 if(Matched::decreasing(c))return false;
 Matched::project(c,w);return true;
}
}
