#include "revision_kernels.h"
extern "C" float revision_predict(int id,float adc){return id==0?Revision::polynomial(adc):Revision::physics(adc);}
extern "C" int revision_project(float *c){Matched::PavaWorkspace w;return Revision::guarded_project(c,w);}
