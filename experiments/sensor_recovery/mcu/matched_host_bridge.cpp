#include "matched_fixture.h"
static Matched::OnlineModel state;
static Matched::PavaWorkspace workspace;
extern "C" float matched_predict(int id,float adc){
  if(id==0)return Matched::physical(Matched::lut32(Matched::normalize(adc,matched_lut_float32.tr),matched_lut_float32.c),matched_lut_float32.tr);
  if(id==1)return Matched::physical(Matched::lut60(Matched::normalize(adc,matched_lut_int16.tr),matched_lut_int16),matched_lut_int16.tr);
  if(id==2)return Matched::physical(Matched::cubic32(Matched::normalize(adc,matched_cubic_compact.tr),matched_cubic_compact.c),matched_cubic_compact.tr);
  return Matched::physical(Matched::horner8(Matched::normalize(adc,matched_cubic_horner.tr),matched_cubic_horner.c),matched_cubic_horner.tr);
}
extern "C" void matched_reset(int method){Matched::initialize(state,method>=2?matched_initial_cubic:matched_initial_lut,method%2,workspace);}
extern "C" float matched_step(int method,float adc,float label){return Matched::update(state,adc,label,method>=2,method%2,workspace);}
extern "C" void matched_state(float *out){for(int i=0;i<32;++i)out[i]=state.model.c[i];}
extern "C" float matched_online_predict(int method,float adc){return Matched::predict_online(state,adc,method>=2);}
extern "C" void matched_project(float *c){Matched::project(c,workspace);}
extern "C" void matched_basis(float x,float *out){float b[4];int k=Matched::cubic_basis(x,b);for(int i=0;i<32;++i)out[i]=0;for(int i=0;i<4;++i)out[k+i]=b[i];}
extern "C" float matched_cubic(float x,const float *c){return Matched::cubic32(x,c);}
