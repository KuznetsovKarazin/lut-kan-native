#pragma once
#include <math.h>
#include <stdint.h>

namespace Matched {
struct Transform { float x_mid,x_half,y_mid,y_half; };
struct FloatModel { float c[32]; Transform tr; };
struct Int16Model { int16_t c[60]; float scale,offset; Transform tr; };
struct OnlineModel { FloatModel model; float eta; };
// Caller-owned reusable workspace: not included in the persistent model count.
struct PavaWorkspace { float means[32]; uint8_t counts[32]; };
static_assert(sizeof(Transform)==16,"Transform storage");
static_assert(sizeof(FloatModel)==144,"Float model storage");
static_assert(sizeof(Int16Model)==144,"Integer model storage");
static_assert(sizeof(OnlineModel)==148,"Online persistent storage");
static_assert(sizeof(PavaWorkspace)==160,"Projection working storage");

inline float bound(float x){return x<-1?-1:(x>1?1:x);}
inline float normalize(float adc,const Transform &t){return bound((adc-t.x_mid)/t.x_half);}
inline float physical(float y,const Transform &t){return y*t.y_half+t.y_mid;}
inline int interval(float x,int pieces,float &u){
  float p=(bound(x)+1)*(0.5f*pieces);int k=(int)p;
  if(k>=pieces)k=pieces-1;
  u=p-k;return k;
}
inline float lerp(float a,float b,float u){return a+(b-a)*u;}
inline float lut32(float x,const float *c){float u;int k=interval(x,31,u);return lerp(c[k],c[k+1],u);}
inline float lut60(float x,const Int16Model &m){
  float u;int k=interval(x,59,u);
  float a=m.c[k]*m.scale+m.offset,b=m.c[k+1]*m.scale+m.offset;
  return lerp(a,b,u);
}
inline float horner8(float x,const float *c){
  float u;int k=interval(x,8,u);c+=4*k;
  return ((c[0]*u+c[1])*u+c[2])*u+c[3];
}
struct CubicLocal { int k; float u,a01,a12,a23,a012,a123; };
inline CubicLocal cubic_local(float x){
  CubicLocal s;s.k=interval(x,29,s.u);float u=s.u;
  s.a01=(u+2)*(1.f/3);s.a12=(u+1)*(1.f/3);s.a23=u*(1.f/3);
  s.a012=(u+1)*.5f;s.a123=u*.5f;
  if(s.k==0){s.a01=u;s.a12=u*.5f;s.a012=u;}
  else if(s.k==1)s.a01=(u+1)*.5f;
  else if(s.k==27)s.a23=u*.5f;
  else if(s.k==28){s.a12=(u+1)*.5f;s.a23=u;s.a123=u;}
  return s;
}
inline float cubic32(float x,const float *c){
  CubicLocal s=cubic_local(x);c+=s.k;
  float d1=lerp(c[0],c[1],s.a01),d2=lerp(c[1],c[2],s.a12),d3=lerp(c[2],c[3],s.a23);
  return lerp(lerp(d1,d2,s.a012),lerp(d2,d3,s.a123),s.u);
}
// Local basis is the exact linear expansion of the same six de Boor lerps.
inline int cubic_basis(float x,float *b){
  CubicLocal s=cubic_local(x);
  float a=(1-s.u)*(1-s.a012),d=(1-s.u)*s.a012+s.u*(1-s.a123),e=s.u*s.a123;
  b[0]=a*(1-s.a01);b[1]=a*s.a01+d*(1-s.a12);
  b[2]=d*s.a12+e*(1-s.a23);b[3]=e*s.a23;
  return s.k;
}
inline void project(float *c,PavaWorkspace &w){
  uint8_t blocks=0;
  for(uint8_t i=0;i<32;++i){
    w.means[blocks]=c[i];w.counts[blocks]=1;++blocks;
    while(blocks>1&&w.means[blocks-2]<w.means[blocks-1]){
      uint8_t l=blocks-2,r=blocks-1,n=w.counts[l]+w.counts[r];
      w.means[l]=(w.means[l]*w.counts[l]+w.means[r]*w.counts[r])/n;
      w.counts[l]=n;--blocks;
    }
  }
  uint8_t at=0;
  for(uint8_t j=0;j<blocks;++j)for(uint8_t n=0;n<w.counts[j];++n)c[at++]=w.means[j];
}
inline bool decreasing(const float *c){for(int i=1;i<32;++i)if(c[i]>c[i-1])return false;return true;}
inline void initialize(OnlineModel &s,const FloatModel &initial,bool projected,PavaWorkspace &w){
  s.model=initial;s.eta=.5f;if(projected)project(s.model.c,w);
}
// Full causal labelled update; return the prediction made BEFORE seeing this label.
inline float update(OnlineModel &s,float adc,float reference,bool cubic,bool projected,PavaWorkspace &w){
  float x=normalize(adc,s.model.tr),b[4];int k,n;
  if(cubic){k=cubic_basis(x,b);n=4;}
  else {float u;k=interval(x,31,u);b[0]=1-u;b[1]=u;n=2;}
  float pred=0,norm=0;
  for(int j=0;j<n;++j){pred+=b[j]*s.model.c[k+j];norm+=b[j]*b[j];}
  float target=(reference-s.model.tr.y_mid)/s.model.tr.y_half;
  float gain=s.eta*(target-pred)/(norm+1e-6f);
  for(int j=0;j<n;++j)s.model.c[k+j]+=gain*b[j];
  if(projected)project(s.model.c,w);
  return physical(pred,s.model.tr);
}
inline float predict_online(const OnlineModel &s,float adc,bool cubic){
  float x=normalize(adc,s.model.tr);
  return physical(cubic?cubic32(x,s.model.c):lut32(x,s.model.c),s.model.tr);
}
} // namespace Matched
