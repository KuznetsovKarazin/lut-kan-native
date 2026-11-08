#pragma once
#include "kernels.h"

// Specialized to 16 coefficients, cubic degree and clamped uniform knots.
// There are 13 nonzero knot intervals. Finite inputs only, as in kernels.h.
static inline int cubic16_interval(float x,float *t) {
  float pos=(bounded_x(x)+1)*6.5f;
  int k=(int)pos;if(k>12)k=12;*t=pos-k;return k;
}
static inline float cubic_lerp(float a,float b,float w){return a+(b-a)*w;}

// Six unrolled de Boor interpolations in integer knot coordinates. Boundary
// cases account for repeated end knots. No knot divisions or added payload.
static inline float cubic16_unrolled_eval(float x,const float *c) {
  float t;int k=cubic16_interval(x,&t);
  float a01=(t+2)*(1.0f/3),a12=(t+1)*(1.0f/3),a23=t*(1.0f/3);
  float a012=(t+1)*.5f,a123=t*.5f;
  if(k==0){a01=t;a12=t*.5f;a012=t;}
  else if(k==1){a01=(t+1)*.5f;}
  else if(k==11){a23=t*.5f;}
  else if(k==12){a12=(t+1)*.5f;a23=t;a123=t;}
  float d1=cubic_lerp(c[k],c[k+1],a01);
  float d2=cubic_lerp(c[k+1],c[k+2],a12);
  float d3=cubic_lerp(c[k+2],c[k+3],a23);
  return cubic_lerp(cubic_lerp(d1,d2,a012),cubic_lerp(d2,d3,a123),t);
}

// 13 rows of [t^3,t^2,t,1] coefficients: 52 floats, not 16.
static inline float cubic16_horner_eval(float x,const float *coefficients) {
  float t;int k=cubic16_interval(x,&t);const float *c=coefficients+4*k;
  return ((c[0]*t+c[1])*t+c[2])*t+c[3];
}
