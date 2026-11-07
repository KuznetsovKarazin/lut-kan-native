#pragma once
#include <math.h>
#include <stdint.h>

static inline float bounded_x(float x) { return x < -1 ? -1 : (x > 1 ? 1 : x); }
static inline int lut_index(float x, int n, float *w) {
  float pos=(bounded_x(x)+1)*(0.5f*(n-1));
  int k=(int)pos; if(k>=n-1)k=n-2;*w=pos-k;return k;
}
template<class T> static inline float uniform_eval(float x,const T *c,int n,float scale,float offset) {
  float w;int k=lut_index(x,n,&w);
  float a=(float)c[k]*scale+offset,b=(float)c[k+1]*scale+offset;
  return a+(b-a)*w;
}
static inline float irregular_eval(float x,const float *knots,const int16_t *c,int n,float scale,float offset) {
  x=bounded_x(x);int lo=0,hi=n-1;
  while(hi-lo>1){int mid=(lo+hi)/2;if(x<knots[mid])hi=mid;else lo=mid;}
  float w=(x-knots[lo])/(knots[hi]-knots[lo]);
  float a=c[lo]*scale+offset,b=c[hi]*scale+offset;return a+(b-a)*w;
}
static inline float chebyshev_eval(float x,const float *c,int n) {
  float b1=0,b2=0;
  for(int j=n-1;j>=1;--j){float b=2*x*b1-b2+c[j];b2=b1;b1=b;}
  return x*b1-b2+c[0];
}
static inline float cubic_knot(int k,int n){
  if(k<=3)return -1;
  if(k>=n)return 1;
  return -1+2.0f*(k-3)/(n-3);
}
static inline float cubic_eval(float x,const float *c,int n) {
  x=bounded_x(x);int span=3+(int)((x+1)*.5f*(n-3));if(span>=n)span=n-1;
  float d[4];for(int j=0;j<4;++j)d[j]=c[span-3+j];
  for(int r=1;r<=3;++r)for(int j=3;j>=r;--j){
    float lo=cubic_knot(span-3+j,n),hi=cubic_knot(span+1+j-r,n);
    float a=(x-lo)/(hi-lo);d[j]=(1-a)*d[j-1]+a*d[j];
  }
  return d[3];
}
// Euclidean projection of a two-node SGD step onto the local decreasing cone.
// No regularization: this is NOT equivalent to the offline fitted optimizer.
static inline void monotone_step(float *c,int n,float x,float target,float rate) {
  float w;int k=lut_index(x,n,&w);float a=c[k],b=c[k+1];
  float error=a+(b-a)*w-target;
  a-=2*rate*error*(1-w);b-=2*rate*error*w;
  float upper=k?c[k-1]:INFINITY,lower=k+2<n?c[k+2]:-INFINITY;
  if(a<b){float mean=.5f*(a+b);a=mean;b=mean;}
  a=fminf(upper,fmaxf(lower,a));b=fminf(upper,fmaxf(lower,b));
  c[k]=a;c[k+1]=b;
}
