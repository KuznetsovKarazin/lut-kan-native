#pragma once
#include "fixture.h"
#include "kernels.h"
#ifndef BENCH_MODEL
#define BENCH_MODEL -1
#endif
#define NOINLINE __attribute__((noinline))
static inline float norm(float adc){return (adc-x_mid)/x_half;}
static inline float physical(float y){return y*y_half+y_mid;}
static NOINLINE float predict_0(float adc){return adc;}
#if BENCH_MODEL == -1 || BENCH_MODEL == 1
static NOINLINE float predict_1(float adc){
  adc=fminf(1-1e-7f,fmaxf(1e-7f,adc));float z=logf(10000*adc/(1-adc));
  return 1/(physics_c[0]+physics_c[1]*z+physics_c[2]*z*z*z)-273.15f;
}
#endif
#if BENCH_MODEL == -1 || BENCH_MODEL == 2
static NOINLINE float predict_2(float adc){return physical(chebyshev_eval(norm(adc),poly_c,POLY_N));}
#endif
#if BENCH_MODEL == -1 || BENCH_MODEL == 3
static NOINLINE float predict_3(float adc){return physical(cubic_eval(norm(adc),cubic_c,16));}
#endif
#if BENCH_MODEL == -1 || BENCH_MODEL == 4
static NOINLINE float predict_4(float adc){float x=bounded_x(norm(adc));return physical(cubic_eval(x,kan_c,16)+kan_c[16]*x/(1+expf(-x)));}
#endif
#if BENCH_MODEL == -1 || BENCH_MODEL == 5
static NOINLINE float predict_5(float adc){return physical(uniform_eval(norm(adc),lut_f32,32,1,0));}
#endif
#if BENCH_MODEL == -1 || BENCH_MODEL == 6
static NOINLINE float predict_6(float adc){return physical(uniform_eval(norm(adc),lut_int16,32,int16_scale,int16_offset));}
#endif
#if BENCH_MODEL == -1 || BENCH_MODEL == 7
static NOINLINE float predict_7(float adc){return physical(uniform_eval(norm(adc),lut_uint8,32,uint8_scale,uint8_offset));}
#endif
#if BENCH_MODEL == -1 || BENCH_MODEL == 8
static NOINLINE float predict_8(float adc){return physical(irregular_eval(norm(adc),irregular_knots,irregular_values,16,irregular_scale,irregular_offset));}
#endif
struct Entry{uint8_t id;const char *name;float (*predict)(float);const float *expected;uint16_t payload;};
#ifdef FOOTPRINT_ONLY
#define EXPECTED(n) nullptr
#else
#define EXPECTED(n) expected_##n
#endif
static const Entry entries[]={
  {0,"overhead",predict_0,EXPECTED(0),0},
#if BENCH_MODEL == -1 || BENCH_MODEL == 1
  {1,"physics",predict_1,EXPECTED(1),32},
#endif
#if BENCH_MODEL == -1 || BENCH_MODEL == 2
  {2,"polynomial",predict_2,EXPECTED(2),4*POLY_N+16},
#endif
#if BENCH_MODEL == -1 || BENCH_MODEL == 3
  {3,"cubic16",predict_3,EXPECTED(3),80},
#endif
#if BENCH_MODEL == -1 || BENCH_MODEL == 4
  {4,"kan16",predict_4,EXPECTED(4),84},
#endif
#if BENCH_MODEL == -1 || BENCH_MODEL == 5
  {5,"uniform32_float32",predict_5,EXPECTED(5),144},
#endif
#if BENCH_MODEL == -1 || BENCH_MODEL == 6
  {6,"uniform32_int16",predict_6,EXPECTED(6),88},
#endif
#if BENCH_MODEL == -1 || BENCH_MODEL == 7
  {7,"uniform32_uint8",predict_7,EXPECTED(7),56},
#endif
#if BENCH_MODEL == -1 || BENCH_MODEL == 8
  {8,"nonuniform16_int16",predict_8,EXPECTED(8),120},
#endif
};
static const int entry_count=sizeof(entries)/sizeof(entries[0]);
