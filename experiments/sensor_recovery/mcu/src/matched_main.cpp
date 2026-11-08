#ifdef SENSOR_MATCHED_SUITE
#include <Arduino.h>
#if defined(__AVR__)
#include <avr/pgmspace.h>
#define MATCHED_FLASH PROGMEM
#define MATCHED_INITIAL_FLASH PROGMEM
static inline float mf(const float *p){return pgm_read_float(p);}
#else
#define MATCHED_FLASH
static inline float mf(const float *p){return *p;}
#endif
#include "matched_fixture.h"
#include "matched_identity.h"
#define M_NOINLINE __attribute__((noinline,unused))
static Matched::OnlineModel online;
static Matched::PavaWorkspace workspace;
static M_NOINLINE float predict_0(float adc){return adc;}
static M_NOINLINE float predict_1(float adc){return Matched::physical(Matched::lut32(Matched::normalize(adc,matched_lut_float32.tr),matched_lut_float32.c),matched_lut_float32.tr);}
static M_NOINLINE float predict_2(float adc){return Matched::physical(Matched::lut60(Matched::normalize(adc,matched_lut_int16.tr),matched_lut_int16),matched_lut_int16.tr);}
static M_NOINLINE float predict_3(float adc){return Matched::physical(Matched::cubic32(Matched::normalize(adc,matched_cubic_compact.tr),matched_cubic_compact.c),matched_cubic_compact.tr);}
static M_NOINLINE float predict_4(float adc){return Matched::physical(Matched::horner8(Matched::normalize(adc,matched_cubic_horner.tr),matched_cubic_horner.c),matched_cubic_horner.tr);}
static M_NOINLINE float update_0(float adc,float reference){return Matched::update(online,adc,reference,false,false,workspace);}
static M_NOINLINE float update_1(float adc,float reference){return Matched::update(online,adc,reference,false,true,workspace);}
static M_NOINLINE float update_2(float adc,float reference){return Matched::update(online,adc,reference,true,false,workspace);}
static M_NOINLINE float update_3(float adc,float reference){return Matched::update(online,adc,reference,true,true,workspace);}
static __attribute__((unused)) void reset_model(int method){
#if defined(__AVR__)
  // Initialization snapshot stays in Flash, not a second persistent SRAM copy.
  memcpy_P(&online.model,method>=2?&matched_initial_cubic:&matched_initial_lut,sizeof(online.model));
  online.eta=.5f;if(method%2)Matched::project(online.model.c,workspace);
#else
  Matched::initialize(online,method>=2?matched_initial_cubic:matched_initial_lut,method%2,workspace);
#endif
}

#ifdef MATCHED_FOOTPRINT
#ifndef MATCHED_METHOD_ID
#error MATCHED_METHOD_ID required
#endif
static volatile float footprint_adc=.5f,footprint_reference=25,footprint_output;
void setup(){
#if MATCHED_METHOD_ID>=11
  reset_model(MATCHED_METHOD_ID-11);
#endif
}
void loop(){
#if MATCHED_METHOD_ID>=11
  #if MATCHED_METHOD_ID==11
  footprint_output=update_0(footprint_adc,footprint_reference);
#elif MATCHED_METHOD_ID==12
  footprint_output=update_1(footprint_adc,footprint_reference);
#elif MATCHED_METHOD_ID==13
  footprint_output=update_2(footprint_adc,footprint_reference);
#elif MATCHED_METHOD_ID==14
  footprint_output=update_3(footprint_adc,footprint_reference);
#endif
#else
  #if MATCHED_METHOD_ID==0
  footprint_output=predict_0(footprint_adc);
#elif MATCHED_METHOD_ID==1
  footprint_output=predict_1(footprint_adc);
#elif MATCHED_METHOD_ID==2
  footprint_output=predict_2(footprint_adc);
#elif MATCHED_METHOD_ID==3
  footprint_output=predict_3(footprint_adc);
#elif MATCHED_METHOD_ID==4
  footprint_output=predict_4(footprint_adc);
#endif
#endif
}
#else
#include "command_protocol.h"
#if !defined(__AVR__)
#include "usb_bench_serial.h"
static UsbBenchSerial bench_usb;
#ifdef Serial
#undef Serial
#endif
#define Serial bench_usb
#endif
static const char *static_names[]={"overhead","lut32_float32","lut60_int16","cubic32_unrolled","horner8"};
static const char *online_names[]={"lut_nlms","lut_projected_nlms","cubic_nlms","cubic_projected_nlms"};
static const char *mode_names[]={"none","step","ramp"};
typedef float (*Predict)(float);
typedef float (*Update)(float,float);
static const Predict predictors[]={predict_0,predict_1,predict_2,predict_3,predict_4};
static const Update updates[]={update_0,update_1,update_2,update_3};
static volatile float checksum;

static float state_gap(int mode,int method,int check){
  float gap=0;
  for(int j=0;j<32;++j){
    if(!isfinite(online.model.c[j]))return INFINITY;
    gap=fmaxf(gap,fabsf(online.model.c[j]-mf(&matched_reference_states[mode][method][check][j]))*online.model.tr.y_half);
  }
  return gap;
}
static bool order_ok(int method){return !(method%2)||Matched::decreasing(online.model.c);}
static void state_row(int mode,int method,int check){
  float gap=state_gap(mode,method,check);bool ok=isfinite(gap)&&gap<=.002f&&order_ok(method);
  Serial.print("STATE,");Serial.print(mode_names[mode]);Serial.print(',');Serial.print(online_names[method]);
  Serial.print(',');Serial.print(matched_checkpoints[check]);Serial.print(',');Serial.print(gap,8);
  Serial.print(',');Serial.print(method%2?(order_ok(method)?"MONOTONE":"FAIL"):"UNCONSTRAINED");
  Serial.print(',');Serial.println(ok?"PASS":"FAIL");
}
static bool static_accuracy(int method){
  float sum=0,sq=0,maxerr=0,gap=0;
  for(int i=0;i<MATCHED_GRID_N;++i){
    float pred=predictors[method](mf(matched_adc+i));
    if(!isfinite(pred)){gap=INFINITY;continue;}
    float err=pred-mf(matched_truth+i);sum+=fabsf(err);sq+=err*err;maxerr=fmaxf(maxerr,fabsf(err));
    gap=fmaxf(gap,fabsf(pred-mf(&matched_expected[method-1][i])));
  }
  Serial.print("ACCURACY,");Serial.print(static_names[method]);Serial.print(',');Serial.print(sum/MATCHED_GRID_N,7);
  Serial.print(',');Serial.print(sqrtf(sq/MATCHED_GRID_N),7);Serial.print(',');Serial.print(maxerr,7);
  Serial.print(',');Serial.print(gap,8);Serial.print(',');Serial.println(isfinite(gap)&&gap<=.002f?"PASS":"FAIL");
  return isfinite(gap)&&gap<=.002f;
}
static void static_timing(int method){
  Predict predict=predictors[method];
  for(int r=0;r<10;++r){
    float sum=0;int index=r;uint32_t start=micros();
    for(uint16_t i=0;i<2048;++i){index+=79;if(index>=MATCHED_GRID_N)index-=MATCHED_GRID_N;sum+=predict(mf(matched_adc+index));}
    uint32_t elapsed=micros()-start;checksum=sum;
    Serial.print("TIME,");Serial.print(static_names[method]);Serial.print(',');Serial.print(r);
    Serial.print(",2048,");Serial.print(elapsed);Serial.print(',');Serial.println(sum,6);delay(1);
  }
}
static bool online_accuracy(int mode,int method){
  reset_model(method);state_row(mode,method,0);int check=1;
  bool ok=state_gap(mode,method,0)<=.002f&&order_ok(method);
  for(int t=1;t<=MATCHED_STEPS;++t){
    updates[method](mf(&matched_stream_adc[mode][t-1]),mf(&matched_stream_reference[mode][t-1]));
    if(check<MATCHED_CHECKS&&t==(int)matched_checkpoints[check]){
      state_row(mode,method,check);float gap=state_gap(mode,method,check);
      ok=ok&&isfinite(gap)&&gap<=.002f&&order_ok(method);++check;
    }
  }
  float sum=0,sq=0,maxerr=0;
  for(int i=0;i<MATCHED_GRID_N;++i){
    float pred=Matched::predict_online(online,mf(&matched_final_adc[mode][i]),method>=2);
    if(!isfinite(pred)){ok=false;continue;}
    float e=pred-mf(matched_truth+i);sum+=fabsf(e);sq+=e*e;maxerr=fmaxf(maxerr,fabsf(e));
  }
  float values[3]={sum/MATCHED_GRID_N,sqrtf(sq/MATCHED_GRID_N),maxerr},gap=0;
  for(int i=0;i<3;++i)gap=fmaxf(gap,fabsf(values[i]-mf(&matched_reference_metrics[mode][method][i])));
  ok=ok&&isfinite(gap)&&gap<=.002f;
  Serial.print("UPDATE_ACCURACY,");Serial.print(mode_names[mode]);Serial.print(',');Serial.print(online_names[method]);
  for(int i=0;i<3;++i){Serial.print(',');Serial.print(values[i],7);}
  Serial.print(',');Serial.print(gap,8);Serial.print(',');Serial.println(ok?"PASS":"FAIL");return ok;
}
static void online_timing(int mode,int method){
  Update step=updates[method];
  for(int r=0;r<10;++r){
    reset_model(method);float sum=0;uint32_t start=micros();
    for(int t=0;t<MATCHED_STEPS;++t)sum+=step(mf(&matched_stream_adc[mode][t]),mf(&matched_stream_reference[mode][t]));
    uint32_t elapsed=micros()-start;checksum=sum;
    float gap=state_gap(mode,method,7);bool ok=isfinite(gap)&&gap<=.002f&&order_ok(method);
    Serial.print("UPDATE_TIME,");Serial.print(mode_names[mode]);Serial.print(',');Serial.print(online_names[method]);
    Serial.print(',');Serial.print(r);Serial.print(",384,");Serial.print(elapsed);Serial.print(',');Serial.print(sum,6);
    Serial.print(',');Serial.print(gap,8);Serial.print(',');Serial.println(ok?"PASS":"FAIL");delay(1);
  }
}
static void update_overhead(int mode){
  for(int r=0;r<10;++r){
    float sum=0;uint32_t start=micros();
    for(int t=0;t<MATCHED_STEPS;++t)sum+=mf(&matched_stream_adc[mode][t])+mf(&matched_stream_reference[mode][t]);
    uint32_t elapsed=micros()-start;checksum=sum;
    Serial.print("UPDATE_OVERHEAD,");Serial.print(mode_names[mode]);Serial.print(',');Serial.print(r);
    Serial.print(",384,");Serial.print(elapsed);Serial.print(',');Serial.println(sum,6);delay(1);
  }
}
static void run_benchmark(const char *session){
  Serial.println("SENSOR_BENCH_V1,synthetic_ntc_frontend,seed2026110801,unit0,B75");
  Serial.println("SUITE,matched_adaptation_v1");Serial.print("SESSION,");Serial.println(session);
#if defined(__AVR__)
  Serial.println("BOARD,mega2560");Serial.println("TRANSPORT,uart-command-matched-v1");
#else
  Serial.println("BOARD,esp32c3");Serial.println("TRANSPORT,usb-jtag-poll-command-matched-v1");
#endif
  Serial.print("CPU_HZ,");Serial.println(F_CPU);
  Serial.print("FIXTURE_SHA256,");Serial.println(MATCHED_FIXTURE_SHA256);
  Serial.println("PRECISION,float32");Serial.println("NOTE,synthetic references; gross timings; no physical sensor/ADC; stack unmeasured");
  for(int method=1;method<=4;++method){Serial.print("PAYLOAD,");Serial.print(static_names[method]);Serial.println(",144");}
  for(int method=0;method<4;++method){Serial.print("UPDATE_STORAGE,");Serial.print(online_names[method]);Serial.print(",148,");Serial.println(method%2?160:0);}
  static_timing(0);
  for(int method=1;method<=4;++method)if(static_accuracy(method))static_timing(method);
  for(int mode=0;mode<3;++mode){
    update_overhead(mode);
    for(int method=0;method<4;++method)if(online_accuracy(mode,method))online_timing(mode,method);
  }
  Serial.println("END_SENSOR_BENCH");
}
void setup(){
#if defined(__AVR__)
  Serial.begin(115200);
#else
  bench_usb.begin();
#endif
}
void loop(){
  static BenchCommand command;static char last_session[33]={0};int c;
#if defined(__AVR__)
  while(Serial.available()&&(c=Serial.read())>=0){
#else
  while((c=bench_usb.read())>=0){
#endif
    if(!command.push((char)c))continue;
    if(command.hello()){
#if defined(__AVR__)
      Serial.println("READY,SENSOR_MATCHED_V1,mega2560");
#else
      Serial.println("READY,SENSOR_MATCHED_V1,esp32c3");
#endif
    }else if(const char *token=command.probe()){Serial.print("ECHO,");Serial.println(token);}
    else if(const char *session=command.session()){
      if(strcmp(last_session,session)==0){Serial.print("ALREADY_RAN,");Serial.println(session);}
      else {strcpy(last_session,session);Serial.print("ACCEPTED,");Serial.println(last_session);run_benchmark(last_session);}
    }else Serial.println("ERROR,INVALID_COMMAND");
  }
  delay(1);
}
#endif // footprint
#endif // matched suite
