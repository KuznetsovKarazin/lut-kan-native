#ifdef SENSOR_REVISION_SUITE
#include <Arduino.h>
#if defined(__AVR__)
#include <avr/pgmspace.h>
#define MATCHED_FLASH PROGMEM
#define MATCHED_INITIAL_FLASH PROGMEM
static inline float mf(const float *p){return pgm_read_float(p);}
#define REVISION_BOARD "mega2560"
#elif defined(CONFIG_IDF_TARGET_ESP32C3)
#define MATCHED_FLASH
static inline float mf(const float *p){return *p;}
#define REVISION_BOARD "esp32c3"
#else
#define MATCHED_FLASH
static inline float mf(const float *p){return *p;}
#define REVISION_BOARD "esp32s3"
#endif
#include "revision_kernels.h"
#include "revision_identity.h"
#define M_NOINLINE __attribute__((noinline,unused))
static M_NOINLINE float predict_0(float adc){return adc;}
static M_NOINLINE float predict_1(float adc){return Matched::physical(Matched::lut32(Matched::normalize(adc,matched_lut_float32.tr),matched_lut_float32.c),matched_lut_float32.tr);}
static M_NOINLINE float predict_2(float adc){return Matched::physical(Matched::lut60(Matched::normalize(adc,matched_lut_int16.tr),matched_lut_int16),matched_lut_int16.tr);}
static M_NOINLINE float predict_3(float adc){return Matched::physical(Matched::cubic32(Matched::normalize(adc,matched_cubic_compact.tr),matched_cubic_compact.c),matched_cubic_compact.tr);}
static M_NOINLINE float predict_4(float adc){return Matched::physical(Matched::horner8(Matched::normalize(adc,matched_cubic_horner.tr),matched_cubic_horner.c),matched_cubic_horner.tr);}
static M_NOINLINE float predict_5(float adc){return Revision::polynomial(adc);}
static M_NOINLINE float predict_6(float adc){return Revision::physics(adc);}
#include "command_protocol.h"
#if defined(CONFIG_IDF_TARGET_ESP32C3)
#include "usb_bench_serial.h"
static UsbBenchSerial bench_usb;
#ifdef Serial
#undef Serial
#endif
#define Serial bench_usb
#endif
static const char *static_names[]={"overhead","lut32_float32","lut60_int16","cubic32_unrolled","horner8","polynomial","physics"};
typedef float (*Predict)(float);
static const Predict predictors[]={predict_0,predict_1,predict_2,predict_3,predict_4,predict_5,predict_6};
static volatile float checksum;
static bool static_accuracy(int method){
  float sum=0,sq=0,maxerr=0,gap=0;
  for(int i=0;i<MATCHED_GRID_N;++i){
    float pred=predictors[method](mf(matched_adc+i));
    if(!isfinite(pred)){gap=INFINITY;continue;}
    float err=pred-mf(matched_truth+i);sum+=fabsf(err);sq+=err*err;maxerr=fmaxf(maxerr,fabsf(err));
    gap=fmaxf(gap,fabsf(pred-(method<=4?mf(&matched_expected[method-1][i]):mf(&revision_expected[method-5][i]))));
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
static void run_benchmark(const char *session){
 Serial.println("SENSOR_BENCH_V1,synthetic_ntc_frontend,seed2026110801,unit0,B75");
 Serial.println("SUITE,matched_references_v1");Serial.print("SESSION,");Serial.println(session);
 Serial.print("BOARD,");Serial.println(REVISION_BOARD);
 Serial.print("CPU_HZ,");Serial.println(F_CPU);
 Serial.print("FIXTURE_SHA256,");Serial.println(REVISION_FIXTURE_SHA256);
 Serial.println("PRECISION,float32");
 for(int method=1;method<=6;++method){Serial.print("PAYLOAD,");Serial.print(static_names[method]);Serial.print(',');Serial.println(method<=4?144:method==5?16+4*REVISION_POLY_N:32);}
 static_timing(0);
 for(int method=1;method<=6;++method)if(static_accuracy(method))static_timing(method);
 Serial.println("END_SENSOR_BENCH");
}
void setup(){
#if defined(CONFIG_IDF_TARGET_ESP32C3)
 bench_usb.begin();
#else
 Serial.begin(115200);
#endif
}
void loop(){
 static BenchCommand command;static char last_session[33]={0};int c;
#if defined(CONFIG_IDF_TARGET_ESP32C3)
 while((c=bench_usb.read())>=0){
#else
 while(Serial.available()&&(c=Serial.read())>=0){
#endif
 if(!command.push((char)c))continue;
 if(command.hello()){Serial.print("READY,SENSOR_REFERENCES_V1,");Serial.println(REVISION_BOARD);}
 else if(const char *token=command.probe()){Serial.print("ECHO,");Serial.println(token);}
 else if(const char *session=command.session()){
  if(strcmp(last_session,session)==0){Serial.print("ALREADY_RAN,");Serial.println(session);}
  else{strcpy(last_session,session);Serial.print("ACCEPTED,");Serial.println(last_session);run_benchmark(last_session);}
 }else Serial.println("ERROR,INVALID_COMMAND");
 }
 delay(1);
}
#endif
