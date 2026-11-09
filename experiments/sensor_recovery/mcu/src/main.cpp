// No sensors connected: reproducible numerical firmware benchmark.
#if !defined(SENSOR_MATCHED_SUITE) && !defined(SENSOR_REVISION_SUITE)
#include <Arduino.h>
#ifndef FOOTPRINT_ONLY
#if !defined(__AVR__)
#include "usb_bench_serial.h"
#include "command_protocol.h"
static UsbBenchSerial bench_usb;
// Existing printing code uses the native driver on C3; numeric kernels are unchanged.
#ifdef Serial
#undef Serial
#endif
#define Serial bench_usb
#endif
#if defined(__AVR__)
#include <avr/pgmspace.h>
#define FIXTURE_FLASH PROGMEM
static inline float read_fixture(const float *p){return pgm_read_float(p);}
#else
#define FIXTURE_FLASH
static inline float read_fixture(const float *p){return *p;}
#endif
#include "models.h"
static volatile float checksum=0;
static const uint16_t calls=2048;
static const uint8_t repeats=10;
#if defined(__AVR__)
static int free_estimate(){extern int __heap_start;extern void *__brkval;int top;return (char*)&top-(__brkval?(char*)__brkval:(char*)&__heap_start);}
#else
static int free_estimate(){return ESP.getFreeHeap();}
#endif

static void accuracy(const Entry &e){
  float sum=0,squares=0,max_error=0,max_gap=0;bool finite=true;
  for(int i=0;i<FIXTURE_N;++i){float value=e.predict(read_fixture(fixture_adc+i));
    if(!isfinite(value)){finite=false;continue;}
    float error=value-read_fixture(fixture_truth+i);sum+=fabsf(error);squares+=error*error;
    max_error=fmaxf(max_error,fabsf(error));max_gap=fmaxf(max_gap,fabsf(value-read_fixture(e.expected+i)));}
  Serial.print("ACCURACY,");Serial.print(e.name);Serial.print(',');Serial.print(sum/FIXTURE_N,7);
  Serial.print(',');Serial.print(sqrtf(squares/FIXTURE_N),7);Serial.print(',');Serial.print(max_error,7);
  Serial.print(',');Serial.print(max_gap,7);Serial.print(',');Serial.println(finite&&max_gap<=.002f?"PASS":"FAIL");
}
static void timing(const Entry &e){
  for(int r=0;r<repeats;++r){
    float sum=0;int index=r;uint32_t start=micros();
    for(uint16_t i=0;i<calls;++i){index+=79;if(index>=FIXTURE_N)index-=FIXTURE_N;
      sum+=e.predict(read_fixture(fixture_adc+index));}
    uint32_t elapsed=micros()-start;checksum=sum;
    Serial.print("TIME,");Serial.print(e.name);Serial.print(',');Serial.print(r);Serial.print(',');
    Serial.print(calls);Serial.print(',');Serial.print(elapsed);Serial.print(',');Serial.println(sum,6);
    delay(1);
  }
}
static void update_timing(){
  float table[32];
  for(int r=0;r<repeats;++r){
    for(int i=0;i<32;++i)table[i]=lut_f32[i];
    int index=r;uint32_t start=micros();
    for(uint16_t i=0;i<calls;++i){index+=79;if(index>=FIXTURE_N)index-=FIXTURE_N;
      monotone_step(table,32,norm(read_fixture(fixture_adc+index)),
        (read_fixture(fixture_truth+index)-y_mid)/y_half,.01f);}
    uint32_t elapsed=micros()-start;float sum=0;bool mono=true;
    for(int i=0;i<32;++i){sum+=table[i];if(i&&table[i]>table[i-1])mono=false;}checksum=sum;
    Serial.print("UPDATE_COST_ONLY,");Serial.print(r);Serial.print(',');Serial.print(calls);Serial.print(',');
    Serial.print(elapsed);Serial.print(',');Serial.println(mono?"MONOTONE":"FAIL");delay(1);
  }
}
static void run_benchmark(const char *session=nullptr){
  Serial.println("SENSOR_BENCH_V1,synthetic_ntc_frontend,seed20261108,unit0,B75");
#ifdef SENSOR_OPTIMIZED_SUITE
  Serial.println("SUITE,spline_optimized_v1");
#endif
#if defined(__AVR__)
  (void)session;
  Serial.println("BOARD,mega2560");
#else
  Serial.print("SESSION,");Serial.println(session);
  Serial.println("TRANSPORT,usb-jtag-poll-command-v3");
  Serial.println("BOARD,esp32c3");
#endif
  Serial.print("CPU_HZ,");Serial.println(F_CPU);Serial.print("FREE_MEMORY_ESTIMATE,");Serial.println(free_estimate());
  Serial.println("NOTE,payload excludes firmware/stack; timings include reads and loop; no ADC acquisition");
  Serial.println("SCHEMA_ACCURACY,method,mae_C,rmse_C,max_C,max_host_gap_C,parity");
  Serial.println("SCHEMA_TIME,method,repeat,calls,total_us,checksum");
  for(int i=0;i<entry_count;++i){const Entry &e=entries[i];Serial.print("PAYLOAD,");Serial.print(e.name);Serial.print(',');Serial.println(e.payload);if(e.id)accuracy(e);timing(e);}
#if BENCH_MODEL == -1
  // Cost demonstration using synthetic targets, not a training-quality test.
  update_timing();
#endif
  Serial.println("END_SENSOR_BENCH");
}
#if defined(__AVR__)
void setup(){
  Serial.begin(115200);uint32_t start=millis();while(!Serial&&millis()-start<5000){}delay(500);
  run_benchmark();
}
void loop(){delay(1000);}
#else
void setup(){bench_usb.begin();}
void loop(){
  static BenchCommand command;
  static char last_session[33]={0};
  int c;
  while((c=bench_usb.read())>=0){
    if(!command.push((char)c))continue;
    if(command.hello())Serial.println("READY,SENSOR_USB_V3,esp32c3");
    else if(const char *token=command.probe()){
      Serial.print("ECHO,");Serial.println(token);
    }
    else if(const char *session=command.session()){
      if(strcmp(last_session,session)==0){Serial.print("ALREADY_RAN,");Serial.println(session);}
      else {strcpy(last_session,session);
        Serial.print("ACCEPTED,");Serial.println(last_session);
        run_benchmark(last_session);}
    }
    else Serial.println("ERROR,INVALID_COMMAND");
  }
  delay(1);
}
#endif
#else
// Footprint-only deployment: no Serial, fixture data or expected outputs.
#include "models.h"
static volatile float input_adc=.5f,output_quantity=0;
void setup(){}
void loop(){output_quantity=entries[entry_count-1].predict(input_adc);}
#endif
#endif // previous benchmark suites
