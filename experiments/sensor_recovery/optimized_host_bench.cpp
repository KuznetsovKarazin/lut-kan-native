// Host timing only. Do not interpret these durations as MCU measurements.
#include "models.h"
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <iomanip>
static volatile float benchmark_sink=0;
int main(int argc,char **argv){
  const int calls=argc>1?std::atoi(argv[1]):262144;
  const int batches=argc>2?std::atoi(argv[2]):15;
  if(calls<2048 || batches<3)return 2;
  for(int m=0;m<entry_count;++m){
    float sum=0;for(int i=0;i<8192;++i)sum+=entries[m].predict(fixture_adc[i%FIXTURE_N]);
    benchmark_sink=sum;
  }
  std::cout<<"method,batch,calls,total_ns,checksum\n"<<std::setprecision(9);
  for(int r=0;r<batches;++r)for(int position=0;position<entry_count;++position){
    int m=(position+r)%entry_count,index=r%FIXTURE_N;float sum=0;
    auto start=std::chrono::steady_clock::now();
    for(int i=0;i<calls;++i){index+=79;if(index>=FIXTURE_N)index-=FIXTURE_N;
      sum+=entries[m].predict(fixture_adc[index]);}
    auto stop=std::chrono::steady_clock::now();benchmark_sink=sum;
    auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count();
    std::cout<<entries[m].name<<','<<r<<','<<calls<<','<<ns<<','<<sum<<'\n';
  }
}
