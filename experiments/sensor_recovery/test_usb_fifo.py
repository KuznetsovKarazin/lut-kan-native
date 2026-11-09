"""Compile the actual C3 transport against a bounded USB FIFO simulation.

This checks byte retention, packet boundaries, idle/restart and backpressure;
it does not emulate a physical USB controller or a Windows driver.
"""
import subprocess
from pathlib import Path


def test_actual_transport_with_idle_partial_fifo_and_timeout(tmp_path):
    (tmp_path/"hal").mkdir()
    (tmp_path/"Arduino.h").write_text(r'''
#pragma once
#include <stdint.h>
#include <stddef.h>
unsigned long millis();
void delay(unsigned long);
class Print {
 public:
 virtual ~Print()=default;
 virtual size_t write(uint8_t)=0;
 virtual size_t write(const uint8_t*,size_t)=0;
};
''')
    (tmp_path/"hal/usb_serial_jtag_ll.h").write_text(r'''
#pragma once
#include <stdint.h>
extern int USB_SERIAL_JTAG;
enum {USB_SERIAL_JTAG_INTR_SERIAL_IN_EMPTY=8,USB_SERIAL_JTAG_INTR_SERIAL_OUT_RECV_PKT=4};
void usb_serial_jtag_ll_enable_bus_clock(bool);
void usb_serial_jtag_ll_disable_intr_mask(uint32_t);
int usb_serial_jtag_ll_read_rxfifo(uint8_t*,uint32_t);
int usb_serial_jtag_ll_write_txfifo(const uint8_t*,uint32_t);
int usb_serial_jtag_ll_txfifo_writable();
void usb_serial_jtag_ll_txfifo_flush();
''')
    (tmp_path/"hal/usb_phy_ll.h").write_text("#pragma once\nvoid usb_phy_ll_int_jtag_enable(int*);\n")
    source=tmp_path/"usb_test.cpp"
    source.write_text(r'''
#include "usb_bench_serial.h"
#include <algorithm>
#include <cassert>
#include <deque>
#include <string>
#include <vector>
int USB_SERIAL_JTAG=0;
unsigned long now=0;
bool busy=false,host_reads=true;
size_t write_limit=63;
uint32_t disabled=0;
std::string fifo,received;
std::vector<size_t> packets;
std::deque<uint8_t> rx;
unsigned long millis(){return now;}
void delay(unsigned long ms){now+=ms;if(host_reads && busy){received+=fifo;fifo.clear();busy=false;}}
void usb_serial_jtag_ll_enable_bus_clock(bool enabled){assert(enabled);}
void usb_phy_ll_int_jtag_enable(int *dev){assert(dev==&USB_SERIAL_JTAG);}
void usb_serial_jtag_ll_disable_intr_mask(uint32_t mask){disabled|=mask;}
int usb_serial_jtag_ll_txfifo_writable(){return !busy;}
int usb_serial_jtag_ll_write_txfifo(const uint8_t *src,uint32_t n){
 assert(!busy);assert(n<=63);
 const size_t sent=std::min((size_t)n,write_limit);
 fifo.append((const char*)src,sent);return sent;
}
void usb_serial_jtag_ll_txfifo_flush(){
 if(!fifo.empty() && !busy){assert(fifo.size()<64);packets.push_back(fifo.size());busy=true;}
}
int usb_serial_jtag_ll_read_rxfifo(uint8_t *dst,uint32_t n){
 unsigned i=0;while(i<n && !rx.empty()){dst[i++]=rx.front();rx.pop_front();}return i;
}
void send(UsbBenchSerial &usb,const std::string &s){
 assert(usb.write((const uint8_t*)s.data(),s.size())==s.size());
}
int main(){
 UsbBenchSerial usb;assert(usb.begin());assert(disabled==12);
 const std::string ready="READY,SENSOR_USB_V3,esp32c3\r\n";
 const std::string header="SENSOR_BENCH_V1,synthetic_ntc_frontend,seed20261108,unit0,B75\r\n";
 send(usb,ready);assert(received==ready);
 delay(1000); // Reply must restart after the FIFO was idle and no TX IRQ exists.
 send(usb,header);assert(received==ready+header);
 std::string expected=received;
 for(unsigned n:{62,63,64,65,126,127,128,255,256,513}){
   const std::string line=std::string(n-1,'x')+"\n";
   send(usb,line);expected+=line;assert(received==expected);
 }
 write_limit=7;send(usb,header);expected+=header;assert(received==expected);
 const std::string command="R,d2d81a1695da4b559ea86a493bd379ae\n";
 for(uint8_t c:command)rx.push_back(c);
 std::string readback;int c;while((c=usb.read())>=0)readback+=(char)c;
 assert(readback==command && usb.read()==-1);
 host_reads=false;const auto start=millis();
 const uint8_t blocked[]={'z','\n'};
 assert(usb.write(blocked,sizeof(blocked))<sizeof(blocked));
 assert(millis()-start==2000);
 assert(usb.write((uint8_t)'z')==0 && usb.read()==-1); // Sticky transport failure.
}
''')
    include=Path(__file__).resolve().parent/"mcu/include"
    exe=tmp_path/"usb_fifo_test"
    subprocess.run(["g++","-std=c++11","-Wall","-Wextra","-Werror",
                    "-I",str(tmp_path),"-I",str(include),str(source),"-o",str(exe)],check=True)
    subprocess.run([str(exe)],check=True)
