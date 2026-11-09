#pragma once
#include <Arduino.h>
#include <hal/usb_serial_jtag_ll.h>
#if defined(CONFIG_IDF_TARGET_ESP32S3)
// IDF 4.4's S3 PHY header contains volatile register copies valid in C but
// rejected by C++. Keep the vendor helper in a C translation unit.
extern "C" void sensor_bench_usb_phy_enable(void);
#else
#include <hal/usb_phy_ll.h>
#endif

// Single-owner polling transport for this command/response benchmark only.
// Avoid the IDF 4.4 interrupt-driven TX queue: an empty queue can leave the
// next response waiting for an interrupt. Never install HWCDC or that driver.
// Short (<64-byte) packets also terminate each USB transfer on Windows.
class UsbBenchSerial : public Print {
  uint8_t line[256];
  size_t used=0;
  bool ready=false;
  bool waitWritable(uint32_t start) {
    while(!usb_serial_jtag_ll_txfifo_writable()) {
      if((uint32_t)(millis()-start)>=2000){ready=false;return false;}
      delay(1); // Outside all measured inference/update loops.
    }
    return true;
  }
 public:
  bool begin() {
    usb_serial_jtag_ll_enable_bus_clock(true);
#if defined(CONFIG_IDF_TARGET_ESP32S3)
    sensor_bench_usb_phy_enable();
#else
    usb_phy_ll_int_jtag_enable(&USB_SERIAL_JTAG);
#endif
    usb_serial_jtag_ll_disable_intr_mask(USB_SERIAL_JTAG_INTR_SERIAL_IN_EMPTY |
                                        USB_SERIAL_JTAG_INTR_SERIAL_OUT_RECV_PKT);
    // Flush any earlier boot text; do not reset, disconnect or purge RX.
    usb_serial_jtag_ll_txfifo_flush();
    used=0;ready=true;
    return true;
  }
  int read() {
    uint8_t c;
    return ready && usb_serial_jtag_ll_read_rxfifo(&c,1)==1 ? c : -1;
  }
  void flushLine() {
    if(!ready || !used)return;
    size_t sent=0;const uint32_t start=millis();
    while(sent<used) {
      if(!waitWritable(start))break;
      const size_t remaining=used-sent;
      const size_t packet=remaining<63?remaining:63;
      int n=usb_serial_jtag_ll_write_txfifo(line+sent,packet);
      if(n<=0){ready=false;break;}
      sent+=(size_t)n;
      usb_serial_jtag_ll_txfifo_flush();
      if(!waitWritable(start))break; // Host has consumed this packet.
    }
    used=0; // Failure is sticky; host validation must reject an incomplete run.
  }
  size_t write(uint8_t c) override {
    if(!ready)return 0;
    line[used++]=c;
    if(c=='\n' || used==sizeof(line))flushLine();
    return ready?1:0;
  }
  size_t write(const uint8_t *data,size_t n) override {
    size_t i=0;for(;i<n;++i)if(!write(data[i]))break;
    return i;
  }
};
