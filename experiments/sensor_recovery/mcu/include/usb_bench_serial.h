#pragma once
#include <Arduino.h>
#include <driver/usb_serial_jtag.h>

// Native ESP-IDF USB Serial/JTAG driver, with complete-line writes.
// No Arduino HWCDC begin/end, USB re-enumeration or DTR-dependent startup.
class UsbBenchSerial : public Print {
  uint8_t line[256];
  size_t used=0;
  bool ready=false;
 public:
  bool begin() {
    usb_serial_jtag_driver_config_t config;
    config.rx_buffer_size=2048;config.tx_buffer_size=2048;
    ready=usb_serial_jtag_driver_install(&config)==ESP_OK;
    return ready;
  }
  int read() {
    uint8_t c;
    return ready && usb_serial_jtag_read_bytes(&c,1,0)==1 ? c : -1;
  }
  void flushLine() {
    if(!ready || !used)return;
    size_t sent=0;
    while(sent<used) {
      int n=usb_serial_jtag_write_bytes(line+sent,used-sent,pdMS_TO_TICKS(2000));
      if(n<=0)break;
      sent+=(size_t)n;
    }
    used=0; // A broken transfer fails host validation; never fabricate output.
  }
  size_t write(uint8_t c) override {
    if(!ready)return 0;
    line[used++]=c;
    if(c=='\n' || used==sizeof(line))flushLine();
    return 1;
  }
  size_t write(const uint8_t *data,size_t n) override {
    size_t i=0;for(;i<n;++i)if(!write(data[i]))break;
    return i;
  }
};
