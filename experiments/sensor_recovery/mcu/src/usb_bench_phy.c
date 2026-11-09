/* Compile the unmodified vendor PHY helper as C: its S3 register-copy
 * implementation in IDF 4.4 is not C++-compatible. No arithmetic kernel lives
 * here; this is initialization before command handling and measurements. */
#if defined(ARDUINO_ARCH_ESP32)
#include "sdkconfig.h"
#if defined(CONFIG_IDF_TARGET_ESP32S3)
#include <hal/usb_phy_ll.h>

void sensor_bench_usb_phy_enable(void)
{
    usb_phy_ll_int_jtag_enable(&USB_SERIAL_JTAG);
}
#endif
#endif
