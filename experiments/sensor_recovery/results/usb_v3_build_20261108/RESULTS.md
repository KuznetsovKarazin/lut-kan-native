# ESP32-C3 USB v3: READY-only capture and transport revision

## Observed evidence

The user's v2 capture on COM11 contains exactly 29 bytes:
`READY,SENSOR_USB_V2,esp32c3\r\n`. Its raw SHA256 matches the uploaded sidecar.
The host recorded a successful write of `H`, followed by a correctly formed
`R,<32-hex-session-id>`. The script hash matches parent commit
`6361c1c7afdc342c850aed596b52ab75d9454732` with Windows CRLF line endings.
The timeout occurred 15 seconds after the run command. The unchanged uploaded
files and a compact analysis are retained in `failed_v2_capture/`.

This confirms bidirectional USB for the first short exchange only. It does
not establish whether the run command was received or the benchmark executed.
There are no usable C3 timing or accuracy records in this capture. The former
timeout wording did not distinguish READY from acknowledgement of a run.

## Source-level diagnosis and change

The pinned Arduino SDK uses IDF 4.4.7. Its interrupt-driven USB TX handler
clears the empty status and disables the TX interrupt when the queue empties;
a subsequent write only queues bytes and enables that interrupt. The lack
of a fresh transmission event is a plausible explanation for the READY-only
symptom. The newer Espressif implementation explicitly flushes on idle and
handles the transition differently. This is a source-level hypothesis, not
a hardware-confirmed diagnosis. The prebuilt SDK also enables a secondary USB
console; competing console/driver access is another integration concern.

V3 uses a single-owner polling FIFO transport for this benchmark, without
installing the interrupt-driven USB driver or HWCDC. It sends packets of at
most 63 bytes, explicitly flushes each packet, retains partial writes, and
waits for FIFO space with a bounded timeout. A transport failure is sticky
and an incomplete capture cannot validate. Polling/waiting and all prints
remain outside the measured inference/update loops.

The host protocol is now:

1. `H` -> `READY,SENSOR_USB_V3,esp32c3`.
2. `Q,<session-id>` -> `ECHO,<same-session-id>`; exercises a full-length command.
3. One `R,<session-id>` -> `ACCEPTED,<same-session-id>`.
4. Benchmark header, session/transport metadata, measurements and END.

Each missing response gets a distinct timeout and the sidecar retains the
failed phase. Incorrect echoes, incompatible firmware and mismatched run
acknowledgements are rejected. No automatic repeat of a run is performed.
Legacy Mega captures and standalone parsing remain compatible.

This FIFO transport is intentionally limited to a single sequential benchmark
task. It must not share the peripheral with other writers, a USB console,
HWCDC or an interrupt-driven USB driver during capture. It is not proposed
as a general-purpose replacement for ESP-IDF serial drivers.

## Verification

37 focused tests and eight model subtests pass. The actual C++ FIFO transport
is compiled against a bounded FIFO simulation that checks idle/restart,
62/63/64/65-byte boundaries, longer lines, partial writes, byte preservation,
the uploaded run token and a non-reading host timeout. Other tests cover the
three acknowledgement phases, old firmware, failed echo, truncated output,
disconnects and distinct capture identities. These are host tests, not USB
hardware emulation or a successful Windows capture.

The numerical kernels, model/fixture headers and measured function text are
unchanged from the parent. Both retained Mega captures parse to their exact
saved JSON. Four cross-compilations pass; source/ELF hashes, tool versions,
build logs and verification metadata are retained alongside this report.

| Build | Flash, bytes | Static RAM, bytes |
| --- | ---: | ---: |
| Mega2560 baseline | 730 | 17 |
| Mega2560 full harness | 28,990 | 1,280 |
| ESP32-C3 baseline | 218,848 | 13,484 |
| ESP32-C3 full harness, USB v3 | 264,798 | 13,964 |

These are compiler size reports, not peak runtime memory. Existing per-model
size measurements are preserved. V3 does not allocate the v2 RX/TX ring
buffers. The first successful physical v3 capture and a repeat session remain
pending. Follow [the Windows instructions](../../mcu/README_RU.md); reflash
once, then capture with `--start-command` without RESET or BOOT.

Primary references:

- [ESP-IDF v4.4.7 USB driver, blob 3905157](https://github.com/espressif/esp-idf/blob/v4.4.7/components/driver/usb_serial_jtag.c)
- [Current Espressif USB driver, blob 4fc315e inspected](https://github.com/espressif/esp-idf/blob/master/components/esp_driver_usb_serial_jtag/src/usb_serial_jtag.c)
- [Espressif on competing USB console/driver access](https://github.com/espressif/esp-idf/issues/12605#issuecomment-1817449614)
- [ESP32-C3 FIFO primitives, IDF v4.4.7](https://github.com/espressif/esp-idf/blob/v4.4.7/components/hal/esp32c3/include/hal/usb_serial_jtag_ll.h)
