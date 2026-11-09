# ESP32-C3 USB capture revision

The user successfully flashed the C3 but received Windows `ClearCommError
failed` while reading COM11 after pressing RESET. This identifies an unusable
open serial handle, not a numerical-model failure. Native USB disconnection
and re-enumeration is the likely mechanism; the physical cause has not yet
been confirmed by a successful capture with this revision.

The C3 firmware now waits for a host command instead of starting at boot.
It uses the ESP-IDF USB Serial/JTAG driver, buffers complete output lines,
and answers `H` with `READY,SENSOR_USB_V2,esp32c3`. A single
`R,<32-lowercase-hex-session-id>` starts the benchmark. The firmware prints
that ID and rejects an immediate repeat of the same request. The capture
script sets DTR/RTS low before opening the port, retains partial byte reads,
validates the echoed session and saves exact raw bytes with acquisition
timestamps and hashes in a separate sidecar. It does not reconnect or retry
a measurement automatically. A failed transfer remains a failed capture.

Flash this firmware once, reconnect the USB cable before opening capture,
then use `capture.py --start-command`. Do not press RESET or BOOT while
capturing. Two command launches are distinct sessions, not cold-boot trials.
The complete Windows procedure is in [README](../../mcu/README.md).

## Local validation

- 31 tests and eight model subtests pass; see `host_tests.txt`.
- Transport tests cover fragmented input, incomplete data, USB failure,
  pre-open DTR/RTS settings, distinct session IDs and the actual C++ command
  parser, including malformed and oversized requests. Mock device output is
  temporary test data and is not a physical MCU result.
- Both retained legacy Mega log/JSON pairs still parse to their exact saved
  JSON objects. Numerical kernels, models, fixture headers, and the text of
  the accuracy, timing and update-cost functions are unchanged from parent
  commit `52156b1bbc5aecd1a448df269a802bc53cc38e09`.
- Four cross-compilations succeed without compiler warnings. Source hashes,
  ELF hashes, framework/tool versions and complete build logs are retained.
  This check covers the baseline and full harness on each board; it does not
  repeat the earlier 20-build per-model audit.

| Build | Flash, bytes | Static RAM, bytes |
| --- | ---: | ---: |
| Mega2560 baseline | 730 | 17 |
| Mega2560 full harness | 28,990 | 1,280 |
| ESP32-C3 baseline | 218,848 | 13,484 |
| ESP32-C3 full harness, USB v2 | 271,614 | 13,972 |

These are compiler size reports, not peak runtime memory. In particular, the
USB driver allocates RX/TX buffers and other state dynamically; static RAM
does not include those allocations. Changing the C3 USB configuration also
changes its baseline, so new absolute sizes should not be mixed with old
per-model deltas. The original size audit is preserved unchanged.

No physical C3 timing, sensor measurement, or successful Windows USB capture
was produced in this environment. The next evidence required is a complete
user-supplied C3 `.log`, `.json` and `.session.json`, followed by a second
command-launched session. Numerical input/target data remain synthetic.

Protocol references: [Espressif USB Serial/JTAG guide (IDF 4.4)](https://docs.espressif.com/projects/esp-idf/en/v4.4/esp32c3/api-guides/usb-serial-jtag-console.html)
and [pySerial open/line-state API](https://pyserial.readthedocs.io/en/latest/pyserial_api.html).
