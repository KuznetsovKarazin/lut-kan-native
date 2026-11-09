# MCU calibration benchmarks

Use [RUN_REFERENCES.txt](RUN_REFERENCES.txt) for the current seven-model V2
comparison on Mega2560, ESP32-C3 and ESP32-S3. Two sessions per board are already
complete and audited. [MATCHED_RUN.md](MATCHED_RUN.md) describes the earlier
144-byte inference/adaptation fixture; [OPTIMIZED_RUN.md](OPTIMIZED_RUN.md)
describes the separate ten-model pilot. Fixture identities must not be pooled.

The benchmark reads stored synthetic ADC/reference pairs. It measures execution
and numerical agreement, with ADC acquisition and serial printing outside timed
loops. Gross timings include input reads, normalization and checksum accumulation.
The overhead control is reported separately. No physical sensor accuracy, energy,
WCET or measured peak stack is established.

Dependencies are pinned in `requirements.txt` and `platformio.ini`. Install with
`py -m pip install -r experiments/sensor_recovery/mcu/requirements.txt` from the
repository root. Close Serial Monitor before capture and select the actual COM
port using `py -m serial.tools.list_ports`. Do not modify the frozen source or
transport during a recorded run. The original 24-file V2 source set remains
byte-identical; documentation changes do not change the firmware.

ESP32 native USB Serial/JTAG and USB-UART are distinct transports. The supplied
C3/S3 configuration uses native USB. Before measurement, the command protocol
checks READY, ECHO and ACCEPTED. Do not reset C3/S3 during a command session.
Failed/partial captures are retained; successful completion requires the final
END marker and all records. Preserve the entire runner-produced ZIP.

`build_audit.py` and `build_matched.py` produce separate deployment-build sizes.
Those sizes exclude the large capture fixture and are not stack high-water
measurements. The historical `FREE_MEMORY_ESTIMATE` has different meanings on AVR
(heap/stack gap estimate) and ESP32 (free heap); do not compare it as peak SRAM.
The historical `UPDATE_COST_ONLY` primitive is not a calibration-convergence test.
