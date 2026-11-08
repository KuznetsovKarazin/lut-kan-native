# Optimized cubic MCU compilation audit

All sixteen sequential builds succeeded without compiler warnings. These
are cross-compilation and size results; the optimized suite has not yet
been timed on a physical board. Each board has seven footprint-only builds
and one complete ten-model benchmark build. The learned functions and
input fixture are unchanged.

## Individual deployment sizes

Flash and RAM are PlatformIO's linked-program Flash and static RAM counts.
Deltas subtract the overhead-only build for the same board; they are not
peak stack, total runtime RAM or training workspace measurements.

| Board | Method | Flash B | Δ Flash B | Static RAM B | Δ RAM B |
| --- | --- | ---: | ---: | ---: | ---: |
| Mega2560 | Overhead only | 730 | 0 | 17 | 0 |
| Mega2560 | Polynomial degree 9 | 1878 | 1148 | 57 | 40 |
| Mega2560 | General cubic16 | 2684 | 1954 | 81 | 64 |
| Mega2560 | Unrolled cubic16 | 2780 | 2050 | 81 | 64 |
| Mega2560 | Horner cubic16 | 2388 | 1658 | 225 | 208 |
| Mega2560 | LUT32/float32 | 2354 | 1624 | 145 | 128 |
| Mega2560 | LUT32/int16 | 2330 | 1600 | 81 | 64 |
| ESP32-C3 | Overhead only | 218848 | 0 | 13484 | 0 |
| ESP32-C3 | Polynomial degree 9 | 219114 | 266 | 13484 | 0 |
| ESP32-C3 | General cubic16 | 219514 | 666 | 13484 | 0 |
| ESP32-C3 | Unrolled cubic16 | 219560 | 712 | 13484 | 0 |
| ESP32-C3 | Horner cubic16 | 219360 | 512 | 13484 | 0 |
| ESP32-C3 | LUT32/float32 | 219286 | 438 | 13484 | 0 |
| ESP32-C3 | LUT32/int16 | 219290 | 442 | 13484 | 0 |

Horner stores more coefficient data but needs less evaluator code than the
general cubic. Its complete individual image is only 34 B larger than
float32 LUT on Mega and 74 B larger on C3. Unrolling preserves the 16 spline
coefficients but increases Flash relative to the general evaluator.
Read-only coefficients on C3 contribute to Flash, so zero static-RAM delta
does not mean zero model storage. On Mega, compiler-folded transforms can
appear in instructions rather than the coefficient array. Model payloads
(80/224/144/88 B for unrolled/Horner/LUT-float32/LUT-int16) therefore differ
from the static-RAM deltas.

## Complete ten-model harness

| Environment | Flash B | Static RAM B |
| --- | ---: | ---: |
| mega2560_opt | 30434 | 1564 |
| esp32c3_opt | 265944 | 13964 |

The complete harness also contains the common fixture, expected outputs,
serial transport, accuracy checks and projected-update cost benchmark.
It is not an individual deployment size. The ordinary eight-model
environments remain available. Host-only optimized KAN candidates are
excluded from these ten-model images because they failed the existing
0.002 °C parity gate.

## Reproduction and provenance

```bash
python experiments/sensor_recovery/mcu/build_audit.py --out runs/optimized_build_new --boards mega2560_opt esp32c3_opt --models 0 2 3 5 6 9 10
```

PlatformIO 6.2.0; atmelavr 5.1.0 / Arduino AVR 5.2.0 / AVR GCC 7.3.0;
espressif32 6.12.0 / Arduino ESP32 3.20017.241212+sha.dcc1105b /
RISC-V GCC 8.4.0+2021r2-patch5. Platform versions are pinned in
`platformio.ini`.

[builds.json](builds.json) contains all sizes and ELF SHA-256 hashes.
[source_hashes.json](source_hashes.json) records the build inputs.
[packages.txt](packages.txt) and all sixteen raw `.log` files are retained.
Source hashes were checked against the final tested source tree. Physical
captures should use the [one-upload, two-run procedure](../../mcu/OPTIMIZED_RUN_RU.md)
to retain the Windows build/upload logs, local image hashes and session IDs.
