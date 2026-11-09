# Matched MCU preparation: ready for acquisition

No new physical-board timing or physical-sensor accuracy was measured. This checkpoint validates target arithmetic, output/capture integrity and cross-compiled storage. The frozen primary case is synthetic distorted NTC, seed 2026110801, unit 0, repeat 0, B=75.

## Numerical validation

All four static payloads and all twelve online reference trajectories match the saved confirmation campaign. The host builds use the actual target float32 kernels with -O2 and -Os, contraction disabled and no fast-math. The largest static prediction gap is 0.000015259 °C; largest checkpoint coefficient gap scaled by y_half is 0.000014901 °C; largest final-metric gap is 0.000011064 °C. All are below the predeclared 0.002 °C gate.

91 focused tests and eight model subtests pass. Independent checks cover cubic basis boundaries and adjacent floats, arbitrary coefficients, PAVA against SciPy, complete float32 trajectories, deterministic export, malformed/incomplete captures and session identity. The actual firmware body is executed once on the host with mocked UART/clock to verify all emitted records; its artificial times are test-only and are not saved as timing evidence.

## Individual build sizes

| Implementation | Mega Flash, B | Mega static RAM, B | C3 Flash, B | C3 static RAM, B |
|---|---:|---:|---:|---:|
| Empty harness | 734 | 21 | 218848 | 13484 |
| LUT32 float32 | 2340 | 165 | 219276 | 13484 |
| LUT60 int16 | 2404 | 165 | 219382 | 13484 |
| Cubic32 unrolled | 2842 | 165 | 219718 | 13484 |
| Horner8 | 2324 | 165 | 219296 | 13484 |
| LUT NLMS | 2834 | 169 | 219586 | 13636 |
| LUT projected NLMS | 3376 | 329 | 219964 | 13796 |
| Cubic NLMS | 3424 | 169 | 220074 | 13636 |
| Cubic projected NLMS | 3962 | 329 | 220460 | 13796 |
| Full capture harness | 49552 | 1949 | 282748 | 14252 |

The empty harness has a volatile input/output path but no calibration. Absolute counts include framework/startup support and compiler alignment; the large C3 baseline is not calibration state. All four static model payloads are 144 bytes, independently of their compiled Flash/RAM. These build results do not establish that LUT has a smaller firmware image: on Mega the Horner image is 16 bytes smaller than LUT32; on C3 it is 20 bytes larger.

Each update keeps 148 bytes of model state. Complete PAVA additionally uses 160 bytes of caller-owned reusable workspace. Local basis and function-call stack are extra; peak stack remains unmeasured. AVR initialization snapshots reside in Flash and are copied once before an update run, avoiding a second persistent SRAM copy. Reset/initial projection is excluded from update timing.

All 20 builds completed without compiler warnings. ELF symbols verify every AVR fixture and initialization array lies below 64 KiB; the highest end address is 36668. Raw logs, versions, source/image hashes and per-build sizes are in builds/. The full benchmark is larger because it contains reference streams, states, validation grids, serial transport and all models.

## Acquisition contract

For each board, upload once and acquire two sessions. Both boards echo and acknowledge a 32-hex-digit session ID, which must match the raw header and host sidecar. Four inference implementations have ten batches of 2048 calls. Each online method has ten complete 384-update trajectories in each of three drift schedules. State checks occur at 0,127,128,160,192,256,320,384; final error uses 401 points.

Gross timing includes input reads, function dispatch and checksum work. Static and two-input update overhead loops are recorded separately. Initialization, numerical checks, serial output and ADC acquisition are outside update timing. The same indirect call pattern is used for all inference methods; their implementations are separately specialized.

Parsers reject wrong suite/fixture/precision/board, missing/duplicate/extra records, nonfinite values, failed parity, bad monotonicity, repeated IDs and raw-log/JSON disagreement. ZIP packages retain actual firmware images, source hashes, upload logs, toolchain metadata and both captures, including partial evidence on failure. This is a provenance record, not runtime cryptographic attestation.

Run instructions: [MATCHED_RUN.md](../../mcu/MATCHED_RUN.md). Real timing, independent-board variation, physical NTC accuracy, energy and peak stack remain unmeasured.
