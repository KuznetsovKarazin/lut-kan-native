# Optimized cubic deployment pilot

Host measurements only: no MCU timings are estimated from these numbers. Frozen float32 model coefficients and the same 401-point fixture are used; no training, model selection or test-point selection was performed.

Two admitted cubic implementations: six unrolled de Boor interpolations (80-byte payload) and 13 precompiled local cubic polynomials evaluated by Horner (224 bytes). Float32 LUT32 uses 144 bytes. KAN variants are retained as host-only numerical diagnostics and are excluded from the ten-model firmware suite.

## Host -O2

15 rotated-order batches × 262144 calls. These include fixture reads, function dispatch and checksum accumulation; they are not isolated arithmetic costs.

| Method | Payload B | Host ns/call | MAE C | Existing 1005-point max gap C | Gate |
|---|---:|---:|---:|---:|---|
| physics | 32 | 9.446 | 0.2665035 | 0.0000583 | PASS |
| polynomial | 56 | 11.973 | 0.0476196 | 0.0000623 | PASS |
| cubic16 | 80 | 25.603 | 0.0465564 | 0.0000132 | PASS |
| kan16 | 84 | 27.311 | 0.0467594 | 0.0018484 | PASS |
| uniform32_float32 | 144 | 3.431 | 0.0524225 | 0.0000071 | PASS |
| uniform32_int16 | 88 | 4.736 | 0.0524278 | 0.0000073 | PASS |
| uniform32_uint8 | 56 | 4.729 | 0.1094830 | 0.0000080 | PASS |
| nonuniform16_int16 | 120 | 9.939 | 0.0623459 | 0.0000077 | PASS |
| cubic16_unrolled | 80 | 7.182 | 0.0465563 | 0.0000115 | PASS |
| cubic16_horner | 224 | 3.797 | 0.0465563 | 0.0000115 | PASS |
| kan16_unrolled | 84 | 10.131 | 0.0467490 | 0.0030189 | FAIL |
| kan16_horner | 228 | 7.157 | 0.0467366 | 0.0029147 | FAIL |

## Host -Os

15 rotated-order batches × 262144 calls. These include fixture reads, function dispatch and checksum accumulation; they are not isolated arithmetic costs.

| Method | Payload B | Host ns/call | MAE C | Existing 1005-point max gap C | Gate |
|---|---:|---:|---:|---:|---|
| physics | 32 | 10.687 | 0.2665035 | 0.0000583 | PASS |
| polynomial | 56 | 11.309 | 0.0476196 | 0.0000623 | PASS |
| cubic16 | 80 | 26.328 | 0.0465564 | 0.0000132 | PASS |
| kan16 | 84 | 29.239 | 0.0467594 | 0.0018484 | PASS |
| uniform32_float32 | 144 | 6.124 | 0.0524225 | 0.0000071 | PASS |
| uniform32_int16 | 88 | 6.424 | 0.0524278 | 0.0000073 | PASS |
| uniform32_uint8 | 56 | 6.486 | 0.1094830 | 0.0000080 | PASS |
| nonuniform16_int16 | 120 | 13.059 | 0.0623459 | 0.0000077 | PASS |
| cubic16_unrolled | 80 | 9.239 | 0.0465563 | 0.0000115 | PASS |
| cubic16_horner | 224 | 5.981 | 0.0465563 | 0.0000115 | PASS |
| kan16_unrolled | 84 | 12.796 | 0.0467490 | 0.0030189 | FAIL |
| kan16_horner | 228 | 10.209 | 0.0467366 | 0.0029147 | FAIL |

## Interpretation

- Use actual MCU captures to establish speed on Mega or C3; the host has a different instruction set, floating-point implementation and compiler environment.
- The Horner representation stores 52 float coefficients instead of 16. Its memory increase must accompany any speed claim; the unrolled variant retains the original payload.
- KAN's fitted spline and SiLU terms have large opposing magnitudes. The optimized float32 rearrangements fail the existing 0.002 C host parity gate and are not admitted to hardware timing. The old implementation passes the existing grid but also has larger errors on the expanded frozen-model diagnostic grid; a passing 401-point device check is not an all-input numerical guarantee.
- All candidates, including failed KAN candidates, are retained. No tolerance was relaxed and no coefficient refit was used to admit a method.
- Numerical accuracy is measured on a synthetic sensor fixture, not a physical NTC. Compiler size reports and MCU measurements are separate evidence.
