# User-supplied MCU run review

Received 2 log/JSON pairs with 1 unique parsed contents.
Each raw log parses completely and reproduces its supplied JSON exactly. Fixture metadata and model payloads match the checked-in manifest.

Identical copies are counted once. A different filename does not establish an independent run. The ten timing batches within a capture measure within-run variation; they are not ten independent boards or reset sessions.

## mega2560_capture01

All eight prediction checks PASS at the predeclared 0.002 °C maximum host-gap tolerance. The largest observed gap is 0.0013905 °C.

Times are medians of ten batches of 2048 calls. Gross times include fixture reads, loop/call/checksum overhead, normalization and output conversion; they exclude ADC acquisition and Serial printing. The separate baseline median is 12.337891 µs/call. Corrected values are estimates, not exact isolated-kernel times.

| Method | Payload B | Gross µs/call | Corrected µs/call | MAE °C | Max error °C |
|---|---:|---:|---:|---:|---:|
| physics | 32 | 314.392578 | 302.054688 | 0.2665035 | 0.6113586 |
| polynomial | 56 | 325.237305 | 312.899414 | 0.0476196 | 0.4141846 |
| cubic16 | 80 | 1134.686523 | 1122.348633 | 0.0465564 | 0.3909073 |
| kan16 | 84 | 1362.857422 | 1350.519531 | 0.0467596 | 0.3905106 |
| uniform32_float32 | 144 | 155.024414 | 142.686523 | 0.0524224 | 0.4023132 |
| uniform32_int16 | 88 | 182.165039 | 169.827148 | 0.0524278 | 0.4015274 |
| uniform32_uint8 | 56 | 181.990234 | 169.652344 | 0.1094830 | 0.5048828 |
| nonuniform16_int16 | 120 | 214.547852 | 202.209961 | 0.0623459 | 0.4357300 |

Gross speed ratios (comparator time / LUT time):

- uniform32_float32: polynomial 2.098×; cubic16 7.319×; kan16 8.791×.
- uniform32_int16: polynomial 1.785×; cubic16 6.229×; kan16 7.481×.

The projected-update cost median is 248.723633 µs/step; all ten update batches retain monotonicity. This primitive uses synthetic targets and does not demonstrate calibration convergence or drift adaptation.

## Interpretation and remaining work

- Compare the reported speed ratios together with accuracy and payload: float32 LUT32 stores 144 bytes versus 56 for the selected degree-nine Chebyshev model. A speed gain does not establish an accuracy or memory gain.
- Int16 LUT32 reduces the payload to 88 bytes, but inference reconstructs floating-point values. It is not fixed-point integer inference; quantization need not reduce latency.
- Uint8 and nonuniform results must be judged using both their timing and their error. This fixed fixture does not determine the best method over all sensors and budgets.
- Cubic/KAN comparisons are against the checked-in general de Boor evaluator. An optimized piecewise-cubic/Horner deployment baseline is still needed before a general speed-superiority claim. The KAN model is one fixed-grid scalar edge plus SiLU.
- Physical execution time is distinct from physical sensor accuracy: all ADC/temperature test pairs are synthetic. These measurements do not establish a new NTC accuracy specification.
- The UART protocol does not embed an ELF/source hash. Matching fixture metadata and numerical checks do not cryptographically identify the flashed firmware. Retain source_hashes.json and packages.txt alongside the device runs.
- Next: a second reset session on Mega2560, two ESP32-C3 captures with the same fixture, then a separately versioned optimized-baseline experiment and physical NTC measurements.
