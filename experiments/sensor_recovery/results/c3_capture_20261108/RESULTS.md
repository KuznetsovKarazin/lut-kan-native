# User-supplied MCU run review

Received 1 log/JSON pairs with 1 unique parsed contents.
Each raw log parses completely and reproduces its supplied JSON exactly. Fixture metadata and model payloads match the checked-in manifest.

Verified session IDs with acquisition sidecars: 1. Recorded session identity distinguishes acquisitions, but does not establish statistical independence, different boards or cold boots. Legacy files without sidecars have no verified acquisition identity. Equal contents alone do not prove a copied acquisition. The ten timing batches within a capture measure within-run variation; they are not ten independent boards or reset sessions.

## esp32c3_capture01

All eight prediction checks PASS at the predeclared 0.002 °C maximum host-gap tolerance. The largest observed gap is 0.0013905 °C.

Times are medians of ten batches of 2048 calls. Gross times include fixture reads, loop/call/checksum overhead, normalization and output conversion; they exclude ADC acquisition and Serial printing. The separate baseline median is 0.718262 µs/call. Corrected values are estimates, not exact isolated-kernel times.

| Method | Payload B | Gross µs/call | Corrected µs/call | MAE °C | Max error °C |
|---|---:|---:|---:|---:|---:|
| physics | 32 | 31.088135 | 30.369873 | 0.2665037 | 0.6113586 |
| polynomial | 56 | 25.768555 | 25.050293 | 0.0476196 | 0.4141846 |
| cubic16 | 80 | 74.259521 | 73.541260 | 0.0465564 | 0.3909073 |
| kan16 | 84 | 92.520508 | 91.802246 | 0.0467585 | 0.3905106 |
| uniform32_float32 | 144 | 10.951660 | 10.233398 | 0.0524224 | 0.4023132 |
| uniform32_int16 | 88 | 13.772949 | 13.054688 | 0.0524278 | 0.4015274 |
| uniform32_uint8 | 56 | 14.118652 | 13.400391 | 0.1094830 | 0.5048828 |
| nonuniform16_int16 | 120 | 15.904297 | 15.186035 | 0.0623459 | 0.4357300 |

Gross speed ratios (comparator time / LUT time):

- uniform32_float32: polynomial 2.353×; cubic16 6.781×; kan16 8.448×.
- uniform32_int16: polynomial 1.871×; cubic16 5.392×; kan16 6.718×.

The projected-update cost median is 18.242188 µs/step; all ten update batches retain monotonicity. This primitive uses synthetic targets and does not demonstrate calibration convergence or drift adaptation.

### Comparison with mega2560

Same numerical fixture and payloads; different boards, clock rates, toolchains and transport. Ratios describe these captures, not a frequency-normalized architectural result.

| Method | Reference µs/call | Current µs/call | Reference/current |
|---|---:|---:|---:|
| physics | 314.392578 | 31.088135 | 10.113× |
| polynomial | 325.237305 | 25.768555 | 12.621× |
| cubic16 | 1134.686523 | 74.259521 | 15.280× |
| kan16 | 1362.857422 | 92.520508 | 14.730× |
| uniform32_float32 | 155.024414 | 10.951660 | 14.155× |
| uniform32_int16 | 182.165039 | 13.772949 | 13.226× |
| uniform32_uint8 | 181.990234 | 14.118652 | 12.890× |
| nonuniform16_int16 | 214.547852 | 15.904297 | 13.490× |

## Interpretation and remaining work

- Compare the reported speed ratios together with accuracy and payload: float32 LUT32 stores 144 bytes versus 56 for the selected degree-nine Chebyshev model. A speed gain does not establish an accuracy or memory gain.
- Int16 LUT32 reduces the payload to 88 bytes, but inference reconstructs floating-point values. It is not fixed-point integer inference; quantization need not reduce latency.
- Uint8 and nonuniform results must be judged using both their timing and their error. This fixed fixture does not determine the best method over all sensors and budgets.
- Cubic/KAN comparisons are against the checked-in general de Boor evaluator. An optimized piecewise-cubic/Horner deployment baseline is still needed before a general speed-superiority claim. The KAN model is one fixed-grid scalar edge plus SiLU.
- Physical execution time is distinct from physical sensor accuracy: all ADC/temperature test pairs are synthetic. These measurements do not establish a new NTC accuracy specification.
- The UART protocol does not embed an ELF/source hash. Matching fixture metadata and numerical checks do not cryptographically identify the flashed firmware. Retain source_hashes.json and packages.txt alongside the device runs.
- Next: complete at least two separately identified ESP32-C3 command sessions with the same fixture; do not repeat Mega runs solely to obtain different numbers. Optimized deployment baselines and physical NTC measurements remain separate stages.
