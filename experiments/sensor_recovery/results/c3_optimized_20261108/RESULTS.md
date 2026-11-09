# User-supplied MCU run review

Received 2 log/JSON pairs with 2 unique parsed contents.
Each raw log parses completely and reproduces its supplied JSON exactly. Fixture metadata and model payloads match the checked-in manifest.

Verified session IDs with acquisition sidecars: 2. Recorded session identity distinguishes acquisitions, but does not establish statistical independence, different boards or cold boots. Legacy files without sidecars have no verified acquisition identity. Equal contents alone do not prove a copied acquisition. The ten timing batches within a capture measure within-run variation; they are not ten independent boards or reset sessions.

## esp32c3_capture01

All 10 prediction checks PASS at the predeclared 0.002 °C maximum host-gap tolerance. The largest observed gap is 0.0013905 °C.

Times are medians of ten batches of 2048 calls. Gross times include fixture reads, loop/call/checksum overhead, normalization and output conversion; they exclude ADC acquisition and Serial printing. The separate baseline median is 0.717285 µs/call. Corrected values are estimates, not exact isolated-kernel times.

| Method | Payload B | Gross µs/call | Corrected µs/call | MAE °C | Max error °C |
|---|---:|---:|---:|---:|---:|
| physics | 32 | 31.124268 | 30.406982 | 0.2665037 | 0.6113586 |
| polynomial | 56 | 25.767578 | 25.050293 | 0.0476196 | 0.4141846 |
| cubic16 | 80 | 74.300781 | 73.583496 | 0.0465564 | 0.3909073 |
| kan16 | 84 | 92.544189 | 91.826904 | 0.0467585 | 0.3905106 |
| uniform32_float32 | 144 | 10.981934 | 10.264648 | 0.0524224 | 0.4023132 |
| uniform32_int16 | 88 | 13.753418 | 13.036133 | 0.0524278 | 0.4015274 |
| uniform32_uint8 | 56 | 14.130371 | 13.413086 | 0.1094830 | 0.5048828 |
| nonuniform16_int16 | 120 | 15.859619 | 15.142334 | 0.0623459 | 0.4357300 |
| cubic16_unrolled | 80 | 26.420898 | 25.703613 | 0.0465563 | 0.3909073 |
| cubic16_horner | 224 | 12.138184 | 11.420898 | 0.0465563 | 0.3909073 |

Gross speed ratios (comparator time / LUT time):

- uniform32_float32: polynomial 2.346×; cubic16 6.766×; kan16 8.427×; cubic16_unrolled 2.406×; cubic16_horner 1.105×.
- uniform32_int16: polynomial 1.874×; cubic16 5.402×; kan16 6.729×; cubic16_unrolled 1.921×; cubic16_horner 0.883×.

The projected-update cost median is 18.296875 µs/step; all ten update batches retain monotonicity. This primitive uses synthetic targets and does not demonstrate calibration convergence or drift adaptation.

## esp32c3_capture02

All 10 prediction checks PASS at the predeclared 0.002 °C maximum host-gap tolerance. The largest observed gap is 0.0013905 °C.

Times are medians of ten batches of 2048 calls. Gross times include fixture reads, loop/call/checksum overhead, normalization and output conversion; they exclude ADC acquisition and Serial printing. The separate baseline median is 0.717285 µs/call. Corrected values are estimates, not exact isolated-kernel times.

| Method | Payload B | Gross µs/call | Corrected µs/call | MAE °C | Max error °C |
|---|---:|---:|---:|---:|---:|
| physics | 32 | 31.124512 | 30.407227 | 0.2665037 | 0.6113586 |
| polynomial | 56 | 25.767578 | 25.050293 | 0.0476196 | 0.4141846 |
| cubic16 | 80 | 74.300781 | 73.583496 | 0.0465564 | 0.3909073 |
| kan16 | 84 | 92.543945 | 91.826660 | 0.0467585 | 0.3905106 |
| uniform32_float32 | 144 | 10.981934 | 10.264648 | 0.0524224 | 0.4023132 |
| uniform32_int16 | 88 | 13.753418 | 13.036133 | 0.0524278 | 0.4015274 |
| uniform32_uint8 | 56 | 14.130371 | 13.413086 | 0.1094830 | 0.5048828 |
| nonuniform16_int16 | 120 | 15.859619 | 15.142334 | 0.0623459 | 0.4357300 |
| cubic16_unrolled | 80 | 26.420898 | 25.703613 | 0.0465563 | 0.3909073 |
| cubic16_horner | 224 | 12.138184 | 11.420898 | 0.0465563 | 0.3909073 |

Gross speed ratios (comparator time / LUT time):

- uniform32_float32: polynomial 2.346×; cubic16 6.766×; kan16 8.427×; cubic16_unrolled 2.406×; cubic16_horner 1.105×.
- uniform32_int16: polynomial 1.874×; cubic16 5.402×; kan16 6.729×; cubic16_unrolled 1.921×; cubic16_horner 0.883×.

The projected-update cost median is 18.296875 µs/step; all ten update batches retain monotonicity. This primitive uses synthetic targets and does not demonstrate calibration convergence or drift adaptation.

## Between-capture repeatability: esp32c3/spline_optimized_v1

Reviewed 2 captures; 2 acquisition IDs have verified sidecars. This checks repeatability of the supplied captures; it does not estimate between-board or cold-boot variation. Timing batches are kept within their sessions rather than pooled as independent acquisitions.

All four reported accuracy fields identical across captures: True. Largest relative range of session medians: 0.000784%. Relative range is 100 × (maximum − minimum) / minimum; it is not a confidence interval.

| Method | Minimum session median, µs/call | Maximum session median, µs/call | Relative range, % |
|---|---:|---:|---:|
| physics | 31.124268 | 31.124512 | 0.000784 |
| polynomial | 25.767578 | 25.767578 | 0.000000 |
| cubic16 | 74.300781 | 74.300781 | 0.000000 |
| kan16 | 92.543945 | 92.544189 | 0.000264 |
| uniform32_float32 | 10.981934 | 10.981934 | 0.000000 |
| uniform32_int16 | 13.753418 | 13.753418 | 0.000000 |
| uniform32_uint8 | 14.130371 | 14.130371 | 0.000000 |
| nonuniform16_int16 | 15.859619 | 15.859619 | 0.000000 |
| cubic16_unrolled | 26.420898 | 26.420898 | 0.000000 |
| cubic16_horner | 12.138184 | 12.138184 | 0.000000 |

## Interpretation and remaining work

- Compare the reported speed ratios together with accuracy and payload: float32 LUT32 stores 144 bytes versus 56 for the selected degree-nine Chebyshev model. A speed gain does not establish an accuracy or memory gain.
- Int16 LUT32 reduces the payload to 88 bytes, but inference reconstructs floating-point values. It is not fixed-point integer inference; quantization need not reduce latency.
- Uint8 and nonuniform results must be judged using both their timing and their error. This fixed fixture does not determine the best method over all sensors and budgets.
- This suite includes general de Boor, unrolled and precompiled Horner cubic evaluation. The original scalar KAN edge remains; optimized float32 KAN candidates failed the existing host gate and were excluded from hardware timing.
- Physical execution time is distinct from physical sensor accuracy: all ADC/temperature test pairs are synthetic. These measurements do not establish a new NTC accuracy specification.
- The UART protocol does not embed an ELF/source hash. Matching fixture metadata and numerical checks do not cryptographically identify the flashed firmware. Retain source_hashes.json and packages.txt alongside the device runs.
- Two optimized-suite C3 command sessions are available. Interpret the unrolled/Horner timing together with payload and accuracy; physical NTC validation remains a separate stage.
