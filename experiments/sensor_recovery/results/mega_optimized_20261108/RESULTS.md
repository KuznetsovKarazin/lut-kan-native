# User-supplied MCU run review

Received 2 log/JSON pairs with 1 unique parsed contents; retained 2 capture observations.
Each raw log parses completely and reproduces its supplied JSON exactly. Fixture metadata and model payloads match the checked-in manifest.

Verified session IDs with acquisition sidecars: 2. C3 command IDs are echoed by the device; legacy auto-start IDs and acquisition times are recorded by the host only. Recorded session identity distinguishes acquisitions, but does not establish statistical independence, different boards or cold boots. Legacy files without sidecars have no verified acquisition identity. Equal contents alone do not prove a copied acquisition. The ten timing batches within a capture measure within-run variation; they are not ten independent boards or reset sessions.

## mega2560_capture01

All 10 prediction checks PASS at the predeclared 0.002 °C maximum host-gap tolerance. The largest observed gap is 0.0013905 °C.

Times are medians of ten batches of 2048 calls. Gross times include fixture reads, loop/call/checksum overhead, normalization and output conversion; they exclude ADC acquisition and Serial printing. The separate baseline median is 12.339844 µs/call. Corrected values are estimates, not exact isolated-kernel times.

| Method | Payload B | Gross µs/call | Corrected µs/call | MAE °C | Max error °C |
|---|---:|---:|---:|---:|---:|
| physics | 32 | 314.391602 | 302.051758 | 0.2665035 | 0.6113586 |
| polynomial | 56 | 325.236328 | 312.896484 | 0.0476196 | 0.4141846 |
| cubic16 | 80 | 1134.688477 | 1122.348633 | 0.0465564 | 0.3909073 |
| kan16 | 84 | 1362.858398 | 1350.518555 | 0.0467596 | 0.3905106 |
| uniform32_float32 | 144 | 155.023438 | 142.683594 | 0.0524224 | 0.4023132 |
| uniform32_int16 | 88 | 182.166016 | 169.826172 | 0.0524278 | 0.4015274 |
| uniform32_uint8 | 56 | 181.991211 | 169.651367 | 0.1094830 | 0.5048828 |
| nonuniform16_int16 | 120 | 214.547852 | 202.208008 | 0.0623459 | 0.4357300 |
| cubic16_unrolled | 80 | 332.082031 | 319.742188 | 0.0465563 | 0.3909073 |
| cubic16_horner | 224 | 165.250000 | 152.910156 | 0.0465563 | 0.3909073 |

Gross speed ratios (comparator time / LUT time):

- uniform32_float32: polynomial 2.098×; cubic16 7.319×; kan16 8.791×; cubic16_unrolled 2.142×; cubic16_horner 1.066×.
- uniform32_int16: polynomial 1.785×; cubic16 6.229×; kan16 7.481×; cubic16_unrolled 1.823×; cubic16_horner 0.907×.

The projected-update cost median is 248.721680 µs/step; all ten update batches retain monotonicity. This primitive uses synthetic targets and does not demonstrate calibration convergence or drift adaptation.

### Comparison with esp32c3

Same numerical fixture and payloads; different boards, clock rates, toolchains and transport. Ratios describe these captures, not a frequency-normalized architectural result.

| Method | Reference µs/call | Current µs/call | Reference/current |
|---|---:|---:|---:|
| physics | 31.124268 | 314.391602 | 0.099× |
| polynomial | 25.767578 | 325.236328 | 0.079× |
| cubic16 | 74.300781 | 1134.688477 | 0.065× |
| kan16 | 92.544189 | 1362.858398 | 0.068× |
| uniform32_float32 | 10.981934 | 155.023438 | 0.071× |
| uniform32_int16 | 13.753418 | 182.166016 | 0.075× |
| uniform32_uint8 | 14.130371 | 181.991211 | 0.078× |
| nonuniform16_int16 | 15.859619 | 214.547852 | 0.074× |
| cubic16_unrolled | 26.420898 | 332.082031 | 0.080× |
| cubic16_horner | 12.138184 | 165.250000 | 0.073× |

## mega2560_capture02

All 10 prediction checks PASS at the predeclared 0.002 °C maximum host-gap tolerance. The largest observed gap is 0.0013905 °C.

Times are medians of ten batches of 2048 calls. Gross times include fixture reads, loop/call/checksum overhead, normalization and output conversion; they exclude ADC acquisition and Serial printing. The separate baseline median is 12.339844 µs/call. Corrected values are estimates, not exact isolated-kernel times.

| Method | Payload B | Gross µs/call | Corrected µs/call | MAE °C | Max error °C |
|---|---:|---:|---:|---:|---:|
| physics | 32 | 314.391602 | 302.051758 | 0.2665035 | 0.6113586 |
| polynomial | 56 | 325.236328 | 312.896484 | 0.0476196 | 0.4141846 |
| cubic16 | 80 | 1134.688477 | 1122.348633 | 0.0465564 | 0.3909073 |
| kan16 | 84 | 1362.858398 | 1350.518555 | 0.0467596 | 0.3905106 |
| uniform32_float32 | 144 | 155.023438 | 142.683594 | 0.0524224 | 0.4023132 |
| uniform32_int16 | 88 | 182.166016 | 169.826172 | 0.0524278 | 0.4015274 |
| uniform32_uint8 | 56 | 181.991211 | 169.651367 | 0.1094830 | 0.5048828 |
| nonuniform16_int16 | 120 | 214.547852 | 202.208008 | 0.0623459 | 0.4357300 |
| cubic16_unrolled | 80 | 332.082031 | 319.742188 | 0.0465563 | 0.3909073 |
| cubic16_horner | 224 | 165.250000 | 152.910156 | 0.0465563 | 0.3909073 |

Gross speed ratios (comparator time / LUT time):

- uniform32_float32: polynomial 2.098×; cubic16 7.319×; kan16 8.791×; cubic16_unrolled 2.142×; cubic16_horner 1.066×.
- uniform32_int16: polynomial 1.785×; cubic16 6.229×; kan16 7.481×; cubic16_unrolled 1.823×; cubic16_horner 0.907×.

The projected-update cost median is 248.721680 µs/step; all ten update batches retain monotonicity. This primitive uses synthetic targets and does not demonstrate calibration convergence or drift adaptation.

### Comparison with esp32c3

Same numerical fixture and payloads; different boards, clock rates, toolchains and transport. Ratios describe these captures, not a frequency-normalized architectural result.

| Method | Reference µs/call | Current µs/call | Reference/current |
|---|---:|---:|---:|
| physics | 31.124268 | 314.391602 | 0.099× |
| polynomial | 25.767578 | 325.236328 | 0.079× |
| cubic16 | 74.300781 | 1134.688477 | 0.065× |
| kan16 | 92.544189 | 1362.858398 | 0.068× |
| uniform32_float32 | 10.981934 | 155.023438 | 0.071× |
| uniform32_int16 | 13.753418 | 182.166016 | 0.075× |
| uniform32_uint8 | 14.130371 | 181.991211 | 0.078× |
| nonuniform16_int16 | 15.859619 | 214.547852 | 0.074× |
| cubic16_unrolled | 26.420898 | 332.082031 | 0.080× |
| cubic16_horner | 12.138184 | 165.250000 | 0.073× |

## Between-capture repeatability: mega2560/spline_optimized_v1

Reviewed 2 captures; 2 acquisition IDs have verified sidecars. This checks repeatability of the supplied captures; it does not estimate between-board or cold-boot variation. Timing batches are kept within their sessions rather than pooled as independent acquisitions.

All four reported accuracy fields identical across captures: True. Largest relative range of session medians: 0.000000%. Relative range is 100 × (maximum − minimum) / minimum; it is not a confidence interval.

| Method | Minimum session median, µs/call | Maximum session median, µs/call | Relative range, % |
|---|---:|---:|---:|
| physics | 314.391602 | 314.391602 | 0.000000 |
| polynomial | 325.236328 | 325.236328 | 0.000000 |
| cubic16 | 1134.688477 | 1134.688477 | 0.000000 |
| kan16 | 1362.858398 | 1362.858398 | 0.000000 |
| uniform32_float32 | 155.023438 | 155.023438 | 0.000000 |
| uniform32_int16 | 182.166016 | 182.166016 | 0.000000 |
| uniform32_uint8 | 181.991211 | 181.991211 | 0.000000 |
| nonuniform16_int16 | 214.547852 | 214.547852 | 0.000000 |
| cubic16_unrolled | 332.082031 | 332.082031 | 0.000000 |
| cubic16_horner | 165.250000 | 165.250000 | 0.000000 |

## Interpretation and remaining work

- Compare the reported speed ratios together with accuracy and payload: float32 LUT32 stores 144 bytes versus 56 for the selected degree-nine Chebyshev model. A speed gain does not establish an accuracy or memory gain.
- Int16 LUT32 reduces the payload to 88 bytes, but inference reconstructs floating-point values. It is not fixed-point integer inference; quantization need not reduce latency.
- Uint8 and nonuniform results must be judged using both their timing and their error. This fixed fixture does not determine the best method over all sensors and budgets.
- This suite includes general de Boor, unrolled and precompiled Horner cubic evaluation. The original scalar KAN edge remains; optimized float32 KAN candidates failed the existing host gate and were excluded from hardware timing.
- Physical execution time is distinct from physical sensor accuracy: all ADC/temperature test pairs are synthetic. These measurements do not establish a new NTC accuracy specification.
- The UART protocol does not embed an ELF/source hash. Matching fixture metadata and numerical checks do not cryptographically identify the flashed firmware. Retain source_hashes.json and packages.txt alongside the device runs.
- Two optimized-suite Mega host acquisition records are available, including any byte-identical deterministic outputs. Compare their timing/payload/error tradeoff with C3; matched-storage evaluation and physical NTC validation remain separate stages.
