# Sensor recovery pilot — numerical results

All results below are synthetic simulations. No physical sensor or MCU was measured.

Run: 144 calibration cases, 7488 model evaluations, 8 virtual units, 3 repeats per unit, two reference budgets (30/75).
The original protocol precedes these results; L=32/B=75 below is a descriptive slice of the full sweep, not a test-selected configuration.

## Comparable deployment formats

All three tabulated methods below use L=32, int16: 64 table bytes + 8 quantizer bytes + 16 transform bytes = 88 bytes.

| Scenario | Direct LUT MAE | KAN-edge → LUT MAE | Physics → LUT MAE | Direct vs KAN → LUT reduction |
|---|---:|---:|---:|---:|
| NTC: matched physics | 0.057175 | 0.103916 | 0.087704 | 45.0% |
| NTC: injected frontend distortion | 0.056805 | 0.104076 | 0.329493 | 45.4% |
| LDR: matched power law | 0.003417 | 0.004706 | 0.003788 | 27.4% |

NTC units are °C; LDR units are log10(lux). KAN means one fixed-grid cubic spline edge plus SiLU, fitted by augmented least squares. It does not mean a multilayer/adaptive-grid KAN.

## Stronger models before LUT compilation

| Scenario | Physics MAE | Polynomial MAE | Cubic spline MAE | Original KAN edge MAE | Physics + LUT residual MAE |
|---|---:|---:|---:|---:|---:|
| NTC: matched physics | 0.040958 | 0.045920 | 0.047114 | 0.047498 | 0.041851 |
| NTC: injected frontend distortion | 0.304389 | 0.046689 | 0.046271 | 0.046629 | 0.045798 |
| LDR: matched power law | 0.001291 | 0.002361 | 0.002594 | 0.002631 | 0.001572 |

Their payloads are 28–32 bytes (physics), 48–56 bytes (selected polynomial), 144 bytes (cubic), 148 bytes (KAN edge), and 156–160 bytes (physics + residual). The last model still evaluates logarithms at inference. Host arithmetic uses float32-rounded stored coefficients; it is not a full MCU arithmetic simulation.

## Interpretation

- Direct training improves over naive sampling of a fitted scalar KAN edge into the same uniform deployment table. The paired unit-bootstrap intervals for these L32/int16 differences exclude zero in all three scenarios at B75; these are exploratory, unadjusted comparisons among many settings.
- It does not establish better accuracy than the uncompiled KAN/cubic model. For matched NTC and LDR physics, the physical inverse is both more accurate and smaller.
- The intentionally distorted NTC exposes model mismatch: physics fails, but a selected polynomial/cubic spline also corrects it. A claim that LUT alone handles such mismatch would be unsupported.
- int16 preserves float32-table accuracy closely; uint8 roughly doubles NTC error at L32. Precision must be assessed after encoding, not inferred from a float32 experiment.
- Some sparse/noisy fits have large isolated errors. Do not select a model by mean error alone; monotonicity, conditioning and maximum-error safeguards are the next step.
- Uniform sampling of a surrogate is only one compilation method. Optimized/nonuniform knots and deployment-aware distillation may remove some of the observed gap and must be included before claiming broad novelty.
- The exact linear solve is an offline reference optimum, not a proposed low-RAM MCU trainer. A portable C inference kernel is supplied; on-device updates, timings and memory still require measurement.

## Quantization (L32, B75)

| Scenario | float32 MAE / 144 B | int16 MAE / 88 B | uint8 MAE / 56 B |
|---|---:|---:|---:|
| NTC: matched physics | 0.057187 | 0.057175 | 0.106246 |
| NTC: injected frontend distortion | 0.056806 | 0.056805 | 0.106775 |
| LDR: matched power law | 0.003417 | 0.003417 | 0.004507 |

## Optimizer and tail diagnostics

For identical fixed-grid objectives, 1000-step Adam / exact-solve loss ratio: median 1.000000, maximum 1.120590. Thus there is no evidence of a universal optimization breakthrough in the scalar problem.

Worst direct-LUT float32 cases in each scenario (all predeclared budgets/sizes):

| Scenario | Budget | L | Unit / repeat | Selected lambda | Max absolute error | MAE |
|---|---:|---:|---|---:|---:|---:|
| ntc | 30 | 32 | 5 / 0 | 1e-06 | 4.293322 | 0.236792 |
| ntc_frontend | 75 | 8 | 6 / 2 | 0.01 | 3.454703 | 0.479181 |
| ldr | 30 | 64 | 2 / 2 | 1e-06 | 0.752845 | 0.037486 |

## Next experiments, in priority order

1. Freeze a follow-up protocol: monotone LUTs, boundary/occupancy guards, adaptive/nonuniform piecewise-linear baseline, maximum error as well as MAE. Use fresh virtual units for confirmation.
2. Use Mega 2560 and ESP32-C3 to measure direct LUT, physical inverse, Chebyshev, cubic and KAN-edge inference; report flash, SRAM, latency and end-to-end physical error. Build a separate memory-bounded online trainer and compare against recursive spline/RLS and SGD baselines.
3. Acquire/borrow NTC probes and a sufficiently accurate reference setup. Start with multiple units of one model, independent measurement sessions and temperature cycles; obtain raw ADC/reference records and an uncertainty budget before adding another sensor family.
4. For IEEE Sensors Journal, frame the contribution around measured sensor-chain recalibration under memory/energy constraints, if the data support it. Without physical evidence or an MCU advantage over polynomial/physics, this pilot alone is insufficient for resubmission elsewhere.

## Reproduction and provenance

See [README](../../README.md) for commands. `run_info.json` records seed/configuration, source SHA256 hashes, versions and the base commit. The working tree was deliberately marked dirty because code/results were committed after execution. All selected hyperparameters are in `raw_results.jsonl.gz`; all generated references and holdouts are in `cases.jsonl.gz` (standard gzip-compressed JSON Lines). The synthesis code is the authoritative provenance, not a manufacturer's measurement dataset.
