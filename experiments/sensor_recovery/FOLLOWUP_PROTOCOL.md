# Monotonicity and MCU follow-up — 2026-11-07

Frozen before this follow-up run. Parent: 015a511f11cb70a31ea466719e3d82c42f69b73b.
The original pilot remains unchanged. This is a synthetic follow-up, not new
physical evidence and not an independent reproduction of a multilayer KAN.

## Stability

- Confirmation seed: 20261108, eight new virtual units per scenario, three
  repeats, budgets 30/75; NTC, deliberately distorted NTC, LDR as before.
- L in {8,16,32,64}. Compare unconstrained uniform LUT, monotone decreasing
  uniform LUT, and monotone decreasing nonuniform LUT. The monotonic direction
  is known from the sensor/divider configuration, not learned from test data.
- Nonuniform knots: quantiles of training inputs, endpoints -1/+1. Keep that
  grid fixed when refitting on pooled references. Count all float32 knot bytes.
- Fit by augmented least squares; monotonicity uses nonnegative adjacent
  differences and SciPy bounded least squares. No hard output clipping.
- Same lambda candidates and validation-MSE selection as the first pilot.
  For nonuniform knots, penalize locally scaled changes of adjacent slopes;
  this reduces exactly to the original second-difference penalty on a uniform
  grid and is zero for any affine function.
- Report float32 and int16 predictions, MAE, maximum absolute error,
  monotonicity violations, payload, and paired unit-bootstrap intervals.
- Retain polynomial, physics, cubic-spline and scalar-KAN comparators.
- Two known failures from the first pilot are explicit regression examples:
  NTC/30/L32/unit5/repeat0 and LDR/30/L64/unit2/repeat2. These are not new
  confirmation data and must not be pooled into confirmation summaries.
- No method or length is selected by test performance. Detailed tables retain
  all settings. A reduction in worst-case error is not a universal guarantee.

## MCU preparation and measurement

- Platforms: user's Mega 2560 and ESP32-C3 SuperMini. Use pinned PlatformIO
  atmelavr 5.1.0 and espressif32 6.12.0; record resolved compiler/frameworks.
- Export one predeclared NTC-front-end case: confirmation seed, unit0/repeat0,
  B75. Models: physics, selected Chebyshev polynomial, cubic/KAN edge, uniform
  LUT float32/int16/uint8, and nonuniform int16 LUT. This is a firmware fixture,
  not another model-selection dataset.
- Firmware evaluates exactly the exported data/models. Report prediction
  deviations from host references, physical errors and repeated timings.
  Serial printing and ADC acquisition must be outside inference timing.
- Measure gross timings and empty-loop/function baseline separately, with
  checksums and runtime input variation to prevent dead-code elimination.
- Cross-compilation determines static binary sections only. SRAM/flash
  reports must distinguish model payload, whole harness and incremental code.
  It does not establish real-board latency, energy, stack high-water use,
  acquisition noise, or training convergence.
- A monotonic two-node SGD update primitive may be timed separately. It is
  not equated to the regularized offline optimizer or a validated trainer.
- Actual device runs require the user to connect/flash their boards. No
  simulated/host time is to be labelled measured MCU time.

## First physical setup

Acquire 10 Vishay NTCLE100E3103JB0 (10 kohm, B25/85=3977 K), a TMP117 breakout
and connection cable. Use one precision divider resistor for sequential
measurements; multiple parallel channels need one resistor per channel.
Begin in a dry, thermally stable 20–50 C setup. The NTC coating has no specified
insulation and is not intended for potting; neither bare NTCs nor the breakout
are waterproof probes. Keep the MCU and divider resistor outside the heated
region, and assess sensor self-heating and temperature gradients.

TMP117 is a working comparison sensor, not a calibration certificate for the
assembled rig. Its maximum specified error is 0.1 C over -20..50 C, increasing
outside that range. Report agreement with it separately from absolute
temperature accuracy. Publication claims near 0.05 C require a better
characterized reference/uncertainty budget. Do not reuse B=3950 for these parts.
