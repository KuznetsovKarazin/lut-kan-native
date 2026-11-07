# Sensor calibration recovery pilot

Protocol frozen before the first recovery run on 2026-11-07. Base commit:
`1a33e7c299937c5ad0a11b022da70805aa657420`.

This is an exploratory computational pilot for a future IEEE Sensors Journal
submission. All data in this pilot are synthetic. No measurement on a physical
sensor or microcontroller is reported. Historical manufacturer-data and MCU
claims are not carried over without raw evidence.

## Questions

1. Does directly fitting deployment LUT values beat a strong polynomial,
   cubic spline, scalar KAN edge, or a physics model compiled to the same LUT?
2. Are improvements retained after actual coefficient quantization?
3. When the physical model is wrong because the measurement chain is distorted,
   can a residual LUT correct it with fewer coefficients?
4. Which effects come from representation, regularization, or optimization?

The one-dimensional, fixed-grid LUT is exactly a degree-one B-spline. That
identity is checked numerically. A result against a cubic spline is a result
about approximation and deployment, not a discovery of a new spline family.
The scalar KAN-edge comparator includes a cubic spline and a SiLU residual
(Liu et al., arXiv:2404.19756, equations 2.10–2.12). It is not a multilayer
KAN, pykan reproduction, or test of adaptive KAN grids.

## Data and splits

- NTC: Beta model, 10 kohm nominal at 25 C, B=3950 K, per-unit B variation
  +/-3% and R25 variation +/-5%, range 0–100 C.
- NTC frontend stress: the same sensor model with a deliberately injected
  smooth ADC distortion, amplitude 4 ADC codes and three periods over full
  scale. This is a controlled model-mismatch stress test, not measured drift.
- LDR: R=R10*(10/E)^gamma, R10=10 kohm +/-30%, gamma=0.7 +/-10%,
  E=1–1000 lux. Fit and report error in log10(lux), not degrees C or lux.
- The nominal divider resistor is 10 kohm. All nominal constants are shared
  across virtual units of the same model; no cross-product factory mismatch.
- Calibration budget B in {30,75} counts every labelled reference, including
  validation. Two endpoints plus stratified interior references are collected.
  Approximately 2/3 train and 1/3 validation, with endpoints kept in train.
- All tunable methods use exactly the same split. Select degree/regularization
  by validation MSE, then refit on all B references. Test data are never passed
  to fit or selection functions. A one-point offset baseline uses exactly one
  preselected reference near the physical-range midpoint; factory uses none.
- Input range comes only from calibration readings (5% margin), never from
  the virtual unit's hidden true parameters or test set.
- ADC is 10 bit, Gaussian noise sigma=1 code BEFORE integer rounding;
  each reported reading averages 16 conversions. Reference noise is 0.02 C
  for NTC and 0.005 log10(lux) for LDR. These are experimental assumptions.
- Tests: 401 independent quantity grid locations, both an ideal analog input
  track (approximation diagnostic) and independently noisy/quantized averaged
  inputs (primary system simulation). Test target is noise-free ground truth.
- Full pilot: 8 virtual units, 3 independently generated calibration/noise
  realizations per unit. Units, not optimizer seeds or test points, are the
  primary independent replication level. This is still a small virtual study.

## Comparators and deployment

- Fixed nominal factory inverse and true one-point offset.
- Chebyshev polynomial, degree selected from {1,2,3,4,5,7,9}.
- Physics: Steinhart–Hart for NTC; log-resistance linear fit for LDR.
- Uniform LUT / regularized linear spline, L in {8,16,32,64}, solved by
  augmented least squares; lambda in {0,1e-6,1e-4,1e-2,1,100}.
- Cubic B-spline with the same coefficient counts and validation-selected
  second-difference regularization; scalar cubic KAN edge adds a SiLU term.
- Physics-to-LUT and KAN-edge-to-LUT use the SAME endpoint-inclusive grid,
  input range, interpolation kernel, and encoding as directly fitted LUT.
- Physics plus regularized LUT residual is a separate two-stage comparator.
  Its deployment memory includes BOTH models. It is not advertised as a pure
  LUT implementation, and still requires physics evaluation at inference.
- Every LUT is evaluated after float32, affine uint8 and affine int16 encoding.
  Memory counts actual table bytes, quantizer scale/offset, and four float32
  input/output transform constants. Code/stack/allocator/firmware bytes are
  excluded and must later be measured separately on MCU.
- Cubic and polynomial comparisons include float32-rounded coefficients.
  SciPy/NumPy predictions are host reference arithmetic, not MCU timing.
- Optimizer diagnostic: fixed-grid full-batch NumPy Adam vs exact solution
  for the same loss. This is not a byte-for-byte replay of legacy PyTorch.

## Analysis and decision

Record MAE, RMSE, maximum absolute error, bias, physical units, selected
hyperparameters, deployment payload, input clipping, provenance and all seeds.
Use paired unit-level bootstrap intervals after averaging repeats per unit;
show every method and precision, including losses. No universal p<0.01 or
sensor-wide winner claims are made from this exploratory pilot.

An engineering lead must survive strong comparators and quantization. A useful
follow-up should show either a material error reduction at equal payload, or
smaller payload at equal error, on real measurement chains. A fixed 10% margin
is an exploratory planning threshold, not a significance criterion. If only
the deliberately distorted scenario shows an advantage, obtain physical
evidence of such distortion before presenting it as a practical contribution.

## Next stages

1. Audit and pilot here; recover actual manufacturer R/T tables separately.
2. Independently validate the selected method; add monotonicity and alternative
   knot placement only as explicitly labelled follow-up experiments.
3. Run the portable kernel on Mega 2560 and ESP32-C3. Record actual SRAM, flash,
   cycles, training convergence and numerical error; no CPU-to-MCU scaling.
4. Measure physical NTC units and, if warranted, a second sensor family against
   a documented reference. Keep unit/batch/day splits and uncertainty budget.

Existing personal MCU boards suffice for stage 3. Stage 4 needs sensors and a
reference setup. The Extech MN35 is a useful wiring/multimeter tool; its
temperature function is not a reference for claims near 0.1 C. Its standard
kit may already include a Type K probe. Sensor purchase follows the pilot.

## Primary sources

- https://arxiv.org/abs/2404.19756
- https://doi.org/10.1016/j.measurement.2011.02.009
- https://www.mdpi.com/2227-7390/9/18/2266
- https://www.microchip.com/en-us/application-notes/an942
- https://www.extech.com/products/mn35/
