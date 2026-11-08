# Matched-storage MCU checkpoint

Frozen before new target code or measurements, 8 November 2026. This follows
the completed matched-storage/adaptation simulation; it does not change that
campaign or its selected settings. No physical sensor is needed for this
computational benchmark.

## Fixture and comparisons

Use the predetermined confirmation case `ntc_frontend`, seed 2026110801,
unit 0, repetition 0, B=75. Select no case or setting by hardware performance.
Export the already specified models from the same numerical pipeline and
verify them against the saved campaign. Static methods at exactly 144 payload
bytes: LUT32/float32, LUT60/int16, compact cubic32 and eight-piece cubic Horner.
The compact cubic uses an unrolled local evaluation, not a generic knot search.

For online updates use the same initial LUT32/cubic32 and eta=0.5 as the saved
confirmation run. Evaluate ordinary NLMS and NLMS followed by full Euclidean
projection onto decreasing coefficients (PAVA), for both families. Export all
384 labelled references for each of the none/step/ramp schedules. Reset to the
same initial state outside the timed region for every trajectory repetition.
This does not include the larger-state RLS comparators.

The timed update includes input/reference normalization, local basis, prediction,
normalized residual update, coefficient writes and, where selected, complete
PAVA. Include fixture reads and checksum accumulation; exclude initialization,
validation and Serial output. Measure one complete 384-sample trajectory per
repeat, ten repeats. Report gross time per update as primary; record a matching
two-input read/checksum loop separately, without silently subtracting it.
Static inference uses 2048 calls per repeat, ten repeats, with a recorded
input/read/checksum overhead loop.

## Numerical and memory gates

Use float32 throughout target arithmetic, no fast-math and no contraction.
This is stricter deployment arithmetic than the saved host float64-intermediate
reference. Preserve that distinction. Compare static predictions and online
coefficient states at predeclared checkpoints 0, 127, 128, 160, 192, 256, 320,
384. The gate is 0.002 °C for predictions and coefficient discrepancy multiplied
by y_half; monotonicity is required for projected states. Use independent SciPy
bases/PAVA, arbitrary coefficient vectors and boundary/adjacent-float cases in
host tests. Validate all selected trajectories in both -O2 and -Os host builds.
If the gate fails, diagnose it; do not loosen it after seeing results.

Retain final 401-point noisy-grid MAE/RMSE/max error for every schedule/method.
Input and reference values are rounded to float32 in the embedded fixture;
this may slightly change errors relative to the original host results.

Inference payload includes coefficient encoding and four transform floats.
Online persistent state is 148 bytes, including eta. Report caller-owned PAVA
workspace and local scratch separately. Record full-harness Flash/static RAM
and individual deployment builds; static RAM and compiler stack reports are
not measured peak stack. AVR fixture arrays must lie in addressable program
memory; reject a build exceeding the near-flash access range for those arrays.

## Acquisition and evidence

Create separate `_matched` environments; preserve previous benchmark entry
points. Both boards acknowledge a host-generated session ID and print it in
the result. Reject old-suite captures, duplicate/reset starts, malformed rows,
missing repetitions, failed parity and mismatched IDs. A runner uploads once,
then captures two sequential command-driven sessions. Save raw logs, parsed
data, sidecars, source hashes, toolchain/upload logs and the actual firmware
images in the evidence ZIP. Image presence verifies packaged bytes, not a
cryptographic attestation of the executing device.

Host tests and cross-compilations cannot supply real board timing. Until user
captures arrive, report this checkpoint as ready for acquisition, not measured.
