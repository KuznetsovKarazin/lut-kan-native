# Matched storage and sequential adaptation — reconstruction run

Frozen on 2026-11-08 before this run. Starting evidence commit:
`05f8f651ba0fecf2e09636d9c8ece55f4c07dda0`.
The interrupted conversation mentioned 3168 evaluations and 5184 trajectories,
but their code/raw results were not recovered. This is a **new experiment**,
not an exact reconstruction of those numbers or an attempt to match them.
Do not overwrite earlier data, firmware, coefficients or capture reports.

## Common design

Reuse `calibration.py`'s three synthetic scenarios, reference/noise model,
validation-only regularization selection, initial reference budgets 30/75,
and 401-point independent noisy test grid. Confirmation uses 8 new virtual
units and 3 calibration/noise repetitions per unit. Root seed is 2026110801;
the separate development seed is 2026110800. Neither was used in the saved
20261107/20261108 pilot. These are synthetic parametric cases, not acquired
manufacturer data or physical sensor measurements.

All validation references count toward B. Input normalization is fixed using
only initial reference inputs and the declared physical operating range;
future observations do not modify it. Report clipping. Coefficients and
normalization parameters are stored as float32; static reference predictions
use float64 accumulation of the stored values. This isolates storage from
kernel arithmetic, which requires separate target validation.

## A. Static comparisons at matched inference payload

Predeclare caps 80, 144 and 224 bytes. Count the same four float32 transforms
(16 B) for every model and 8 B for affine int16 scale/offset. Grid sizes are
compile-time constants, with uniform knots generated from size and endpoints;
code, stack, headers and training workspace are not included in this payload.

At each cap fit these seven models on the identical reference split:

| Model | Cap 80 B | Cap 144 B | Cap 224 B |
|---|---:|---:|---:|
| Uniform LUT float32, unconstrained / monotone | 16 nodes | 32 nodes | 52 nodes |
| Uniform LUT int16, unconstrained / monotone | 28 nodes | 60 nodes | 100 nodes |
| Compact cubic B-spline float32 | 16 coefficients | 32 coefficients | 52 coefficients |
| Cubic compiled to local Horner pieces float32 | 4 pieces / 7 coefficients | 8 / 11 | 13 / 16 |
| Compact scalar KAN edge float32 | 15 spline coefficients + SiLU | 31 + SiLU | 51 + SiLU |

Horner uses four stored coefficients per piece, with a local coordinate in
[0,1], plus the common transforms. The underlying fitted function is C2
before rounding; report any value/derivative boundary mismatch after rounding.
Validate Horner conversion against independent BSpline evaluation on dense
points, exact knots, adjacent floats and clipped endpoints. Select its
regularization after conversion/rounding on validation, then refit on B.
Select LUT regularization after the declared encoding; do not use float32
validation predictions to select an int16 deployment.

Add one selected Chebyshev polynomial (degree <=9) and one fitted physical
model per calibration case as smaller-payload references, not relabelled as
equal-cap models. The fixed matrix has 144 cases and 3312 model evaluations.
Retain every model, coefficient payload, lambda, reference and outcome.
Primary endpoint is noisy-test MAE; also retain ideal-input error, RMSE,
maximum error and bias. Compare within scenario, budget and payload.
Use paired unit bootstrap (average the three repetitions first), 2000 draws;
intervals are exploratory, with no multiplicity-adjusted significance claim.

## B. Sequential adaptation

Initial models: LUT32 and cubic32, each 144 B inference payload; selected
Chebyshev polynomial as an additional reference. Initial fits may be offline;
this does not establish a low-memory on-device initialization algorithm.

Each stream supplies 384 new labelled references, one at a time. Sample one
quantity per bin in 32 uniform bins of the operating range, in independently
shuffled blocks. This assumes access to controlled reference measurements
across the range; it is not label-free, autonomous self-calibration.
Each method sees the identical observation/label stream and no future labels.
Do not replay samples or tune on the confirmation streams.

Drift schedules, fixed before results:

- none: multiplier d=0;
- step: d=0 through sample 127, d=1 from sample 128;
- ramp: d=clip((t-128)/128,0,1).

For unperturbed normalized ADC a, inject
`d * (2 + 2*sin(2*pi*a)) / 1023`, then apply the original conversion noise,
rounding and 16-conversion average. This controlled offset/nonlinearity is
an assumption, not a measured aging law. Truth/hidden sensor parameters are
available only to the evaluator, never to fit/update functions.

Nine methods:

1. frozen LUT32;
2. frozen cubic32;
3. frozen selected polynomial;
4. LUT32 NLMS;
5. LUT32 NLMS followed by decreasing isotonic projection;
6. cubic32 NLMS;
7. cubic32 NLMS followed by decreasing coefficient projection;
8. cubic32 RLS with full covariance (larger-state reference);
9. selected polynomial RLS with full covariance (larger-state reference).

For NLMS use `c += eta * error * basis / (basis @ basis + 1e-6)` and round
stored state to float32 after each sample. Project with PAVA, including the
initial coefficients for the projected methods. Decreasing cubic coefficients
are a sufficient shape constraint, not a necessary characterization of all
monotone cubic functions. Projection cost/workspace must be counted separately.
The original MCU `UPDATE_COST_ONLY` timings do not measure these new algorithms.

RLS starts from the same fitted coefficients, with P=100I. Update with
`k=P b/(forget+b.T P b)`, `c+=k*error`,
`P=(P-k (b.T P))/forget`, symmetrize, store float32.
Record covariance numerical failures; do not drop unsuccessful streams.
Its O(n^2) persistent state is not comparable to a LUT's inference payload.

Choose one global eta per NLMS method from {0.05,0.2,0.5,1.0} and one global
forgetting factor per RLS method from {0.98,0.995,1.0}, using ONLY development
streams (4 virtual units, 2 repetitions, all scenarios/budgets/drifts).
Selection criterion: mean normalized heldout MSE integrated over the fixed
checkpoints (divide physical errors by the common scenario half-range).
This objective includes the no-drift condition. It is a development choice,
not optimization of a confirmation endpoint. Record all candidates.

Evaluation checkpoints: 0,32,64,96,127,128,129,132,144,160,192,224,256,320,384.
At each, evaluate an independently noisy 401-point grid, both immediately
before and after the current update where applicable. Keep prequential
single-reference prediction errors for every sample and the full model
state at checkpoints. Report final MAE, integrated MAE, peak grid error,
clipping, and sufficient coefficient-order checks. Ground truth is never
passed to the updater. Retain observations, references, predictions/metrics
and actual payload/state sizes. Scratch/stack bytes and MCU latency for these
new adaptation paths remain unmeasured.

Confirmation: 3 scenarios x 2 budgets x 3 drift modes x 8 units x 3 repeats
=432 streams, 3888 method trajectories. No reshuffling/reselection after
seeing confirmation results. Include no-drift losses and negative results.

## Checks and decision

Test independent Horner parity, exact byte budgets, encoding, projection
optimality/monotonicity, NLMS/RLS against direct algebra, causal prefix
invariance, no test-label access, independent RNGs and deterministic replay.
Run the full focused sensor suite, then replay a small case and regenerate
the complete report from saved raw results. Verify source/result hashes.

Do not extrapolate old MCU times to the new 8-piece Horner or adaptation
paths. Do not claim physical NTC accuracy, measured drift recovery, energy,
peak stack, general KAN superiority, or article acceptance. The physical NTC
stage remains deferred as requested. Assess the paper using the actual
error/storage/time tradeoff and the strength of the classical comparators.
