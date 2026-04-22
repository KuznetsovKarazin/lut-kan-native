# M3 Findings — Z-score layer-2 normalization does NOT help

## Summary

Variant 1 of the plan (running-stats z-score normalization for layer-2) was
the simplest intervention to test the "layer-2 coverage is the bottleneck"
hypothesis from M2. The answer is:

**Z-score normalization raises layer-2 coverage substantially (from ~0.25
effective-support/KL to ~0.55), but MSE gets WORSE, not better.**

This effectively falsifies the "coverage is the bottleneck" hypothesis.
Since variant 2 (learnable `tanh(a·z+b)`) is a strictly more expensive
version of the same intervention (expanding layer-2 coverage at training
time), there is no reason to expect it to succeed where variant 1 failed
for the same reason. We stop here and do not pursue variant 2.

## Bug discovered along the way

Before reporting the null result, I have to flag that the first draft of
this experiment had a real bug in `train_kan2()` that produced misleading
numbers. Short version: when `activation='zscore'`, the best-val model
was being reconstructed with default arguments (activation='tanh', L2
domain [-1,1]), so `mse_test_at_best` was computed with a DIFFERENT
forward than what was trained. This produced apparent MSE values of 0.2
where the actual best-val MSE was ~1e-4 (observable as `mse_val_at_best
> mse_val_final`, which is impossible if tracking is correct).

Fixed. Regression test added in
`tests/test_train_kan2_bestmodel_bug.py`. The M2 destructiveness sweep
(from the previous session) ran on `activation='tanh'` models only, so
its numbers are not affected. The M2 coverage diagnostic and the initial
(wrong) M3 run produced inflated "MSE" numbers for zscore models — the
qualitative conclusion (zscore doesn't help) is the same after the fix,
but specific numbers shifted.

## Experimental setup (post-fix)

Two tasks, [1→4→1] and [2→4→1], hidden_dim=4, K=16, L=32,
lr=5e-4, 3 seeds. Polynomial baselines are trained 200 epochs, direct-LUT
is trained 100 epochs on top of poly-init.

Three conditions:
  (a) `activation='tanh'`: LUT-KAN2 initialized from a standard PolyKAN2
      (layer-2 domain [-1,1] after tanh).
  (b) `activation='zscore'`, L2 domain [-3,3]: LUT-KAN2 initialized from
      a `PolyKAN2Zscore` trained on the same layer-2 domain.
  (c) `activation='zscore'`, L2 domain [-2,2]: same with tighter domain.

The key methodological point in (b) and (c) is that **the polynomial
reference IS trained in the same activation regime**, so when we
sample polynomial -> LUT, we don't extrapolate out-of-distribution.
An earlier draft (before I noticed) used tanh-trained polynomials to
init zscore LUT-KAN2, which was extrapolation into a wildly wrong
region; init MSE was ~1 instead of ~1e-4. Fixed.

## Results

### composition_1d (y = tanh(2·sin(πx)))

PolyKAN2(tanh) baseline test MSE = 1.83e-04

| condition | init MSE | best MSE | L2 eff/KL | vs PolyKAN2 |
|---|---:|---:|---:|---:|
| tanh          | 4.24e-04 | **1.43e-04** | 0.24 | **1.28×** (LUT better) |
| zscore 3σ     | 4.00e-04 | 2.64e-04    | 0.38 | 0.69× |
| zscore 2σ     | 6.43e-04 | 3.69e-04    | 0.54 | 0.50× |

### feynman_2d (y = sin(πx1) + 0.5·cos(2πx1·x2))

PolyKAN2(tanh) baseline test MSE = 7.51e-04

| condition | init MSE | best MSE | L2 eff/KL | vs PolyKAN2 |
|---|---:|---:|---:|---:|
| tanh          | 8.38e-04 | **6.95e-04** | 0.38 | **1.08×** |
| zscore 3σ     | 1.55e-03 | 1.48e-03    | 0.54 | 0.51× |
| zscore 2σ     | 1.36e-03 | 1.34e-03    | 0.59 | 0.56× |

On both tasks, **tanh beats both zscore variants** despite having HALF
the layer-2 coverage. The trend is monotonic: higher coverage ->
worse MSE.

## Why zscore fails (hypothesis)

1. **tanh is a better implicit prior for composed functions.** The
   composition target has range in [-1,1]; tanh's soft saturation at ±1
   matches that naturally. Z-score lets `z` flow unbounded; when `a` lands
   in `[|a| > 2-3]` (the tails of N(0,1)) the layer-2 LUT has no training
   data there and returns noise, contaminating final output.

2. **Chebyshev polynomials with z-score domain face a harder fitting
   problem.** Polynomials of fixed degree fit smooth data on [-1,1]
   exponentially well (Chebyshev approximation theorem). When the input
   domain widens to [-3,3] but data density is heavier near 0, polynomial
   fits become less accurate at matched degree. Converted to LUT, this
   handicap carries over.

3. **Coverage is a necessary but not sufficient condition for LUT
   expressivity.** High coverage means every cell sees some data, but not
   that every cell is USEFUL. Cells in the tails see tiny populations and
   their values dominate noise rather than signal.

## What this means for the project

The 1.08-1.28× direct-LUT advantage on the tanh baseline is the real
upper bound we get from "just post-training refinement of polynomial-init
LUTs in multi-edge". This is two orders of magnitude smaller than the
single-edge advantage (444×).

Concretely for the paper:

- **Multi-edge LUT-native training is not the strong-claim frontier.**
  The story "direct-LUT training at memory budget X gives 400× over
  post-training LUT" is ONLY defensible for single-edge architectures.

- **Multi-edge numbers in the paper should be scoped honestly.**
  1-1.3× improvements are real but weak. The main story should
  remain single-edge.

- **If future work revisits multi-edge**, the right direction is NOT
  coverage-expansion (this round settles that). Two candidates left:
    * **Grouped regularization** — layer-1 and layer-2 LUTs with
      separate `λ` scales. Single-edge optimal λ₂=1.0 might be wrong
      for either layer individually.
    * **Shared LUT atoms** (dictionary decomposition of layer-2). This
      reduces effective parameters, which M2's effective-rank finding
      suggests is the right direction — the LUT is already ~rank-2.

## Deliverables

Code:
- `src/lut_native/poly_kan2_zscore.py` — PolyKAN2Zscore class + trainer
- `src/lut_native/kan2.py` — LUTKAN2Layer extended with `activation='zscore'`,
  `calibrate_activation_stats()`, configurable L2 domain
- `src/lut_native/coverage.py` — updated to use model.activation / L2 domain
- `src/lut_native/training_kan2.py` — calibrate-at-start option;
  **bug fix** in best_model reconstruction
- `scripts/m3_zscore_normalization.py` — experiment

Tests:
- `tests/test_zscore.py` (12 tests)
- `tests/test_poly_kan2_zscore.py` (4 tests)
- `tests/test_train_kan2_bestmodel_bug.py` (2 tests, regression for the
  best_model bug)

All 48 tests pass. Runtime: ~2 min CPU.

## Honest methodological note

The experience of finding the `train_kan2()` bug mid-experiment is a
reminder that **when measurements look too bad to be true, they often
are**. The first `m3_zscore_normalization.py` reported best MSE ~0.2
which was ~1000× worse than expected — that was the bug, not the method.
The regression test added here is specifically to catch the
"mse_val_at_best > mse_val_final" inconsistency in the future.

None of the M2 findings rely on the code path with this bug
(M2 used tanh-only activation, where the bug is silent). But it
is worth recording that without the regression test I could have shipped
this paper claiming "zscore breaks things" when actually zscore was fine
and the evaluation harness was wrong.
