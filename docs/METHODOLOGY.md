# Methodology and design decisions

This document explains *why* the experiments are set up the way they are,
and what the known limitations are. Nothing here is marketing; the goal is
for a reviewer (or a future reader) to be able to criticize the claims
accurately.

## 1. What is being tested

**Claim (H1):** On a fixed memory budget (K × L LUT entries), training LUT
values *directly* with a curvature prior produces a lower test MSE than
sampling a fitted polynomial and quantizing it (the v2.1 pipeline).

**Claim (H3):** The advantage is specific to the small-L regime. At large L,
post-training LUT sampling approaches the polynomial's accuracy and the
advantage vanishes.

**Not claimed:** that LUT representation is more expressive than a
polynomial of matched memory. Where the polynomial fits in RAM (e.g. a
deg=20 float32 polynomial takes 84 bytes), the polynomial is **much** more
accurate than any LUT we tested. Direct-LUT is a deployment-regime
technique, not a representation-class improvement.

## 2. Design decisions (and why)

### 2.1 Half-open sampling, endpoint-inclusive interpolation

The main `lut-kan` v2.1 repo samples the segment grid as
`x[k, i] = x_min + k*seg_width + i*(seg_width/L)` for `i = 0..L-1`, i.e.
the rightmost sample in segment `k` is at `x_{k+1} - step`, NOT at the knot.

At runtime, however, interpolation is endpoint-inclusive: `pos = u*(L-1)`
places the last LUT cell at the segment boundary.

These two conventions are asymmetric. We preserve them **exactly** because
the aim is to produce LUTs that drop into the existing v2.1 C kernels
without modification. This asymmetry has one concrete consequence that
affected the design of the boundary penalty (section 2.3).

### 2.2 Validation-split best-model selection

Early draft experiments reported test MSE from the final training epoch.
Given 512 LUT parameters and 500 training points, this is susceptible to
overfitting, and any "advantage" might be a train-MSE improvement not
reflecting generalization.

All current experiments:

- use a distinct `(x_val, y_val)` drawn from the same distribution (separate
  RNG seed) to select the best-epoch LUT;
- report test MSE on a *deterministic* `x_test` grid that never changes
  across seeds or regimes — apples-to-apples.

Without this change, the measured ratios would be optimistic by an unknown
factor. See `tests/test_reproducibility.py` for the paired determinism test.

**Concrete magnitude.** `scripts/diagnostic_training_dynamics.py` produces
`results/diagnostics/training_dynamics.png`, which shows: at the best
regularization setting (λ₂=1.0), val MSE reaches ~8e-7 around epoch 25-35,
then deteriorates to ~1e-5 by epoch 150. Naive "final-epoch" reporting
would claim our method achieves 1.3e-5 instead of 7.9e-7 — **roughly a
16× pessimistic bias**. This matters: the post-training baseline is
deterministic and does not have this degradation, so a naive protocol
would understate the direct-LUT advantage.

### 2.3 Boundary continuity penalty: formulation

A naive penalty `((lut[:-1, -1] - lut[1:, 0])**2).mean()` penalizes
`f(x_{k+1} - step) ≠ f(x_{k+1})`, which is wrong by construction for any
non-flat function under half-open sampling.

The corrected penalty extrapolates the last in-segment slope:

```
predicted_at_knot = lut[k, L-1] + (lut[k, L-1] - lut[k, L-2])
penalty = (predicted_at_knot - lut[k+1, 0])**2
```

This is what `boundary_continuity_penalty` computes. The slope-continuity
counterpart is `boundary_slope_penalty`.

**Empirical finding:** even the corrected boundary penalty does **not**
help on top of the best second-difference penalty (H1a secondary ablation:
lbv=0 → MSE 4.3e-7, lbv=0.1 → MSE 2.0e-6). We believe this is because the
learned LUT is already effectively rank-2 across segments (section 3.3),
so boundary smoothness emerges for free; enforcing it as an explicit
constraint just reduces available capacity.

### 2.4 Paired bootstrap confidence intervals

Standard deviations across seeds can mislead: different methods may have
wildly different variance. We report paired bootstrap CIs on the log-ratio
`log(post_MSE) - log(direct_MSE)`. When post-training is deterministic
(same data, same polynomial), the same scalar is "paired" to each direct
seed. This gives an honest 95% interval for the ratio.

See `lut_native.metrics.paired_bootstrap_ci`.

### 2.5 Effective rank diagnostic (H2)

Confound: perhaps direct-LUT wins because the optimizer converges to
something that essentially *is* a polynomial (shared shape across segments,
varying by scale/offset), and the 512 nominal parameters collapse to ~d
DOF. If so, "LUT-native training" is better described as "polynomial
training with a piecewise-linear evaluator".

We test this by computing the singular value spectrum of `lut - per_segment_mean`
(i.e., after removing DC offsets). Rank tells us how many shape templates
the LUT is actually using.

**Finding:** for sine target at K=16, L=32, the effective rank at 99%
energy is **2** across *all* regimes we tested — post-training LUT,
direct-LUT without regularization, and direct-LUT with λ₂ ∈ {0.01, 0.1, 1.0}.

This is consistent with the sine target itself: two dominant frequencies
plus per-segment offset ≈ 2 independent shapes. It implies direct-LUT is
**not** exploiting its 512 nominal DOF. The advantage comes from *where*
those two dominant modes are placed in LUT-space — specifically, chosen
to minimize piecewise-linear approximation error on the data rather than
to be "samples of a smooth function".

This reframes the claim: direct-LUT is essentially learning the *optimal
piecewise-linear* approximation under Adam with curvature regularization,
using LUT storage as a convenient substrate. It does not need all K·L DOF.

## 6. Multi-edge KAN on 2D target (H4)

v0.1.0 only had single-edge [1→1] evidence. v0.2.0 extends to a proper 2-layer
KAN [2 → H → 1] on a 2D target where univariate representation is not enough:

    f(x, y) = sin(πx) + 0.5 · cos(2πx·y)

The `x·y` term forces genuine cross-variable structure.

### Key finding: single-edge advantage does NOT transfer intact

At matched architecture [2→4→1], K=16, L=32:

| Method | Test MSE | Ratio vs Post-LUT |
|---|---|---|
| Polynomial KAN (deg=8) | 3.89e-04 ± 2.4e-04 | 0.74× |
| Post-training LUT-KAN (uint8) | 2.89e-04 | 1.0× |
| Direct LUT-KAN (uint8) | 1.74e-04 ± 1.8e-05 | **1.66× [1.54, 1.84]** |

The 444× advantage from single-edge sine collapses to 1.66× here. Two reasons:

1. **Learning-rate and lambda mismatch.** Single-edge used lr=1e-2 with λ₂=1.0.
   Multi-edge requires lr=5e-4 and λ₂=0 (any curvature penalty at single-edge
   scale destabilizes the 8-LUT joint optimization). The per-edge effective
   regularization is now n_edges× weaker if same λ is used.

2. **Dead-cell pathology in layer 2.** In a converged model, hidden activations
   occupy only ~30% of the [-1, 1] range after tanh squash (std ~0.12). Layer-2
   LUT cells corresponding to |a_h| > 0.5 are never visited and remain noisy.
   Direct-LUT training cannot fix unvisited cells — by construction, gradients
   on them are zero.

### Honest conclusion

Direct-LUT training has a genuine but much smaller advantage in multi-edge
KANs. The "single-edge 444×" number from v0.1.0 is real but is NOT the
representative operating point.

What would improve this? Hidden-unit activation spreading (e.g. BN-like
rescaling before tanh, or KAN-specific L2 output-variance loss per hidden
unit) would allocate more active range to layer-2 LUTs. Not investigated.

## 7. Resource accounting (H5)

For the same 2D KAN task, measured on x86 CPU:

| Method | Memory | Latency | MSE |
|---|---|---|---|
| Polynomial KAN (deg=8) | 432 B | 1.35 µs/sample | 2.77e-04 |
| LUT-KAN K=8, L=16 | 1920 B | 1.64 µs/sample | 3.33e-03 |
| LUT-KAN K=16, L=32 | 6912 B | 1.71 µs/sample | 5.94e-04 |
| LUT-KAN K=32, L=64 | 26112 B | 1.78 µs/sample | 3.21e-04 |

**On CPU, polynomial wins on all three axes.** This is NOT the regime where
LUT helps. LUT advantage relies on:
- No FPU (every float multiply is emulated at ~10-100 cycles vs 1 for integer add)
- Cache-friendly sequential access to small LUTs
- Fixed-point arithmetic throughout

Which is MCU. Main lut-kan v2.1 repo reports 20-30× MCU speedups for post-training
LUT vs polynomial; direct-LUT would inherit those deployment properties automatically
since the LUT byte-layout is unchanged.

## 8. Known limitations

1. **Single-edge only.** All experiments are single-input, single-output
   `[1 → 1]`. Multi-edge KANs may behave differently because of gradient
   competition between edges and learned feature distributions.

2. **Smooth init assumed.** All direct-LUT runs start from polynomial-init.
   A random-init ablation would separate "effect of training" from "effect
   of good starting point". We have not run this.

3. **Scalar λ₂ across L.** The best λ₂ is fitted at L=32 and reused across
   L. Section H1b shows the crossover at L=128 where direct-LUT loses; it
   is plausible that a smaller λ₂ would recover. Not investigated.

4. **Adam-specific.** We did not test SGD or second-order optimizers. The
   sparse-gradient structure of LUT training (2 cells per sample) interacts
   with per-parameter adaptivity in a way that may not generalize.

5. **No QAT.** Post-hoc uint8 quantization adds <1% MSE across all regimes.
   Int8-symmetric or int4 were not tested and may show different behavior.

6. **Data size fixed.** 500 training points. Dataset scaling not studied.

## 9. What would falsify the claims

- **H1 would fail** if: at any L ≤ 32 on any of the three targets, the
  paired bootstrap CI on the post/direct ratio includes 1.0. Current data
  shows ratios from 37× to 1100× with CIs excluding 1.0 by large margins.

- **H3 is already partially borne out**: at L=128 on sine, the ratio is
  0.84× with CI [0.65, 1.21], which *includes* 1.0 — i.e., we cannot claim
  direct-LUT is better than post-training at this L, and one could argue
  it's slightly worse. We do not claim an advantage there.

- **H2 (effective rank)** might change with different targets or with a
  much richer data distribution. It is a negative diagnostic finding
  specific to the tested setting. A future experiment with a high-rank
  target (e.g., a function that genuinely differs across segments, such
  as a sum of K different local bumps) would probe this.

## 10. Things we deliberately did *not* do in v1

- Multi-edge KAN
- Quantization-aware training
- MCU benchmarking / latency / memory measurement
- Comparison with other direct-LUT baselines from literature (LUT-NN, etc.)
- Hyperparameter sweep on `K`

These are not essential to defend H1/H3 on single-edge toy tasks. They
become necessary for a full paper but would dilute the minimal reproducible
claim this repository is designed to support.
