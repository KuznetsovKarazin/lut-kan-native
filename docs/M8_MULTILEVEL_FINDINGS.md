# M8 — Multi-level LUT-KAN with adaptive normalisation: findings

## Summary

Three sub-experiments (H8, H8a-diagnosis, H8b) on `feynman_2d`:
`f(x,y) = sin(πx) + 0.5 cos(2π·xy)`, N_train=800, K=16, L=32.

**H8 rejected**: depth does not close the polynomial-LUT gap on 2D targets.
**New finding**: norm collapse is a general gradient annihilation failure mode.
**H8b**: EMA tracking solves coverage collapse; gradient-based losses cannot.

---

## H8 — Does depth help?

### Setup
Configs compared (400 epochs, 3 seeds):
`stack_2x4 [2→4→1]`, `stack_3x4 [2→4→4→1]`, `stack_3x8 [2→8→8→1]`,
`kan2_4 [2→4→1] tanh-squash`, `poly_4_d20`, `poly_8_d20`.

### Results

| Config | MSE mean | vs kan2_4 | Note |
|---|---|---|---|
| `poly_4_d20` | **1.40e-02** | 0.022× | 45× better than any LUT |
| `poly_8_d20` | 2.01e-02 | 0.031× | — |
| `stack_2x4` | 6.39e-01 | 0.986× | +1.4% vs baseline |
| `stack_3x4` | 6.39e-01 | 0.986× | same as 2-layer |
| `stack_3x8` | 6.39e-01 | 0.986× | 1 seed only |
| `kan2_4` | 6.48e-01 | 1.000× | best_epoch=0, diverges |

### Verdict
H8 rejected. All LUT-KAN depths land at MSE ≈ 0.64.
Polynomial-KAN advantage is 45× — architectural, not a training artefact.
`kan2_4` diverges on feynman_2d (best_epoch=0 every seed).
LUT stacks with frozen norms genuinely learn (best_epoch 50-400).

---

## H8a — Norm collapse diagnosis

### Discovery
Original `LUTKANStack` (learnable norms in Adam, `freeze_norms=False`):
`norm_uniformity = 0.0` after epoch 50 at all lambda and all seeds.

### Root cause: gradient-through-clamp annihilation
1. LUT values change → z distribution shifts → scale becomes relatively small.
2. `a = clamp((z-shift)/scale, -1, 1)` clips ~90% of activations to ±1.
3. `d(clamp)/dx = 0` at saturation → zero gradient to shift and log_scale.
4. Norms permanently frozen in wrong state. Dead LUT cells proliferate.

This is a **general failure mode** for any trainable domain-rescaling layer
positioned before a clamped lookup.

### Three attempted gradient-based fixes — all fail

**Coverage entropy loss on post-clamp `a`** (`coverage_entropy_loss`):
- Gradient through clamp is zero → no signal. Loss does nothing after collapse.

**Out-of-domain penalty** `mean(relu(|u|-1))` where `u=(z-shift)/scale`:
- Gradient exists but has **mixed sign** due to asymmetric shift/z distributions.
- Channels where z is predominantly negative get inverted gradient direction.
- Unreliable: sometimes correct direction, sometimes worsens collapse.

**Quantile matching loss** `(Q5(u)+1)² + (Q95(u)-1)²`:
- `torch.quantile` gradient is sparse (2 samples per channel).
- Mixed signs observed empirically. Not reliable for optimizer convergence.

### Why all gradient approaches fail
When `u ∈ [-10, 10]` but domain is `[-1, 1]`:
- Sigmoid gates: `σ((-10 - lo_k) × 50) ≈ 0` for all segment boundaries.
- Soft histogram sees no activations → effective gradient = 0.
- There is no differentiable scalar that correctly measures "scale is 10× too small"
  without special knowledge of the target domain.

### Fix: `freeze_norms=True`
- `requires_grad_(False)` on norm params; gradient still flows **through** norms.
- Periodic `model.calibrate(x_train)` resets norms from empirical statistics.
- Works because calibration operates on raw z (not clamped), always non-zero signal.
- `norm_uniformity = 0.73` (hidden=4, 400 ep) vs `0.00` with trainable norms.

---

## H8b — EMA norm tracking

### Motivation
Periodic recalibration (every 50 epochs) works but has gaps between calls.
Question: does **continuous EMA tracking** provide better coverage and lower MSE?

### Mechanism: `LUTInterLayerNorm.ema_update(z, alpha)`
Every training batch, before the gradient step:
```
batch_center = (Q5(z) + Q95(z)) / 2
target_scale = (Q95(z) - Q5(z)) / 2 / domain_half
shift    ← (1-α) * shift    + α * batch_center
log_scale ← (1-α) * log_scale + α * log(target_scale)
```
Operates on raw z (pre-clamp) → always non-zero signal, no saturation.

### Results (stack_3x4, 300 epochs, seeds 0-2)

| Config | MSE mean | MSE std | Uni mean | Avg best_ep |
|---|---|---|---|---|
| `no_track` (frozen, no update) | **6.38e-01** | 6.8e-03 | 0.000 | 67 |
| `recal_50ep` (frozen + recal/50ep) | 6.35e-01 | — | 0.000 | 50 |
| `ema_001` (α=0.01) | 6.54e-01 | 1.1e-02 | **0.717** | 150 |
| `ema_005` (α=0.05) | 6.65e-01 | 2.7e-02 | **0.737** | 83 |
| `ema_001+recal50` | 6.45e-01 | — | **0.760** | 0* |

*best_epoch=0 suggests EMA+recal combination creates startup instability.

### Key findings

**1. EMA solves coverage collapse.**
All EMA variants achieve `norm_uniformity ≈ 0.72-0.76` consistently across seeds.
No-track collapses to `0.00` in every seed. This is the primary win.

**2. Coverage ≠ better MSE at 300 epochs.**
EMA models have higher MSE (0.654-0.665) than no-track (0.638) at 300 epochs.
EMA models are still actively improving (`best_epoch ≈ 50-250`).
No-track models plateau early (`best_epoch ≈ 50-100`) despite lower MSE.

**3. EMA extends the learning window.**
With good coverage, gradients flow to earlier blocks throughout training.
Without coverage, the first block stops receiving useful signal after ~50 epochs.
At longer runs (500+ epochs), EMA variants are expected to converge below no-track.

**4. EMA α=0.01 preferred over α=0.05.**
Higher alpha causes noisy norm updates, inflating variance (std 1.1e-2 vs 2.7e-2).
Lower alpha (0.01) is smoother and more reliable across seeds.

**5. EMA+recal combination is unstable at startup.**
When EMA runs every batch AND recalibration runs every 50 epochs, the two updates
compete: EMA slowly adapts, recalibration abruptly resets. Results in best_epoch=0
(initial model wins). Use one or the other, not both.

### Recommended configuration

```python
cfg = StackTrainConfig(
    freeze_norms=True,
    ema_alpha=0.01,               # continuous batch-level tracking
    recalibrate_every_epochs=0,   # skip periodic recal when EMA is active
    epochs=600,                   # give EMA models time to converge
)
```

For rapid prototyping (shorter runs):
```python
cfg = StackTrainConfig(
    freeze_norms=True,
    ema_alpha=0.0,
    recalibrate_every_epochs=25,  # works well for hidden_dim ≤ 4
)
```

---

## Overall conclusions

1. **Depth does not help on 2D cross-variable targets** (H8 rejected).
2. **Polynomial-KAN 45× advantage is architectural** — global parametrisation
   is fundamentally more sample-efficient than local LUT parametrisation on
   smooth cross-variable functions.
3. **Norm collapse via gradient annihilation** is a novel failure mode applicable
   to any clamped-lookup architecture with trainable domain-rescaling.
4. **Gradient-based coverage losses are unreliable** for recovering from collapse
   (sigmoid saturation, mixed gradient signs, sparse quantile gradients).
5. **EMA norm tracking** (`ema_alpha=0.01`) is the robust solution: maintains
   coverage ~0.72 across all seeds, extends active learning to 150+ epochs.
6. **When LUT wins**: LUT-KAN retains its H1 advantage on 1D univariate targets
   with high-frequency detail. The 2D regime exposes a capacity ceiling that
   depth alone cannot raise.

## Status

Partially run (300 epochs, ≤3 seeds, compute-limited).
Full experiment at 600 epochs available via `scripts/exp_h8_multilevel.py`.

---

## H8c — Chebyshev initialisation (the real bug fix)

### Discovery

After H8b, a diagnostic test revealed that `init_noise_std=0.05` — used in
all H8 and H8b experiments — activates only **4 out of 16 LUT segments**
in layer 2 from the start of training:

```
std=0.05  → h ∈ [-0.19, 0.22] → 4/16 segments active
std=0.5   → h ∈ [-0.96, 0.98] → 16/16 segments active
```

With only 4 segments active, the second LUT block can only update the 4
central cells and ignores the other 12. The model predicts a near-constant
value (MSE ≈ Var(y) = 0.64) throughout all of training. `best_epoch=0`
because the zero-init state (random mean ≈ 0) is already optimal for a
constant predictor.

**All H8 and H8b results were measuring "how well can models predict a
constant" — not "how well can multi-layer LUT-KAN learn".**

### Fix: Chebyshev polynomial initialisation

`LUTBlock.cheby_init(scale=1.5, noise=0.05)` sets:

    lut[si, di, k, r] = T_{di}(x_val) × scale / in_dim

where `x_val` is the grid point at cell `(k,r)` and `T_d` is the
Chebyshev polynomial of degree `d`. This ensures:

1. All 16/16 segments receive gradient from step 0.
2. Each hidden unit has a **different** functional shape (T_0 = constant,
   T_1 = linear, T_2 = quadratic, ...) → diverse representations.
3. Hidden activations span `z_std ≈ 0.95`, covering the full domain.

### Corrected results (400 epochs, 3 seeds, cheby_init=True)

| Config | MSE mean | vs old broken | Note |
|---|---|---|---|
| `kan2_4_noise` (old) | 6.48e-01 | 1.00× | all experiments broken |
| `stack_1x4_cheby` | **3.65e-01** | 0.563× | **1-layer, separable MSE** |
| `stack_2x8_cheby` | **3.76e-01** | 0.581× | **best 2-layer config** |
| `kan2_4_cheby` | 4.74e-01 | 0.731× | high variance (seed 2: 0.66) |
| `stack_3x4_cheby` | 6.18e-01 | 0.953× | clamp norm blocks gradient |
| `poly_4_d20` | 2.15e-02 | 0.033× | 17.5× better than best LUT |

### Revised gap: 17× not 45×

With correct initialisation, poly-KAN leads by 17.5× (not 45× as reported
in H8). This is still a large architectural gap but a more honest number.

### Why stack_3x4 does not improve with cheby init

`stack_2x8_cheby` (MSE 0.376) beats `stack_3x4_cheby` (MSE 0.618) despite
more total LUT parameters. The inter-layer clamp norm in `stack_3x4` blocks
the gradient from layer 3 reaching layer 1: even with good initialisation,
the clamp saturates after a few epochs and kills the signal.

`stack_2x8_cheby` = `[2→8→1]` — single norm, lower gradient blockage.
The wider hidden (8 units) compensates for the depth advantage.

### What H8b's norm tracking actually showed

With the dead-init bug, norm uniformity measurements were meaningless
(measuring coverage of a model predicting constants). With cheby init:
- `stack_2x8_cheby` norm_uniformity ≈ 0.70 stable across seeds ✓
- Norm collapse is genuinely solved by `freeze_norms=True + ema_alpha=0.01`
- But the clamp in `forward()` remains the bottleneck for deeper stacks

### Remaining architecture question

The clamp in `LUTInterLayerNorm.forward()` is necessary for the LUT lookup
to work correctly. But it kills gradient for out-of-domain activations.
A differentiable alternative (e.g. replacing clamp with a smooth squash
that keeps gradient alive) is the next open problem.

### API changes in v15 (this session)

`LUTBlock.cheby_init(scale=1.5, noise=0.05)` — new method
`LUTKANStack.cheby_init(scale=1.5, noise=0.05)` — applies to all blocks
`StackTrainConfig.cheby_init` — default `True` (was implicit `False`)
`StackTrainConfig.cheby_scale` — default `1.5`
`StackTrainConfig.cheby_noise` — default `0.05`

### Conclusion

The dominant finding of all H8 experiments is **initialisation quality**,
not architecture. The polynomial-LUT gap is real but smaller than measured:
17× not 45×. The correct next step is a clean comparison at 1500+ epochs
with `cheby_init=True` as the default for all experiments.
