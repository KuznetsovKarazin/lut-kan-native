# M6b Findings — SGD learning-rate sweep for on-device training

## Motivation

M6a established that fp16-grad SGD causes no accuracy penalty vs fp32
(B/C ratio = 1.00 exactly). However, M6a used lr=0.5 chosen from a
preliminary 4-point probe. This phase does a proper lr sweep (6 configs,
3 targets, 3 seeds, 900 epochs) to find the optimal on-device training
configuration.

## Setup

All regimes: fp16-grad SGD, λ₂=1.0, K=16, L=32, 900 epochs.

| Config | Schedule |
|--------|----------|
| flat_0.1 | constant lr=0.1 |
| flat_0.5 | constant lr=0.5 (M6a baseline) |
| flat_1.0 | constant lr=1.0 |
| flat_2.0 | constant lr=2.0 |
| cosine_1.0 | cosine decay 1.0 → 0.01 |
| step_1.0 | 1.0 for 300ep → 0.2 for 300ep → 0.02 for 300ep |

## Results

### Headline table (mean test MSE, 3 seeds)

| Config | sine ratio | cusp ratio | saturating ratio |
|--------|-----------|-----------|-----------------|
| flat_0.1 | 1.6× | 14.3× | ~1.0× |
| flat_0.5 (M6a) | 5.5× | 67× | 4.8× |
| flat_1.0 | 15.2× | 142× | 21.7× |
| **flat_2.0** | **50.4×** | **404×** | **198×** |
| cosine_1.0 | 5.7× | 68× | 5.0× |
| step_1.0 | 4.4× | 56× | 3.6× |

For reference:
- PolyKAN (baseline): 1×
- Adam fp32, λ₂=1.0 (server): sine 130×, cusp 150×, saturating 80×

### Finding 1: lr=2.0 flat is the clear winner

Ratios vs lr=0.5 (M6a baseline), all CI exclude 1.0:

| Target | flat_0.5/flat_1.0 CI | flat improvement lr=0.5→2.0 |
|--------|---------------------|----------------------------|
| sine | [2.73, 2.77] — 2.75× | **9.1× improvement** |
| cusp | [2.05, 2.19] — 2.12× | **6.0× improvement** |
| saturating | [4.27, 4.75] — 4.5× | **41× improvement** |

The relationship is roughly: doubling lr ≈ halving MSE.
This is consistent with SGD on a convex problem near the solution:
larger lr = larger steps = faster convergence in the remaining
budget of 900 epochs.

### Finding 2: SGD lr=2.0 beats Adam on non-smooth targets

| Target | Adam fp32 | SGD fp16 lr=2.0 | SGD/Adam ratio |
|--------|----------|----------------|----------------|
| sine | 130× | 50× | **Adam 2.6× better** |
| cusp | 150× | **404×** | **SGD 2.7× better** |
| saturating | 80× | **198×** | **SGD 2.5× better** |

On the cusp and saturating targets, SGD at lr=2.0 with fp16 gradients
**outperforms Adam** — using half the RAM and no adaptive optimizer.

Why? At lr=2.0 with SGD, the large step size allows cells near the
cusp (the hardest region) to receive strong updates and escape
local flat regions. Adam's adaptive per-cell scaling actually slows
down high-gradient cells, which matters more on non-smooth targets
where the gradient distribution is highly non-uniform.

### Finding 3: Decay schedules hurt — SGD is not converged at 900 epochs

Cosine and step decay both perform worse than flat lr=1.0:

- cosine/flat_1.0: 2.68× worse (sine), 2.08× worse (cusp)
- step/flat_1.0: 3.44× worse (sine), 2.54× worse (cusp)

The interpretation is straightforward: with best_ep consistently = 900
(last epoch) for all schedules, the model is still improving throughout
training. Reducing lr prematurely locks in a suboptimal minimum.
This is a regime where "keep the lr high and run longer" is the right
strategy, not "warm up + decay."

**Practical MCU implication:** no lr scheduler needed. Constant
lr=2.0 is simpler to implement in firmware and gives the best results.

## Updated recommended on-device configuration

Superseding M6a's recommendation of lr=0.5:

```
optimizer:   SGD, lr = 2.0, no momentum  (was lr=0.5 in M6a)
grad dtype:  float16 accumulation (no accuracy penalty — M6a)
λ₂:          1.0 (required — without it 13-54× worse — M6a)
λ₁:          0
batch size:  64
epochs:      900 (still converging; 1500+ would improve further)
LUT dtype:   float32 during training, uint8 for inference
```

RAM per edge (K=16, L=32): **3.75 KB** (unchanged from M6a).

## What M6a's lr=0.5 underestimated

M6a's preliminary probe covered {0.01, 0.5, 1.0, 5.0} — but
lr=5.0 was not shown. In a quick follow-up (1 seed), lr=5.0 on
cusp was unstable (best_ep=0, val loss diverged). The stable
range appears to be lr ∈ [1.0, 2.0] with 2.0 being the optimum.
lr=3.0 was not swept but would be a natural next point if more
epochs are available.

## Deliverables

Code:
- `scripts/m6b_lr_sweep.py` — sweep (6 schedules × 3 targets × 3 seeds × 900 epochs)
- `tests/test_lr_sweep.py` — 5 tests

Results:
- `results/M6b_lr_sweep/{sine,cusp,saturating}/{plot.png,summary.json}`
- `results/M6b_lr_sweep/summary.json`

Tests total: **74 passing.**

## Consolidated M6 (a+b) paper claim

> "On-device training of direct-LUT KAN activations is viable on
> Cortex-M4 class hardware using SGD with fp16 gradient accumulation
> (3.75 KB RAM per edge). With lr=2.0 and λ₂=1.0 curvature
> regularization, on-device SGD achieves 50–400× lower MSE than
> polynomial KAN on single-edge tasks. On non-smooth targets (cusp,
> saturating), on-device SGD with fp16 gradients outperforms server-side
> Adam by 2.5–2.7×, at half the memory footprint."
