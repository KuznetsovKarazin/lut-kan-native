# M5a Findings — Low-rank residual (final multi-edge experiment)

## Stop criterion triggered. Closing multi-edge topic.

Per the agreement: one final attempt (M5a), if flat → close everything.
**M5a is flat.** Multi-edge exploration ends here.

## Headline

Low-rank residual parameterization `Δ = U V^T` (per-edge, rank r) does
not improve over full-rank residual training. The ceiling on feynman_2d
stays at **1.22×** over PolyKAN2 (achieved in M4c L1_slower).

### Key table (feynman_2d, 3 seeds)

| rank | params/Δ | test MSE | vs PolyKAN2 | paired vs full |
|---|---:|---:|---:|---|
| full | 6144 | 6.59e-04 | 1.14× | reference |
| 8    | 4608 | 6.46e-04 | 1.16× | not sig |
| 4    | 2304 | 6.59e-04 | 1.14× | not sig |
| 2    | 1152 | 6.60e-04 | 1.14× | not sig |
| 1    |  576 | 6.78e-04 | 1.11× | sig WORSE |

No rank beats full significantly; rank 1 is significantly worse.

## What we learned from M5a specifically

1. **"Excess parameters" is not the bottleneck.** If a small effective
   rank captured the data, rank-1 or rank-2 would outperform full.
   Instead, rank-1 is the worst and rank-8 is indistinguishable from
   full. The model uses more than 2 degrees of per-edge freedom.

2. **Low-rank trades training stability for capacity.** Val-MSE trace
   shows low-rank models converge monotonically to stable final MSE,
   while full-rank overshoots and drifts (same pattern as α<1 in M4a).
   Both mechanisms regularize the optimization path, neither unlocks
   a better minimum.

3. **The post-train best-val MSE is the same regardless of
   parametrization.** Trust-region (small α), anchor penalty (M4b),
   and low-rank (M5a) all produce the same ~1.22× ceiling.
   The ceiling is a property of the (poly-init, lr schedule, best-val
   selection) triple — not of parametrization.

## Consolidated multi-edge findings (M2 through M5a)

### What hypotheses we tested

| Hypothesis | Phase | Outcome |
|---|---|---|
| Layer-2 coverage is the bottleneck | M2, M3 | FALSIFIED (high coverage → worse MSE) |
| Training is destructive to poly-init | M2 destruct | TRUE at lr=1e-2, FALSE at lr=5e-4 |
| Trust region prevents drift | M4a | Stabilizes path, no change in best MSE |
| Proximal penalty prevents drift | M4b | Null (as predicted) |
| Separate per-layer lr helps | M4c | +3-4% relative (statistically real) |
| Progressive unfreezing helps | M4d | Null |
| Low-rank residual helps | M5a | Null (this phase) |

### What actually moved the needle

- Using lr = 5e-4 instead of lr = 1e-2 (single-edge default): small-cap
  drift, lets training actually improve poly-init.
- Val-based best-model selection: saves late-epoch overfit.
- Per-layer lr (lr_l1=1e-4, lr_l2=5e-4): +3-4% over uniform lr.

That's it. Everything else is noise within error bars.

### Final scoped claim

```
Multi-edge direct-LUT training initialized from polynomial coefficients
gives a 1.09-1.22× test-MSE improvement over PolyKAN2 on 2D targets,
using:
  - residual parameterization (α ≤ 1.0, structure doesn't matter for
    best-val MSE, only for training stability)
  - lr_l1 = 1e-4, lr_l2 = 5e-4 (per-layer asymmetry helps marginally)
  - no additional regularization
  - validation-based best-model selection

The improvement is modest compared to single-edge direct-LUT (~400×),
reflecting that polynomial KANs at matched architecture are already
near the expressivity ceiling of this task family.
```

## Why ceiling is ~1.2×, not something more dramatic

Best understanding after M2-M5a:

1. **PolyKAN2 at matched architecture already captures the signal.**
   Degree-8 Chebyshev per edge has plenty of capacity for the targets
   we test. Converting poly coefficients to LUT and then fine-tuning
   the LUT locally cannot add more than what the polynomial already
   represents — it can only tweak the sampling slightly.

2. **The LUT's theoretical advantage (many cells → higher asymptotic
   capacity) does not manifest at this task complexity.** For
   trigonometric + product targets, a degree-8 polynomial is essentially
   exact; adding 512 LUT cells is overkill and training just shuffles
   noise in unvisited cells.

3. **Single-edge advantage was specific to K=16, L=32 configuration
   where the polynomial itself had to fit a target of varying
   structure (sine, cusp, saturating).** In multi-edge, each per-edge
   polynomial has a simpler target (decomposed by KAN structure), so
   the polynomial is already close to optimal.

This explains why extending to more complex targets (not tested) might
revive the multi-edge LUT advantage, but for the standard benchmarks
in this project, multi-edge is what it is.

## What NOT to pursue (even though listed in the plan document)

- **Shared LUT atoms / Dictionary (Variant A).** After M5a's finding
  that per-edge low-rank does nothing, cross-edge shared structure
  via atom mixing is unlikely to help more. The bottleneck is not
  parameter redundancy.
- **Distillation from poly-KAN teacher (Variant F).** Fundamental
  methodology concern: it would measure imitation capacity, not
  intrinsic LUT expressivity. Different paper, different claim.
- **Visited-region masking (Variant C).** Small optimization win at
  best (freezes ~20% of cells). Doesn't change the ceiling question.
- **Soft indexing / STE (Variant G).** Train/eval mismatch. High
  implementation cost for speculative gain.

If the paper needs a bigger multi-edge number, the place to look is
**harder target functions** (non-smooth cusps, discontinuities,
high-frequency oscillations), not more optimization tricks on the
existing targets.

## Deliverables

Code (new in M5a):
- `src/lut_native/kan2_low_rank.py` — `LowRankResidualLUTKAN2Layer` with
  per-edge rank constraint
- `src/lut_native/training_low_rank.py` — training loop
- `tests/test_low_rank.py` — 8 tests (construction, rank constraint
  correctness, numpy/torch parity, regression for the M3 bug)

Experiments:
- `scripts/m5a_low_rank.py`
- `results/M5a_low_rank/{composition_1d,feynman_2d}/{summary.json,plot.png}`

Test total: **63 passing.**

Docs: this file (`docs/M5a_FINAL.md`).

## What happens next

Per your call: **close multi-edge topic, switch to single-edge
strengthening**. Concrete priorities:

1. **Extend H1 sweep to K ∈ {8, 16, 32}** on the three targets.
   Current paper has only K=16. Reviewer will ask.

2. **Random-init vs poly-init ablation for single-edge.** Current
   single-edge story initializes from Chebyshev polynomial. Need to
   show random-init also works, or explain why poly-init matters.

3. **MCU latency benchmark.** The core motivation for LUT is MCU
   deployment. No latency numbers yet = incomplete story.

If you want, I can scope each of these as a separate phase (e.g. M6a,
M6b, M6c) and we'll work through them. None of these require
architectural changes — just experimental completeness.
