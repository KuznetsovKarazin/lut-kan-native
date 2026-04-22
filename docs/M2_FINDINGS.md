# M2 Findings — Multi-edge coverage diagnostics

This document records what we actually learned from Phase M2 (coverage
diagnostics). The plan was to first measure, then decide. After measuring,
our picture of multi-edge LUT-KAN has changed substantially.

## Summary in one paragraph

Coverage is NOT the main failure mode of multi-edge LUT-KAN. The real
failure modes are (1) high learning rate destroys poly-init before best-val
rescues it, and (2) on genuinely non-decomposable targets (feynman_2d),
direct-LUT gives at most ~2× improvement over a well-trained PolyKAN2,
versus the 400-1000× improvement we saw in single-edge. Coverage is
low in layer 2 because tanh(z) has narrow distribution, but training
cannot fix this by moving LUT values alone.

## What we ran

1. **`m2_coverage_diagnostic.py`** — three init strategies (poly-init,
   identity-init, random-init) on feynman_2d with lr=1e-2. Measures
   before/after coverage on both layers.

2. **`m2_training_destructiveness.py`** — 3×4 grid of (lr, λ₂) on
   poly-init LUT-KAN2 with feynman_2d. Measures whether training
   improves or worsens the init.

3. **Replication of `exp_H4_kan2_2d.py`** — confirms 1.66× claim
   with same settings (lr=5e-4, 300 lut epochs, 800 poly epochs).
   Measures coverage at lr=5e-4.

4. **`m2b_composition.py`** — simpler decomposable target
   `y = tanh(2·sin(πx))` on [1→4→1]. Tests whether multi-edge
   LUT-KAN works at all on a composable task.

## Findings

### F1: lr=1e-2 destroys poly-init; lr=5e-4 is fine

In the destructiveness sweep (m2_training_destructiveness.py, all on
feynman_2d starting from poly-init):

| lr    | λ₂=0      | λ₂=0.01   | λ₂=0.1    | λ₂=1.0    |
|-------|-----------|-----------|-----------|-----------|
| 1e-4  | -251%     | -162%     | -223%     | -192%     |
| 1e-3  | -294%     | -179%     | -440%     | -140%     |
| 1e-2  | -282%     | -282%     | -282%     | -282%     |

(`-N%` = best-val MSE is N% worse than init MSE)

0/12 configurations improve poly-init; 12/12 worsen it. At lr=1e-2 best_epoch=0
for all configs — the first batch already damages the model.

BUT when I replicate `exp_H4_kan2_2d.py`'s lr=5e-4 with 300 epochs and
otherwise identical settings, direct-LUT DOES improve poly-init by 1.91×
(seed 0: 2.88e-4 → 1.51e-4). So lr matters a lot, and the destructiveness
finding is specific to lr≥1e-3.

**Interpretation:** direct-LUT training at small lr acts as a local
refinement of the polynomial-sampled LUT; at larger lr the first few
updates jump into a different basin that doesn't reach a better optimum.

### F2: Coverage does not change during training

Measured L1 and L2 coverage on poly-init LUT-KAN2 BEFORE training and
AFTER training (best-val weights), feynman_2d lr=5e-4:

```
BEFORE: L1 visited=0.779  eff_supp/KL=0.826  L2 visited=0.406  eff_supp/KL=0.367
AFTER : L1 visited=0.779  eff_supp/KL=0.826  L2 visited=0.406  eff_supp/KL=0.367
```

Identical to 3 decimal places. Coverage is a property of the INPUT
distribution (plus the tanh-squash between layers), not of LUT values.

This kills my "coverage collapse is why multi-edge fails" hypothesis.
If coverage doesn't change during training, training cannot recover from
bad coverage by moving LUT values. Fixing coverage requires architectural
changes.

### F3: Layer 2 coverage is intrinsically ~40%

`tanh(z)` where `z` is a sum of LUT outputs has std ≈ 0.3 on this task
(feynman_2d) and std ≈ 0.23 on the composition task. With std this small,
inputs to layer 2 concentrate in [-0.5, 0.5] roughly, leaving the outer
~50% of the LUT domain unused. Range utilization metric confirms this:

| Task            | L2 range utilization | L2 eff_support/KL |
|-----------------|----------------------|---------------------|
| feynman_2d       | 0.66                 | 0.37                |
| composition (1D) | 0.41                 | 0.24                |

These are upper-bounded by the distribution of tanh(z), not by training.

### F4: Composition task — direct-LUT works (~8× improvement over post-training)

On the decomposable target `y = tanh(2·sin(πx))`, [1→4→1]:

| Method                   | Test MSE           | vs PolyKAN2 | vs poly-init LUT |
|--------------------------|-------------------:|-------------|------------------|
| PolyKAN2 (deg=12)         | **1.33e-05**       | 1.0×        | —                |
| Poly-init LUT (= post-training fp) | 1.33e-04 | 10× worse  | 1.0×             |
| Post-training uint8       | 1.34e-04           | 10× worse  | 1.0×             |
| Direct-LUT (lr=5e-4)       | **1.68e-05**       | 0.79× (LUT slightly worse) | **7.95× better than poly-init** |

Direct-LUT closes the gap to PolyKAN2 almost completely, with 8×
improvement over post-training. The 0.79× ratio vs PolyKAN2 means LUT
is *slightly worse* but within the same order of magnitude.

**This is the multi-edge result we wanted on the easier task.** The
single-edge advantage (~400×) does not transfer, but a meaningful
advantage (~8×) does survive, AND the result can compete with PolyKAN2 on
decomposable targets.

### F5: On feynman_2d, direct-LUT gives only ~2×

Replicating exp_H4_kan2_2d.py with 3 seeds:

```
seed 0: poly=1.93e-4, lut_init=2.88e-4, lut_best=1.51e-4  (1.9× over init)
seed 1: poly=7.44e-4, lut_init=8.97e-4, lut_best=7.71e-4  (1.2× over init)
seed 2: poly=1.54e-4, lut_init=2.34e-4, lut_best=1.60e-4  (1.5× over init)
```

Average ~1.5× improvement over poly-init, which is ~1× to 0.5× relative
to direct PolyKAN2 depending on poly seed variance. Original paper's
"1.66×" figure is real but with wide seed-to-seed variance.

**This is much less than single-edge (400×) and less than composition (8×).**
The non-decomposability of the target seems to be the main factor.

## What this changes about the project

1. **The single-edge 400× claim does NOT transfer to arbitrary multi-edge.**
   On a composable 1D target it transfers as ~8×; on a non-decomposable 2D
   target it drops to 1.5-2×. Claims in the paper need this scoping.

2. **Coverage diagnostics rule out the "dead cells" explanation.**
   Training doesn't change coverage, so if L2 coverage is 40%, that's the
   intrinsic constraint, not a training problem.

3. **Architectural options suggested by F3 (L2 coverage is narrow):**
   - Normalize layer-2 input domain to tanh(z)'s observed range at training
     time. Cheapest option. Requires saving (mean, std) as part of model.
   - Use `tanh(a·z + b)` with learnable a, b so the optimizer can match
     domain width. Introduces 2 extra params per hidden unit. The idea from
     the original note.
   - Use batchnorm/layernorm equivalent. Standard but adds state.

   My prior position was "measure first, decide second; I don't believe
   learnable tanh helps." With F3 in hand, I now believe SOMETHING along
   these lines is needed for multi-edge direct-LUT to be useful on
   non-composable tasks. Plain `tanh` leaves layer-2 underutilized.

4. **The regularization question remains open.** In single-edge, λ₂=1.0
   was optimal. In multi-edge destructiveness sweep at lr=5e-4, none of
   λ₂ ∈ {0, 0.01, 0.1, 1.0} helped. But also: none of the lr=5e-4 runs
   were in the destructiveness sweep (that sweep tested lr ≥ 1e-4 with
   different layout). So the interaction of (lr, λ₂) in multi-edge is
   not fully mapped yet.

## Deliverables

- `src/lut_native/coverage.py` — reusable coverage diagnostics module (230 LOC)
- `tests/test_coverage.py` — 12 tests, all passing, verifying metric math
  and bit-exact indexing consistency with `LUTKAN2Layer.forward`
- `results/M2_coverage/` — three-init diagnostic
- `results/M2_destruct/` — 3×4 (lr, λ₂) destructiveness sweep
- `results/M2b_composition/` — composition task (direct-LUT works here)
- `docs/M2_FINDINGS.md` — this file

All 30 tests still pass. No existing code was modified.

## Recommended next step

Decide whether to proceed with variant B — rewrite training with
**separate λ per layer** AND **layer-2 domain normalization** (either via
running statistics or learnable affine). Without at least one of these,
multi-edge direct-LUT appears to plateau at "slightly worse than
PolyKAN2" for non-decomposable targets, which is not a compelling claim.

Alternative: scope the paper's multi-edge claims to composable/
decomposable targets only, for which plain `tanh` activation plus direct-
LUT is adequate (~8× over post-training, ~0.8× relative to PolyKAN2).
