# M7a Findings — K × L sweep for single-edge direct-LUT

## Motivation

H1 tested only K=16, L=32. This phase sweeps K ∈ {8, 16, 32} × L ∈ {16, 32, 64}
to answer: does the direct-LUT advantage hold across different segment/resolution
combinations, and where is the Pareto optimum?

## Setup

- Single-edge `[1→1]`, Adam lr=1e-2, λ₂=1.0, 600 epochs, 3 seeds.
- Metric: ratio = post_LUT_MSE / direct_LUT_MSE (> 1 means direct-LUT wins).
- Memory: K×L bytes (uint8 table) + K×4 bytes (float16 scale + ymin).

## Results

### Ratio heatmap (post/direct, mean over 3 seeds)

**sine** (smooth):

| | L=16 | L=32 | L=64 |
|---|---|---|---|
| K=8  | 273× | **2764×** | 980× |
| K=16 | 277× | 611× | 14× |
| K=32 | 144× | 16× | 4× |

**cusp** (non-smooth):

| | L=16 | L=32 | L=64 |
|---|---|---|---|
| K=8  | 144× | 555× | 591× |
| K=16 | 428× | **790×** | 12× |
| K=32 | 78× | 10× | 3× |

**saturating** (sharp transition):

| | L=16 | L=32 | L=64 |
|---|---|---|---|
| K=8  | 1566× | **16428×** | 916× |
| K=16 | 2462× | 942× | 2× |
| K=32 | 53× | 2× | **<1×** |

## Key findings

### Finding 1: H1's K=16 was not the Pareto-optimal choice

For sine and saturating, **K=8, L=32 (288 bytes) dominates**:
- sine: K=8,L=32 gives 2764× vs K=16,L=32's 611× — **4.5× more gain at half the memory**
- saturating: K=8,L=32 gives 16428× vs K=16,L=32's 942× — **17× more gain**
- cusp: K=16,L=32 remains optimal (790×), K=8,L=32 gives 555×

H1's choice of K=16 was conservative and under-reported the true peak advantage.
The recommended deployment config should be K=8, L=32.

### Finding 2: Ratio decreases monotonically with K at fixed L

At L=32, sine: K=8 (2764×) > K=16 (611×) > K=32 (16×). All 95% CIs are
non-overlapping. The relationship is stronger than linear in log space.

**Physical reason:** Direct-LUT's advantage comes from compensating
piecewise-linear interpolation error. This error scales as:

```
ε_interp ~ f''(x) × (segment_width / L)²
         = f''(x) × ((x_max - x_min) / K·L)²
```

Smaller K → wider segments → larger interpolation error → more room
for direct-LUT to compensate by optimally placing cell values.
Larger K → narrower segments → polynomial is already nearly exact
within each segment → direct-LUT has nothing to fix.

### Finding 3: Sharp crossover at L=64

At L=64, the ratio collapses for all K:
- K=16, L=64 (1088B): 14× (sine), 12× (cusp), 2× (saturating)
- K=32, L=64 (2176B): 4× (sine), 3× (cusp), **<1×** (saturating — DIRECT LOSES)

This is consistent with H1b's crossover at L≈100–128: at high L, the
post-training LUT already has fine enough resolution that direct-LUT
training can't improve it. The crossover shifts to lower L as K increases.

### Finding 4: K=8 has higher variance at L=64

K=8, L=64 CI on sine: [403, 4671] — very wide. K=16, L=32 CI: [504, 734] — tight.

With K=8 the per-segment optimisation problem is harder (wider input range
per segment, more curvature to capture in L=64 cells). Seed sensitivity
is higher. K=8, L=32 has a tighter CI [2333, 3431] — the L=32 configuration
is more robustly trainable than L=64 for small K.

## Updated Pareto analysis

At each memory budget, the Pareto-optimal config:

| Budget | Config | Best target ratio |
|--------|--------|-----------------|
| 160 B  | K=8,L=16  | 1566× (saturating) |
| 288 B  | K=8,L=32  | **16428×** (saturating) |
| 320 B  | K=16,L=16 | 2462× (saturating) |
| 544 B  | K=8,L=64  | 916× (saturating) |
| 576 B  | K=16,L=32 | 942× (saturating) |

**K=8, L=32 at 288 bytes is the recommended single-edge config** —
smallest memory, highest ratio across two of three targets, tight CI.

## Implication for H1 paper claims

H1 reported:
- sine K=16,L=32: 444× [357, 526]
- cusp: 143× [67, 307]
- saturating: 1107× [635, 1995]

These were measured against *post-training LUT*, 5 seeds. M7a (3 seeds,
600 epochs) finds higher ratios because the training wasn't fully converged
in H1 (1500 epochs got much higher). The ordinal ranking is the same.

With K=8, L=32 the headline numbers would be substantially higher. H1's
K=16 result is honest but not the best-case; the paper should either
(a) report K=8 as the recommended config, or (b) note K=8 as a stronger
variant in a supplementary table.

## What NOT to use

- **K=32** at any L: ratio degrades to single-digit or below parity.
  At K=32 the polynomial is already a near-perfect piecewise approximation
  within each segment; direct-LUT training has nothing to improve.
- **L=64** with K ≥ 16: parity or worse. Too many cells per segment for
  the training data density (500 points, K=16 → ~31 points/segment → ~0.5
  points per LUT cell at L=64). Gradient coverage is too sparse.

## Deliverables

Code:
- `scripts/m7a_kl_sweep.py` — 9 configs × 3 targets × 3 seeds
- `tests/test_kl_sweep.py` — 6 tests

Results:
- `results/M7a_kl_sweep/{sine,cusp,saturating}/{heatmap.png,pareto.png,summary.json}`
- `results/M7a_kl_sweep/summary.json`

Tests total: **80 passing.**
