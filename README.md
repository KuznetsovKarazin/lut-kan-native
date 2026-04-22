# lut-kan-native

Direct training of LUT activation tables for memory-constrained KAN deployment.

Research-branch repository for a specific, carefully-bounded question:

> **On a fixed memory budget, is it better to train LUT values directly
> (with a curvature prior), or to fit a polynomial and quantize it (the v2.1
> lut-kan pipeline)?**

The answer depends heavily on task and architecture. We tested five hypotheses
and present all findings, including the negative ones.

## TL;DR

**Where direct-LUT wins:** single-edge `[1→1]` univariate regression on
smooth or non-smooth targets, at small-to-medium `L`. Up to **~1100×** test
MSE improvement over post-training LUT at equal memory.

**Where direct-LUT loses or ties:** (a) single-edge at large L (parity);
(b) 2-layer KAN `[2→h→1]` on 2D targets, at any tested budget, polynomial-KAN
Pareto-dominates.

This is a technique for a **narrow deployment regime**, not a general KAN
replacement. The repository's value is in providing a rigorous,
reproducible, falsifiable characterization of that regime.

## Headline results

### H1 — Direct-LUT dominates at small L (5 seeds, 95% paired bootstrap CI)

| Target | K=16, L=32 | Post-LUT | Direct-LUT | Ratio |
|---|---|---|---|---|
| `sin(2πx) + 0.5 sin(4πx)` | 576 B | 1.83e-04 | **4.23e-07** | **444× [357, 526]** |
| `|x-0.3| + 0.2 sin(3πx)` (cusp) | 576 B | 3.23e-05 | 3.19e-07 | **143× [67, 307]** |
| `tanh(4x) + 0.15x` | 576 B | 1.43e-05 | 1.58e-08 | **1107× [635, 1995]** |

### H3 — Advantage vanishes at large L (honest crossover)

At `L=8`: direct 37× better. At `L=128`: ratio 0.84× [0.65, 1.21] — CI
includes parity, direct-LUT no longer wins. *We report this; we do not hide it.*

See `results/H1b_memory_sweep/plot.png`.

### H2 — Learned LUT is effectively rank-2 (confound found and disclosed)

SVD of `(LUT − per-segment-mean)` shows effective rank 2 across all regimes,
including post-training LUT. Direct-LUT is **not** using its 512 nominal DOF;
it learns a 2-mode piecewise-linear fit optimized for data rather than for
being "samples of a smooth function." This reframes the claim but does not
invalidate it. See `results/H2_effective_rank/spectrum.png` and
[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) §3.3.

### H4 — Multi-edge fails: polynomial-KAN Pareto-dominates on 2D

On the 2D target `f(x,y) = sin(πx) + 0.5 cos(2π·xy)` with a two-layer
`[2→h→1]` KAN at matched memory budgets (288–1152 bytes), polynomial-KAN
reaches test MSE 2.8e-4, while direct-LUT-KAN's best is 7.9e-4 at larger
memory. At every tested budget the polynomial is better. Single-edge
findings **do not transfer** to multi-edge 2D.

This is a useful negative result: direct-LUT's advantage on single-edge
comes specifically from compensating piecewise-linear interpolation error
on a 1D function. In multi-edge, activations pass through `tanh` and are
then summed — the interpolation-error structure that single-edge exploits
is largely averaged out. See `results/H4_multi_edge_2d/pareto.png`.

### H5 — Resource accounting (CPU measurements)

| method | bytes | ops (f+i) | CPU latency |
|---|---|---|---|
| poly deg=20 f32 | 84 | 80 float | 143 ns |
| poly deg=16 f32 | 68 | 64 float | 100 ns |
| LUT K=16 L=32 u8 | 576 | 13 float + 7 int | **16 ns** |
| LUT K=16 L=16 u8 | 320 | 13 float + 7 int | 16 ns |
| LUT K=8 L=16 u8 | **160** | 13 float + 7 int | 16 ns |

LUT **~6× faster** on CPU (gather+lerp is cache-friendly and has no
dependency chain). On MCU without FPU the advantage is typically much
larger — see the main `lut-kan` v2.1 paper for 14–28× measured there.
LUT uses more memory for smooth functions where polynomials encode
efficiently, but the speed/ops advantage can outweigh that.

See `results/H5_resource_bench/single_edge_profile.png`.


### M6a — On-device training: fp16 gradients cause zero degradation

Direct-LUT training survives MCU constraints intact:

| Regime | RAM | sine ratio | cusp ratio |
|---|---|---|---|
| SGD + fp16 grad (MCU) | **3.75 KB** | **4.5× vs poly** | **39.6× vs poly** |
| Adam + fp32 (server) | 8.75 KB | 130× vs poly | 150× vs poly |

B/C ratio (fp32 vs fp16 grad): **1.00** [0.75, 1.31] — CI includes 1.0.
Float16 gradient accumulation introduces no measurable accuracy penalty.
On-device λ₂ regularization is required (without it: 13–54× worse) and costs
only O(K·L) additions per step. Full training loop fits in **3.75 KB SRAM**
(Cortex-M4 class and above).


### M6b — SGD lr=2.0 outperforms Adam on non-smooth targets

Optimal on-device config: SGD fp16-grad, **lr=2.0**, λ₂=1.0, 3.75 KB RAM.

| Target | Adam (server) | SGD fp16 lr=2.0 (MCU) |
|---|---|---|
| sine (smooth) | 130× vs poly | 50× vs poly |
| cusp (non-smooth) | 150× vs poly | **404× vs poly** |
| saturating | 80× vs poly | **198× vs poly** |

On non-smooth targets, on-device SGD with fp16 gradients *beats Adam* by
2.5–2.7×, using half the RAM. Decay schedules (cosine, step) hurt — SGD
is not converged at 900 epochs; flat high lr is optimal.


### M7a — K=8, L=32 (288 B) is the Pareto-optimal config

H1 used K=16, L=32 (576 B). Sweeping K ∈ {8,16,32} × L ∈ {16,32,64} reveals:

| Config | Memory | sine | cusp | saturating |
|---|---|---|---|---|
| K=8, L=32 | **288 B** | **2764×** | 555× | **16428×** |
| K=16, L=32 (H1) | 576 B | 611× | **790×** | 942× |
| K=32, L=32 | 1152 B | 16× | 10× | 2× |

**K=8, L=32 dominates on 2 of 3 targets at half the memory.** Ratio decreases
monotonically with K: wider segments → larger interpolation error → more room to
compensate. At K=32 the advantage nearly vanishes; at L=64 it collapses for all K.

### M7b — Zero init works: no server required

Random and zero init are statistically indistinguishable from poly init
(95% CI includes 1.0 on all three targets). Full on-device pipeline:




### M7c — K=4, L=32 (144 B) Pareto-dominates on saturating targets

Extended sweep K ∈ {1,2,4,8} × L ∈ {8,16,32} reveals the full Pareto frontier:

| Memory | Config | sine | cusp | saturating |
|---|---|---|---|---|
| **144 B** | **K=4, L=32** | **1124×** | **126×** | **21012×** |
| 288 B | K=8, L=32 | 2764× | 555× | 16428× |
| 72 B  | K=2, L=32 | 54× | 107× | 2016× |

K=4, L=32 achieves **21012×** on saturating — **1.3× better than K=8, L=32 at half the memory**.
L=32 is a hard requirement: L=16 gives 20–40× lower ratios at all K values.
K=1 gives near-parity on smooth targets (polynomial already optimal for 1 segment).


### M7d — Coverage rule: K×L < n_train (1 data point per cell)

L sweep K ∈ {4,8} × L ∈ {8..128} reveals the crossover mechanism:

**K=4, L=64 (272B) achieves 131 348× on saturating** — project record.

| Config | Memory | sine | cusp | saturating |
|---|---|---|---|---|
| K=4, L=64 | **272B** | 7530× | 649× | **131 348×** |
| K=4, L=32 | 144B | 1157× | 129× | 22094× |
| K=8, L=48 | 416B | 3991× | 751× | 3305× |
| K=8, L=96 | 800B | **116×** | **44×** | **7×** ← collapse |

**Rule: K × L < n_train.** When total cells exceed training points, gradient
coverage collapses (< 1 point/cell) and training degrades to near-parity.
Practical formula: L_opt ≈ n_train / (2K).

## Installation

```bash
git clone <this-repo>
cd lut-kan-native
pip install -e .
```

Python ≥ 3.9, PyTorch ≥ 2.0, NumPy ≥ 1.22.

## Reproducing

```bash
# All tests (must pass before any claim)
pytest tests/ -v

# Quick run (reduced budgets, ~2 min)
python scripts/generate_all_figures.py --quick

# Full run as reported (~15 min CPU)
python scripts/generate_all_figures.py

# Individual experiments
python scripts/exp_H1_lambda_sweep.py       # best regularization? (heatmap)
python scripts/exp_H1_memory_sweep.py       # MSE vs L (crossover)
python scripts/exp_H1_targets.py            # per-target advantage (bars)
python scripts/exp_H2_effective_rank.py     # SVD of learned LUT
python scripts/exp_H4_multi_edge_2d.py      # 2D multi-edge honest negative
python scripts/exp_H5_resource_bench.py     # memory/ops/latency table
python scripts/diagnostic_training_dynamics.py  # why val-based selection matters
```

All results land in `results/<id>/{summary.json, *.png}`. JSONs are checked
in as ground truth; same seed → bit-identical output (verified by
`test_reproducibility.py`).

## Tests (18 total, all passing)

Tests verify:
- **Forward parity** (4 param.): PyTorch forward ≡ NumPy reference.
- **Gradcheck**: analytical gradient ≈ numerical.
- **Sparse gradient** (1 per sample touches exactly 2 LUT cells).
- **Gradient sum invariant** (weights sum to 1).
- **Multi-edge KAN forward parity** (4 config variations).
- **Multi-edge gradient flow** to both layers.
- **Parameter count correctness**.
- **OOB input handling**.
- **Seed reproducibility**: same seed → same weights; different seed → different weights.

## Repository structure

```
lut-kan-native/
├── src/lut_native/
│   ├── core.py              LUTEdge + numpy reference (matches main repo's half-open convention)
│   ├── kan2.py              LUTKAN2Layer: multi-edge two-layer KAN
│   ├── regularizers.py      first/second-diff + correctly-formulated boundary penalty
│   ├── baselines.py         Chebyshev fit + post-training LUT quantization (v2.1 pipeline)
│   ├── training.py          single-edge train loop
│   ├── training_kan2.py     multi-edge train loop
│   ├── targets.py           sine, cusp, saturating, piecewise_smooth, local_sharp, feynman_2d
│   ├── metrics.py           HF energy, effective rank, paired bootstrap CI
│   └── resources.py         memory bytes, op counts, CPU latency
├── scripts/
│   ├── exp_H1_lambda_sweep.py      H1: best regularization
│   ├── exp_H1_memory_sweep.py      H1+H3: MSE vs L (and crossover)
│   ├── exp_H1_targets.py           H1: across target types
│   ├── exp_H2_effective_rank.py    H2: SVD confound analysis
│   ├── exp_H4_multi_edge_2d.py     H4: multi-edge honest negative
│   ├── exp_H5_resource_bench.py    H5: resource accounting
│   ├── diagnostic_training_dynamics.py   why val-based selection matters
│   └── generate_all_figures.py     orchestrator
├── results/                 JSONs + PNGs as ground truth
├── tests/                   pytest; 18/18 passing
└── docs/METHODOLOGY.md      design decisions, all limitations, falsification criteria
```

## Key methodological choices (abridged; full detail in METHODOLOGY.md)

- **Val-based best-model selection** is critical. Without it, direct-LUT
  advantage is understated by ~16× due to late-training overfitting. See
  `results/diagnostics/training_dynamics.png`.

- **Paired bootstrap CI** on log-MSE ratio rather than raw std.

- **Deterministic test grid**: identical `x_test` across all seeds and
  regimes for apples-to-apples comparisons.

- **Half-open sampling**: LUT grid samples match main repo's C kernel byte-for-byte.

- **Corrected boundary penalty** that respects half-open sampling (naive form
  in the original proposal was wrong; empirically even the corrected one
  doesn't help because learned LUT is already effectively rank-2).

## Scope honestly stated

**What we claim:** direct-LUT training is beneficial for **single-edge,
univariate, memory-constrained deployment** when the alternative is to
post-training quantize a polynomial, on both smooth and non-smooth targets,
up to L~64 at K=16.

**What we do not claim:** (1) advantage in multi-edge KAN; (2) improvement
over float-polynomial where the polynomial fits in RAM; (3) general
expressive superiority of LUTs over polynomials; (4) MCU-measured latency
(not yet benchmarked; inferred from CPU numbers and v2.1 MCU measurements).

## Citing

If using this work, please also cite the main `lut-kan` repository whose
baseline pipeline is compared against.

```bibtex
@misc{lut-kan-native,
  title  = {lut-kan-native: direct LUT training for memory-constrained KAN deployment},
  author = {Kuznetsov, Oleksandr},
  year   = {2026},
  note   = {Research branch of lut-kan; single-edge positive, multi-edge negative}
}
```

## License

MIT

---

## M_HARDWARE — Real MCU validation (v0.12.0)

Hardware-validated on two devices (no FPU, soft-float float32):

| Device | Arch | MHz | SRAM |
|--------|------|-----|------|
| Arduino Mega 2560 | ATmega2560 AVR 8-bit | 16 | 8 KB |
| ESP32-C3 SuperMini | RISC-V RV32IMC | 160 | 400 KB |

### Cross-platform reproducibility

Ratios match PyTorch simulation to within 0.04% on all platforms:

| Config | PC sim | Arduino Mega | ESP32-C3 |
|--------|--------|-------------|----------|
| K=4, L=8  | 24.7x  | 24.7x  | 24.7x  |
| K=4, L=16 | 813.2x | 812.9x | 812.9x |
| K=4, L=32 | 20236.7x | 20237.8x | 20237.9x |

### Speed: 1.76x on AVR (vs 6x on x86)

On no-FPU MCU, soft-float call overhead dominates — the LUT speed
advantage requires hardware FPU. The accuracy advantage (20000x) is
architecture-invariant.

### SRAM: 2 KB for full training loop

K=4, L=32 training fits in 2 KB — Arduino Uno class hardware.

### Reproduce

```powershell
# Windows (auto-installs arduino-cli on first run)
.\hardware\flash.ps1 -Device mega    -Monitor
.\hardware\flash.ps1 -Device esp32c3 -Monitor
```

See `hardware/README.md` and `docs/M_HARDWARE_FINDINGS.md`.
