# lut-kan-native

Direct training of LUT activation tables for memory-constrained KAN deployment.

A bounded research question, answered rigorously:

> **On a fixed memory budget, is it better to train LUT values directly,
> or to fit a polynomial and quantize it?**

Eleven hypotheses tested across two development cycles (v14–v15). All findings
reported, including negative results, confounds found, and bugs discovered.

## Quick start

```bash
pip install -e ".[dev]"
python scripts/run_quick.py                     # smoke test ~1 min
python scripts/run_h10_depth_correct_kl.py      # H10 main experiment ~60 min
pytest tests/ -q                                # 38 tests
```

**Windows PowerShell:**
```powershell
$env:PYTHONPATH = "src"
python scripts\run_quick.py
.\run.ps1 experiment   # H10
```

See [RUNNING.md](RUNNING.md) for full instructions.

---

## Summary

### Where direct-LUT wins

**Single-edge `[1→1]` on 1D targets (H1):** up to **1107×** lower MSE than
post-training quantization at equal memory.

**Multilayer at correct K,L (H9):** K=2, L=8 `[2→4→1]` achieves MSE=3.9e-4
vs poly degree=15 MSE=1.5e-2 at matched budget. **40× win.**

**Sensor calibration (M_sensor):** NTC thermistor MAE ~0.17°C vs factory ~0.5°C,
with 50 calibration points, running on-device.

**On-device training (M6):** SGD + fp16 gradients, 2 KB SRAM, no accuracy penalty.

### Where direct-LUT loses or ties

- Large L (≥128): parity with polynomial; advantage disappears (H1b).
- K×L ≫ N/50: sparse gradient → constant predictor (H4, H8 original).
- Cross-variable 2D with poly degree ≥ 4: parity or poly wins (H10).

### The critical rule

```
N / (K × L) ≥ 50
```

Use `K=2` (not 4, not 16). Fewer segment boundaries = smoother gradient.
With N=800: K=2, L=8 (ratio=50). This single change fixed what looked like
an architectural failure of multi-layer LUT-KAN.

---

## Experiment results

### H1 — Direct-LUT dominates on 1D targets

Setup: single-edge `[1→1]`, K=16, N_train=400, 5 seeds, 95% bootstrap CI.
Comparison: direct-LUT training vs post-training quantization of polynomial.

| Target | Memory | Post-LUT MSE | Direct-LUT MSE | Ratio |
|---|---|---|---|---|
| `sin(2πx) + 0.5 sin(4πx)` | 576 B | 1.83e-4 | **4.23e-7** | **444× [357, 526]** |
| `\|x−0.3\| + 0.2 sin(3πx)` | 576 B | 3.23e-5 | 3.19e-7 | **143× [67, 307]** |
| `tanh(4x) + 0.15x` | 576 B | 1.43e-5 | 1.58e-8 | **1107× [635, 1995]** |

Advantage vanishes at L ≈ 64–128 (H1b): at L=8 direct-LUT is 37× better;
at L=128 ratio=0.84× [0.65, 1.21] (CI includes parity).

### H2 — Learned LUT is effectively rank-2 (confound disclosed)

SVD of `(LUT − per-segment-mean)` shows effective rank 2 across all regimes.
Direct-LUT is not using its nominal 512 DOF — it learns a 2-mode fit.
This reframes but does not invalidate H1. See `results/H2_effective_rank/`.

### H4 — Polynomial Pareto-dominates on 2D (in the K=16,L=32 regime)

Target: `f(x,y) = sin(πx) + 0.5 cos(2π·xy)`, two-layer `[2→h→1]`, K=16,
L=32. At every tested budget, poly-KAN MSE < direct-LUT MSE.

**Retrospective (H9):** K=16,L=32 has N/KL=1.6 — far below the N/KL≥50
requirement. H4 is valid for this K,L regime and still characterises the
behaviour of the "obvious" default configuration.

### H5 — Speed: LUT ~3.3× faster on MCU, ~6× on CPU

| Method | Bytes | Ops | CPU latency |
|---|---|---|---|
| poly deg=20 f32 | 84 | 80 float | 143 ns |
| LUT K=16, L=32, u8 | 576 | 13 float + 7 int | **16 ns** |
| LUT K=8, L=16, u8 | **160** | 13 float + 7 int | **16 ns** |

Gather+lerp is cache-friendly with no dependency chain. On MCU without FPU:
~3.3× (corrected; v13 docs erroneously reported 6.25× — latency unit was ns
not μs).

### M6a — On-device training: SGD + fp16 in 3.75 KB SRAM

| Regime | RAM | sine ratio | cusp ratio |
|---|---|---|---|
| SGD + fp16 grad (MCU sim) | **3.75 KB** | **4.5×** | **39.6×** |
| Adam + fp32 (server) | 8.75 KB | 130× | 150× |

fp16 gradient accumulation: no measurable accuracy penalty (CI includes 1.0).
On-device λ₂ regularisation required (without it: 13–54× worse) and costs
only O(K·L) additions per step.

### M6b — SGD lr=2.0 beats Adam on non-smooth targets

On-device optimal: SGD fp16, lr=2.0, λ₂=1.0, 3.75 KB RAM.

| Target | Adam (server) | SGD fp16 lr=2.0 (MCU) |
|---|---|---|
| sine (smooth) | 130× | 50× |
| cusp (non-smooth) | 150× | **404×** |
| saturating | 80× | **198×** |

On non-smooth targets SGD with fp16 *beats* Adam by 2.5–2.7× at half the RAM.

### M7a — K=8, L=32 (288 B) is Pareto-optimal

H1 used K=16, L=32 (576 B). Sweeping K ∈ {8,16,32} × L ∈ {16,32,64}:

| Config | Memory | sine | cusp | saturating |
|---|---|---|---|---|
| **K=8, L=32** | **288 B** | **2764×** | 555× | **16428×** |
| K=16, L=32 | 576 B | 611× | 790× | 942× |
| K=32, L=32 | 1152 B | 16× | 10× | 2× |

K=8,L=32 dominates on 2/3 targets at half the memory.

### M7c — K=4, L=32 (144 B) dominates on saturating

| Memory | Config | sine | cusp | saturating |
|---|---|---|---|---|
| **144 B** | **K=4, L=32** | **1124×** | **126×** | **21012×** |
| 288 B | K=8, L=32 | 2764× | 555× | 16428× |

L=32 is a hard requirement: L=16 gives 20–40× lower ratios at all K.

### M7d — Coverage rule: K×L < N_train

L sweep reveals the collapse threshold:

| Config | Memory | sine | cusp | saturating |
|---|---|---|---|---|
| K=4, L=64 | **272 B** | 7530× | 649× | **131 348×** ← project record |
| K=4, L=32 | 144 B | 1157× | 129× | 22094× |
| K=8, L=96 | 800 B | 116× | 44× | **7×** ← collapse |

Rule: **K×L < N_train** (≥1 point/cell absolute minimum).
Practical formula: `L_opt ≈ N / (2·K)`.

The v15 multilayer experiments refined this to `N/(K×L) ≥ 50`
(see H9 below).

### M_sensor — Sensor calibration study

8 NTC thermistor units, 50 calibration points, K=1, L=32, on-device.

| Method | Mean MAE (°C) |
|---|---|
| Factory datasheet | ~0.50 |
| One-point offset | ~0.28 |
| Polynomial (3-param) | ~0.28 |
| Steinhart-Hart (noisy, fair) | ~0.04 |
| **LUT-KAN K=1,L=32** | **~0.17** |

LUT halves factory error. SHH wins when the parametric form matches physics
exactly. LUT preferred for sensors with uncertain or non-standard physics
(MQ gas, pH, LDR, thermocouples). 6 sensor types characterised; see
`results/sensor_calib_v14/regime_map.png`.

### M_hardware — Real MCU validation (Arduino Mega, ESP32-C3)

| Device | Arch | MHz | SRAM |
|---|---|---|---|
| Arduino Mega 2560 | ATmega2560, AVR 8-bit | 16 | 8 KB |
| ESP32-C3 | RISC-V RV32IMC | 160 | 400 KB |

Cross-platform ratios match PyTorch simulation to within **0.04%**:

| Config | PC sim | Mega | ESP32-C3 |
|---|---|---|---|
| K=4, L=8 | 24.7× | 24.7× | 24.7× |
| K=4, L=16 | 813× | 812.9× | 812.9× |
| K=4, L=32 | 20237× | 20238× | 20238× |

Speed on no-FPU AVR: **1.76× vs poly** (vs 6× on x86). Without hardware
FPU, soft-float call overhead dominates — the accuracy advantage (20000×)
is architecture-invariant; the speed advantage requires FPU.

Training loop fits in **2 KB SRAM** — Arduino Uno class.

```powershell
.\hardware\flash.ps1 -Device mega    -Monitor
.\hardware\flash.ps1 -Device esp32c3 -Monitor
```

---

## v15: Multilayer stacking (H8–H10)

### Three bugs found in v15

All previous multilayer experiments (H4, H8 original) were measuring a broken
configuration. Three bugs caused this:

**Bug 1 — Dead initialisation.**
`init_noise_std=0.05` activates 4/16 LUT segments. Model predicts constant
(MSE ≈ Var(y) = 0.66) throughout training. Fix: `cheby_init(scale=1.5)`.

```python
# Bad: 4/16 segments active, model predicts constant
StackTrainConfig(cheby_init=False, init_noise_std=0.05)

# Good: 16/16 segments active from step 0
StackTrainConfig(cheby_init=True, cheby_scale=1.5)
```

**Bug 2 — Norm collapse.**
Trainable inter-layer norms collapse after ~50 epochs via gradient-through-clamp
annihilation. Three gradient-based fixes fail (soft histogram entropy, out-of-domain
penalty, quantile matching). Fix: freeze norms + EMA batch tracking.

```python
StackTrainConfig(freeze_norms=True, ema_alpha=0.01)
```

**Bug 3 — Wrong K,L (most impactful).**
K=16, L=32 → N/KL = 1.6 examples/cell → sparse gradient → MSE ≈ Var(y).
Fix: K=2, L=8 → N/KL = 50 → MSE = 3.9e-4. **1200× improvement.**

### H9 — Fair budget comparison

| K | L | N/KL | MSE | vs poly matched budget |
|---|---|---|---|---|
| 2 | 8 | 50 | **3.9e-4** | **40× better** than poly_d15 |
| 2 | 4 | 100 | 6.7e-4 | — |
| 4 | 4 | 50 | 5.6e-3 | 10× worse than K=2,L=8 |
| 16 | 32 | 1.6 | 0.47 | constant predictor |

K=2 beats K=4 at equal cell count: fewer segment boundaries = smoother landscape.

### H10 — Depth vs width at correct K,L

**Depth (K=2,L=8, hidden=4, 3 seeds):**

| Architecture | Median MSE | Max/min ratio |
|---|---|---|
| `[2→4→1]` | **5.5e-4** | 8× |
| `[2→4→4→1]` | 1.0e-3 | 64× |
| `[2→4→4→4→1]` | 9.5e-3 | 5× |

Depth increases both median MSE and variance on `feynman_2d`.

**Width (1-layer, K=2,L=8, 3 seeds):**

| Architecture | Median MSE | std/mean |
|---|---|---|
| `[2→4→1]` | 5.5e-4 | 0.95 |
| `[2→16→1]` | 1.1e-3 | **0.08** |

Width reduces variance 12× more than it changes median MSE.

**Most stable multi-layer result:** K=2,L=4 `[2→4→4→1]` —
median=5.6e-4, max/min=2×. Smaller L reduces inter-layer tracking sensitivity.

**Poly crossover (seed=0):** LUT K=2,L=8 at parity with poly degree=4–5;
poly degree=8 MSE=3.3e-4 beats LUT (single seed, unconfirmed).

---

## Recommended configuration

```python
from lut_native import LUTKANStack, StackTrainConfig, train_lut_stack

# Rule: N / (K*L) >= 50. With N=800: K=2, L=8.
model = LUTKANStack(dims=[2, 4, 1], K=2, L=8)

cfg = StackTrainConfig(
    cheby_init=True,      # required
    cheby_scale=1.5,
    freeze_norms=True,    # required for multi-layer
    ema_alpha=0.01,
    lr=1e-2,
    epochs=3000,
    batch_size=64,
)

result = train_lut_stack(model, x_tr, y_tr, x_val, y_val, x_te, y_te, cfg)
```

For single-edge 1D (e.g. sensor calibration):
```python
from lut_native import LUTEdge, train_lut
# L_opt ≈ N_cal / (2*K); K=4 or K=8 depending on target smoothness
```

---

## Repository layout

```
src/lut_native/
  core.py              LUT forward, numpy reference
  kan2.py              LUTKAN2Layer (2-layer, tanh squash)
  kan_stack.py         N-layer stack: LUTBlock, LUTInterLayerNorm, LUTKANStack
  training_stack.py    Training: cheby_init, freeze_norms, EMA, patience, block_lr
  coverage_stack.py    Per-block / per-norm coverage diagnostics
  poly_kan2.py         Chebyshev polynomial baseline
  sensors.py           NTC, MQ, pH, thermocouple, LDR sensor models
  targets.py           Synthetic regression targets (1D and 2D)
  metrics.py           Bootstrap CI, paired comparisons
  resources.py         Memory and operation counting
  training.py          Single-edge training loop
  training_kan2.py     Two-layer KAN training loop
  baselines.py         Polynomial fit + post-training LUT quantization
scripts/               One script per experiment
tests/                 38 unit and integration tests
results/               JSON + PNG outputs (JSON checked in as ground truth)
docs/                  METHODOLOGY.md, per-experiment findings
hardware/              Arduino / PlatformIO firmware for on-device tests
```

---

## Open questions

1. **Depth on other targets.** H10 used `feynman_2d` which has a cross-variable
   `x·y` product term. A separable 2D target (`g(x) + h(y)`) may respond
   differently to depth. Not tested.

2. **High seed variance in multi-layer.** K=2,L=8 `[2→4→4→1]` shows 64× max/min
   ratio across seeds. K=2,L=4 same architecture: 2×. Root cause not identified.

3. **Poly crossover (H10 Part D).** Measured at one seed. Multi-seed confirmation
   needed to determine statistical parity point.

4. **MCU speed with FPU.** Hardware tests on AVR (no FPU) show 1.76× speed
   advantage. FPU-equipped MCU expected to match the 6× CPU result. Not measured.

---

## Scope

**We claim:** direct-LUT is beneficial for single-edge univariate deployment
at small-to-medium L, when the alternative is post-training quantization.
Also for sensor calibration when physics is uncertain.

**We do not claim:** advantage in multi-edge KAN at large K,L; superiority
over polynomial where polynomial fits in RAM; general expressiveness advantage.

---

## Citation

```bibtex
@software{lut_kan_native,
  title  = {lut-kan-native: direct LUT training for memory-constrained KAN},
  year   = {2026},
  url    = {https://github.com/<your-username>/lut-kan-native}
}
```

## License

MIT
