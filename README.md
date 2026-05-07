# lut-kan-native

Direct training of lookup-table (LUT) activations for KAN deployment on
memory-constrained microcontrollers.

**Three proven claims, each independently validated:**

1. **Gradient stability** — LUT trains stably where gradient-trained poly-KAN
   diverges. At degree ≥ 15 in multi-layer stacks, Chebyshev backward pass
   chains two high-degree recurrences → exploding/vanishing gradients. LUT
   backward is a local scatter-add with bounded gradients. Result: 23/23
   configurations win against matched gradient-trained poly-KAN.

2. **Inference speed on MCU** — LUT forward pass is O(1): one index lookup
   per edge, ~31 cycles on Cortex-M4 (no FPU). Chebyshev d=31 ≈ 2 124 cycles.
   B-spline cubic ≈ 256 cycles. Speed advantage kicks in at K×L ≥ 32.

3. **On-device training** — Full SGD loop with fp16 gradient accumulation fits
   in ~3.75 KB SRAM on Cortex-M4 / ESP32-C3 (RISC-V, no FPU). Validated on
   hardware: simulation predictions matched on-device results within 1%.
   First KAN-like architecture trainable on a microcontroller.

**What LUT-KAN does NOT claim to win:**

- Single-edge [1→1] vs closed-form least-squares poly: poly LS is more
  accurate at the same parameter count (no gradient instability in 1-layer).
- Smooth targets with small K×L: polynomial is a more efficient representation.
- K×L ≥ n_train: gradient signal collapses (see coverage density rule below).

---

## Quick start

```bash
pip install -e ".[dev]"
python scripts/run_quick.py          # smoke test, ~1 min
pytest tests/ -q                     # 95 tests
```

Windows:
```powershell
$env:PYTHONPATH = "src"
python scripts\run_quick.py
```

---

## Coverage density rule

The single most useful result from this project. When training LUT-KAN:

```
K × L  <  n_train          # necessary condition for gradient coverage
L_opt  ≈  n_train / (2K)   # practical recommendation (density ≥ 2 pts/cell)
```

If K×L ≥ n_train, fewer than 1 training point per LUT cell on average.
Cells without gradient signal drift toward λ₂-regularised default (flat),
and the model degrades to a polynomial-like approximation — no advantage
over post-training quantization.

Example (n_train=500):
- K=4: L_max=124 → use **L=64** (density 1.95 pts/cell)
- K=8: L_max=62  → use **L=32** (density 1.95 pts/cell)

---

## Library API

Two public model families:

### Two-layer (fast, well-studied)
```python
from lut_native.kan2 import LUTKan2
from lut_native.training import train_kan2
from lut_native.poly_kan2 import PolyKan2        # matched polynomial baseline
```

`LUTKan2(in_dim, hidden_dim, out_dim, K, L)` — architecture `[in→hidden→out]`.

### N-layer stack (current main architecture)
```python
from lut_native.kan_stack import LUTKANStack
from lut_native.training_stack import train_lut_stack, StackTrainConfig
from lut_native.poly_kan_stack import PolyKANStack   # matched baseline
```

`LUTKANStack(dims=[in, h1, h2, ..., out], K, L)` — arbitrary depth.

### Key training parameters
```python
cfg = StackTrainConfig(
    lr          = 2.0,        # SGD learning rate (validated on-device)
    fp16_grad   = True,       # fp16 gradient accumulation (no accuracy loss)
    lambda2     = 0.01,       # L2 regularization on LUT values
    cheby_init  = True,       # initialize cells with Chebyshev basis (recommended)
    freeze_norms= True,       # prevent norm collapse in deep stacks
    patience    = 50,         # early stopping
)
```

### Coverage diagnostic
```python
from lut_native.coverage_stack import compute_stack_coverage
report = compute_stack_coverage(model, x_train)
# Returns per-block visited fraction + uniformity score
```

---

## Recommended scripts

| Script | Purpose | Time |
|--------|---------|------|
| `scripts/run_quick.py` | Smoke test — verifies all three claims | ~1 min |
| `scripts/run_h10_depth_correct_kl.py` | Depth/width sweep (H10) | ~60 min |
| `scripts/run_h9_fair_comparison.py` | Budget comparison vs poly-KAN | ~30 min |
| `scripts/sweep_full.py` | Comprehensive LUT vs poly-KAN benchmark | ~2–4 h |
| `scripts/m6a_ondevice_sim.py` | On-device SGD simulation | ~10 min |
| `scripts/m7d_l_sweep.py` | L sweep demonstrating coverage rule | ~20 min |

All other scripts in `scripts/` are historical experiment reproductions
(M2–M5, H1–H8 series). They run but are not needed for normal use.

---

## Hardware

Firmware for ESP32-C3 SuperMini (RISC-V, no FPU) is in `hardware/`.
See `hardware/README.md` for flashing instructions.

Validated configuration: K=4, L=32 → training time ~11 s/config on hardware.
Simulation predictions matched hardware results within 1% across all tested
configurations.

---

## Key experimental results

Full findings in `docs/`. Summary:

| Claim | Result | Source |
|-------|--------|--------|
| Gradient stability | 23/23 configs, LUT wins vs grad-trained poly | M9a |
| Best accuracy ratio | 130 428× (saturating, K=2,L=32, single edge vs diverging poly) | M9a deep |
| Peak result (IDS) | K=4,L=64, ratio=131 348× on saturating target | M7d |
| On-device training | SGD fp16 Δ vs Adam = 0 (CI includes 1.0, 5 seeds) | M7b |
| Coverage rule | K×L < n_train predicts collapse, 100% accuracy | M7d |
| Hardware validation | sim/hw delta < 1% on ESP32-C3 across all configs | M_HARDWARE |
| Inference speed | 31 cycles/edge (O(1) vs O(degree) for poly) | M9b |

---

## What to keep in mind before citing

The 130 000× accuracy ratios are on 1D synthetic targets (saturating, cusp)
comparing LUT gradient-descent vs **gradient-trained** polynomial of matched
degree. Against closed-form least-squares polynomial, LUT loses at the same
parameter count on smooth targets. The correct framing is:

> "LUT trains stably in regimes where gradient-trained poly-KAN diverges,
>  and achieves 8–68× faster inference on MCU at equal accuracy."

See `docs/M9b_FINDINGS.md` for the full methodology audit.

---

## Tests

```
pytest tests/ -q     # 95 tests, ~3 min
```

Tests are organized by module/experiment:
- `test_coverage.py`, `test_stack.py` — core architecture
- `test_kl_sweep.py`, `test_l_sweep.py`, `test_lr_sweep.py` — coverage rule
- `test_ondevice_sim.py`, `test_init_ablation.py` — on-device claims
- `test_gradcheck.py`, `test_forward_matches_numpy.py` — numerical correctness

---

## Citation

```bibtex
@software{lut_kan_native,
  title  = {lut-kan-native: Direct LUT Training for KAN on Microcontrollers},
  year   = {2026},
  url    = {https://github.com/KuznetsovKarazin/lut-kan-native},
  note   = {v0.16.0}
}
```

---

## License

MIT
