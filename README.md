# lut-kan-native

Direct training of lookup-table (LUT) activations for KAN deployment on
memory-constrained microcontrollers.

## Research components

- **Sensor-calibration implementation study:** [reproduction entry point](experiments/sensor_recovery/README.md)
  and [IEEE Sensors Journal manuscript](manuscripts/ieee_sensors_tradeoff/README.md).
  Controlled synthetic accuracy is paired with seven-model execution measurements
  on Mega2560, ESP32-C3 and ESP32-S3. The recorded builds show conditional
  accuracy, payload and timing trade-offs; no universal LUT advantage is claimed.
- **LUT-native KAN training:** `src/lut_native`, `scripts/` and the other result
  directories contain separate training/deployment experiments. Their conditions
  and historical claims must not be transferred to the sensor benchmark.

The earlier multi-sensor v14 figures are historical development artifacts;
see [their status](docs/SENSOR_STUDY_HISTORY.md). Use `sensor_recovery` for the
current calibration paper. Public raw evidence is retained, including unsuccessful
captures, while internal correspondence is excluded from the publication tree.

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

Historical experimental heuristic for some LUT-KAN configurations:

```
K × L  <  n_train          # heuristic only, not a necessary condition
L_opt  ≈  n_train / (2K)   # historical rule of thumb, not a general optimum
```

Parameter count alone neither guarantees nor rules out sufficient coverage.
Each linear interpolation sample touches two nodes; regularization couples
nodes and can produce useful solutions even when L exceeds n_train. Inspect
the design matrix and compare errors on held-out data. No general loss of
advantage over post-training quantization follows from this inequality.

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
