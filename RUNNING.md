# Running the experiments

## Prerequisites

```bash
pip install -e ".[dev]"
# or
pip install torch numpy scipy matplotlib pytest
```

Python 3.9+. No GPU required; all experiments run on CPU.

---

## Windows PowerShell

PowerShell does not support `VAR=value cmd` syntax.
Use `$env:` instead:

```powershell
cd lut-kan-native
$env:PYTHONPATH = "src"

python scripts\run_quick.py                    # ~1 min smoke test
python scripts\run_h10_depth_correct_kl.py    # ~60 min, H10 main experiment
python -m pytest tests\ -q                     # test suite
```

To avoid setting `$env:PYTHONPATH` every session, use the helper:
```powershell
.\run.ps1 quick        # smoke test
.\run.ps1 experiment   # H10
.\run.ps1 test         # pytest
```

---

## Linux / macOS

```bash
cd lut-kan-native
PYTHONPATH=src python3 scripts/run_quick.py
PYTHONPATH=src python3 scripts/run_h10_depth_correct_kl.py
PYTHONPATH=src python3 -m pytest tests/ -q
```

---

## Script descriptions

| Script | Runtime | Description |
|---|---|---|
| `run_quick.py` | ~1 min | Smoke test: verifies cheby_init fix, 1-layer, 3-layer, poly baseline |
| `run_h8_corrected.py` | ~30 min | H8 with cheby_init fix, K=16,L=32 (still suboptimal K,L) |
| `run_h9_fair_comparison.py` | ~60 min | K,L sweep: finds K=2,L=8 optimal |
| `run_h10_depth_correct_kl.py` | ~60 min | Depth, width, K,L×depth, poly crossover at K=2,L=8 |
| `exp_sensor_calib_v14.py` | ~20 min | Full sensor calibration study (NTC, thermocouple, LDR, MQ, pH) |
| `exp_H1_*.py` | ~30 min each | H1 single-edge experiments |
| `exp_H4_*.py` | ~45 min | H4 multi-edge 2D (K=16,L=32, original regime) |

---

## Key API

```python
import sys; sys.path.insert(0, "src")
from lut_native import LUTKANStack, StackTrainConfig, train_lut_stack

# Correct K,L: N / (K*L) >= 50
# N=800 → K=2, L=8 (ratio 50) or K=2, L=4 (ratio 100, more stable)
model = LUTKANStack(dims=[2, 4, 1], K=2, L=8)

cfg = StackTrainConfig(
    cheby_init=True,      # required: Chebyshev init prevents dead segments
    cheby_scale=1.5,
    freeze_norms=True,    # required: prevents norm collapse in multi-layer
    ema_alpha=0.01,       # continuous batch-level norm tracking
    lr=1e-2,
    epochs=3000,
    batch_size=64,
)

result = train_lut_stack(model, x_tr, y_tr, x_val, y_val, x_te, y_te, cfg)
print(f"Test MSE: {result.mse_test_at_best:.2e}  best_epoch={result.best_epoch}")
```

### Coverage diagnostics (multi-layer)

```python
from lut_native import compute_stack_coverage

report = compute_stack_coverage(model, x_train)
print(report)   # per-block visited_fraction, per-norm uniformity
```

### Sensor calibration

```python
from lut_native.sensors import NTCSensorUnit
from lut_native.training import train_lut

sensor = NTCSensorUnit(B=3950, R25=10000)
lut = train_lut(sensor, n_cal=50, K=1, L=32)
print(f"MAE: {lut.evaluate_mae():.3f} °C")
```
