# Changelog

## v0.15.0 (2026-05-xx) — Multilayer stacking: bugs, fixes, and correct regime

### New modules

**`src/lut_native/kan_stack.py`** — N-layer LUT-KAN stack
- `LUTBlock` — single LUT layer as a standalone `nn.Module`
- `LUTInterLayerNorm` — learnable per-channel affine norm with:
  - `smooth=True` (default): tanh squash, differentiable everywhere
  - `smooth=False` (legacy): hard clamp, zero gradient at boundary
  - `calibrate(z, plo, phi)`: percentile-based one-shot init
  - `ema_update(z, alpha)`: continuous EMA batch tracking (new in v15)
- `LUTKANStack` — N-layer stack with `cheby_init`, `calibrate`, snapshots
- `coverage_entropy_loss` — soft histogram entropy (deprecated: zero gradient after collapse)
- `norm_coverage_loss` — pre-clamp version (also unreliable after collapse; use EMA instead)
- `_lut_fwd_compat` — numpy forward matching PyTorch clip convention exactly

**`src/lut_native/training_stack.py`** — Training loop
- `StackTrainConfig` with all new fields:
  - `cheby_init`, `cheby_scale`, `cheby_noise` — Chebyshev initialisation
  - `freeze_norms`, `ema_alpha` — norm management
  - `block_lr_scales` — per-block LR scaling
  - `smooth_norms` — tanh vs clamp (passed to LUTKANStack)
  - `patience` — early stopping
- `auto_block_lr_scales(model, x, y)` — auto-compute scales from gradient ratio
- `train_lut_stack(..., patience)` — main training loop

**`src/lut_native/coverage_stack.py`** — Diagnostics
- `NormCoverageReport`, `StackCoverageReport`
- `compute_stack_coverage(model, x)` — per-block visited fraction + per-norm uniformity

### New scripts

- `scripts/run_quick.py` — smoke test, ~60 sec, verifies all three fixes
- `scripts/run_h8_corrected.py` — H8 with cheby_init, 5 seeds (K=16,L=32 still wrong)
- `scripts/run_h9_fair_comparison.py` — budget comparison: K×L sweep vs poly degrees
- `scripts/run_h10_depth_correct_kl.py` — definitive depth/width test at K=2,L=8

### Three bugs found and fixed

**Bug 1 — Dead initialisation** (affects all previous multilayer runs).
`init_noise_std=0.05` produces z_std≈0.06 → 4/16 segments active → constant predictor.
Fix: `cheby_init=True` initialises cells with Chebyshev T_0…T_{dim−1}, z_std≈0.94.

**Bug 2 — Norm collapse** (affects multilayer with trainable norms).
Adam shrinks `scale` until all activations clamp to ±1; gradient through clamp=0;
norms permanently frozen. Gradient-based losses (entropy, penalty, quantile) fail
because sigmoid gates saturate. Fix: `freeze_norms=True` + `ema_alpha=0.01`.

**Bug 3 — Wrong K,L** (affects H4 and H8, the most important discovery).
K=16, L=32 → N/KL = 800/512 = 1.6 examples/cell → sparse gradient → MSE≈Var(y).
Fix: K=2, L=8 → N/KL = 50 → MSE=3.9e-4. 1200× improvement from choosing K,L correctly.

### Key experimental results (v15)

- **H9 (fair comparison):** LUT K=2,L=8 MSE=3.9e-4 vs poly_d15 MSE=1.5e-2 (matched budget, 40× win).
- **H10 depth sweep:** Depth increases median MSE and variance on feynman_2d.
  1-layer `[2,4,1]` median=5.5e-4 beats 2-layer median=1.0e-3.
- **H10 width sweep:** Width reduces variance. `[2,16,1]` std/mean=0.08 vs `[2,4,1]` std/mean=0.95.
- **H10 K,L depth:** K=2,L=4 `[2,4,4,1]` most stable (median=5.6e-4, max/min=2×).
- **H10 poly crossover:** Parity at poly degree 4–5 (seed=0 only; needs confirmation).

### Other changes

- `LUTInterLayerNorm.smooth=True` default (tanh, gradient ratio 2× vs 3609× for K=16,L=32)
- `LUTKANStack.cheby_init(scale, noise)` propagates to all blocks
- `train_lut_stack` gains `patience` parameter for early stopping
- Tests: 38 tests, including `TestChebyInit` class verifying dead-segment fix
- `RUNNING.md` — Windows PowerShell instructions
- `run.ps1` — convenience runner for PowerShell

### Version

`__version__`: `0.2.0` → `0.3.0`
`pyproject.toml`: `0.14.0` → `0.15.0`

---

## v0.14.0 (2026-04-25)

### New sensors (`src/lut_native/sensors.py`)
- `TypeKThermocoupleUnit` — Type K thermocouple (NIST ITS-90, full polynomial + exponential correction)
- `LDRSensorUnit` — GL55-series photoresistor (power-law log characteristic)

### Revised baselines
- `fit_shh_baseline(..., use_noisy=True/False)` — fair Steinhart-Hart comparison:
  `use_noisy=False` = SHH from 3 ideal points (literature baseline);
  `use_noisy=True` = SHH from same 50 noisy points as LUT (apples-to-apples).
- `fit_onepoint_offset_baseline()` — factory poly + single-point offset (practical minimum)

### Sensor calibration study v14
- `scripts/exp_sensor_calib_v14.py`: 4 experiments × 5 sensor types × 8 units
- NTC extended (n=8): LUT MAE ~0.17°C vs factory ~0.5°C, poly ~0.28°C
- Regime map: LUT preferred when physical model uncertain; SHH preferred when parametric form exact
- Coverage rule confirmed: K×L < N_cal / 10

### Bug fix (critical)
Latency unit error in docs and paper: reported as ns, should be μs.
AVR: ~80 μs/call; ESP32-C3: ~5 μs/call; speedup 3.3× (not 6.25×).
Fixed in `docs/M_HARDWARE_FINDINGS.md` and `README.md`.

---

## v0.13.0 (2026-04-24)
- Multi-sensor calibration study (NTC, MQ, pH)
- Coverage rule K×L < n_train validated
- On-device training: SGD + fp16, zero init

## v0.12.0 (2026-04-21)
- On-device training experiments (M6a)
- Hardware validation: AVR, RISC-V
- L-sweep analysis (M7d)

## v0.11.0 (2026-04-20)
- KAN2 multi-edge experiments (H4)
- Effective rank analysis (H2)
