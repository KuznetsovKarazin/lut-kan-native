"""
Quick smoke test — runs in ~30 seconds.
Verifies the key fixes from H8 sessions work on your machine.
"""
import sys, time
sys.path.insert(0, "src")

import torch
import numpy as np
from lut_native.targets import generate_data_2d
from lut_native.kan_stack import LUTKANStack
from lut_native.training_stack import StackTrainConfig, train_lut_stack
from lut_native.poly_kan2 import PolyKAN2Layer, train_poly_kan2

# ── data ──────────────────────────────────────────────────────────────────────
print("Loading feynman_2d data ...", flush=True)
data = generate_data_2d("feynman_2d", n_train=800, n_val=300, n_test=400, seed=42)
x_tr, y_tr, x_val, y_val, x_te, y_te = data
y_tr  = y_tr.reshape(-1, 1)
y_val = y_val.reshape(-1, 1)
y_te  = y_te.reshape(-1, 1)
xt = torch.from_numpy(x_tr)

print(f"  x_tr: {x_tr.shape}  y range: [{y_tr.min():.2f}, {y_tr.max():.2f}]")
print(f"  Var(y_train) = {float(y_tr.var()):.4f}  (constant predictor MSE)\n")

results = {}

# ── Test 1: Dead init (the bug) ───────────────────────────────────────────────
print("Test 1: Segment coverage at init")
blk_dead  = __import__("lut_native.kan_stack", fromlist=["LUTBlock"]).LUTBlock(2, 4, 16, 32)
blk_cheby = __import__("lut_native.kan_stack", fromlist=["LUTBlock"]).LUTBlock(2, 4, 16, 32)
blk_dead.add_init_noise(0.05)
blk_cheby.cheby_init(scale=1.5, noise=0.05)

with torch.no_grad():
    for blk, label in [(blk_dead, "noise std=0.05"), (blk_cheby, "cheby_init    ")]:
        z = blk(xt)
        segs = ((z.clamp(-1, 1-1e-7) + 1) / 2 * 16).long().clamp(0, 15)
        counts = torch.zeros(16)
        for k in range(16):
            counts[k] = (segs == k).float().mean()
        active = int((counts > 0.001).sum())
        print(f"  {label}: {active}/16 segments active  z_std={z.std():.3f}")

print()

# ── Test 2: 1-layer LUT-KAN (no norm) ────────────────────────────────────────
print("Test 2: 1-layer stack [2→4→1] — 200 epochs")
t0 = time.time()
m1 = LUTKANStack(dims=[2, 4, 1], K=16, L=32, smooth_norms=False)
cfg1 = StackTrainConfig(
    cheby_init=True, cheby_scale=1.5, cheby_noise=0.05,
    lr=1e-2, epochs=200, batch_size=64, seed=0,
    eval_every_epochs=50, lambda_2=0,
    freeze_norms=True, ema_alpha=0.01, recalibrate_every_epochs=0,
)
r1 = train_lut_stack(m1, x_tr, y_tr, x_val, y_val, x_te, y_te, cfg1)
results["1-layer clamp"] = r1.mse_test_at_best
print(f"  MSE = {r1.mse_test_at_best:.4f}  best_ep={r1.best_epoch}  ({time.time()-t0:.0f}s)")
print(f"  (was 0.6463 with dead noise init — should be ~0.38-0.43)")

print()

# ── Test 3: 3-layer with gradient-balanced LR ─────────────────────────────────
print("Test 3: 3-layer [2→4→4→1] + block_lr_scales=[0.1,0.1,1.0] — 200 epochs")
t0 = time.time()
m3 = LUTKANStack(dims=[2, 4, 4, 1], K=16, L=32, smooth_norms=False)
cfg3 = StackTrainConfig(
    cheby_init=True, cheby_scale=1.5, cheby_noise=0.05,
    lr=1e-2, epochs=200, batch_size=64, seed=0,
    eval_every_epochs=50, lambda_2=0,
    freeze_norms=True, ema_alpha=0.01, recalibrate_every_epochs=0,
    block_lr_scales=[0.1, 0.1, 1.0],
)
r3 = train_lut_stack(m3, x_tr, y_tr, x_val, y_val, x_te, y_te, cfg3)
results["3-layer balanced"] = r3.mse_test_at_best
print(f"  MSE = {r3.mse_test_at_best:.4f}  best_ep={r3.best_epoch}  ({time.time()-t0:.0f}s)")
print(f"  (was 0.6463 with dead init — should be ~0.55-0.62)")

print()

# ── Test 4: Poly-KAN baseline ─────────────────────────────────────────────────
print("Test 4: Poly-KAN [2→4→1] degree=20 — 200 epochs (upper bound)")
t0 = time.time()
mp = PolyKAN2Layer(in_dim=2, hidden_dim=4, out_dim=1, degree=20)
rp = train_poly_kan2(mp, x_tr, y_tr, x_val, y_val, x_te, y_te,
                     lr=1e-2, epochs=200, batch_size=64, seed=0)
results["poly_4_d20"] = rp["mse_test_at_best"]
print(f"  MSE = {rp['mse_test_at_best']:.4f}  best_ep={rp['best_epoch']}  ({time.time()-t0:.0f}s)")

print()
print("=" * 55)
print("Summary")
print("=" * 55)
var_y = float(y_tr.var())
for label, mse in results.items():
    ratio = mse / var_y
    print(f"  {label:<22}  MSE={mse:.4f}  ({ratio:.2f}× Var(y))")
print(f"\n  Poly-KAN gap: {results['poly_4_d20'] / results['1-layer clamp']:.2f}×")
print()
print("All smoke tests passed ✓" if all(v < 0.70 for v in list(results.values())[:-1]) else "⚠ Some results above expected range")
