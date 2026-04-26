"""
H9: Fair comparison — LUT vs poly at MATCHED parameter budgets.

The H8 comparison was unfair:
  poly_4_d20:    252 float32 params  (21 per edge × 12 edges)
  stack_2x4 LUT: 6144 float32 params (512 per edge × 12 edges)

Poly uses 24× FEWER parameters and wins by 33×.
This means either:
  (a) poly is architecturally superior at ALL budgets, OR
  (b) LUT at small budgets can compete with poly at small budgets

This experiment tests (b) by sweeping over matched (K*L ≈ degree) budgets.

Architectures:
  LUT  K=1,L=4  (~4 params/edge):  vs  poly degree=3   (4 params/edge)
  LUT  K=2,L=4  (~8 params/edge):  vs  poly degree=7   (8 params/edge)
  LUT  K=4,L=8  (~32 params/edge): vs  poly degree=31  (32 params/edge)
  LUT  K=8,L=16 (~128 params/edge):vs  poly degree=127 (128 params/edge)  ← overfit?
  LUT  K=16,L=32 (~512/edge):      vs  poly degree=511 (512 params/edge)  ← overfit

Also test: what is the MINIMUM poly degree that beats best LUT config?

Expected finding: poly wins at ALL matched budgets because of global
gradient vs local gradient. But at what degree do they reach parity?
"""
import sys, time, json, os
sys.path.insert(0, "src")

import torch
import numpy as np
from lut_native.targets import generate_data_2d
from lut_native.kan_stack import LUTKANStack
from lut_native.poly_kan2 import PolyKAN2Layer, train_poly_kan2
from lut_native.training_stack import StackTrainConfig, train_lut_stack

OUT_DIR = os.path.join("results", "H9_fair_comparison")
os.makedirs(OUT_DIR, exist_ok=True)
EPOCHS = 2000
SEEDS  = [0, 1, 2]

data = generate_data_2d("feynman_2d", n_train=800, n_val=300, n_test=400, seed=42)
x_tr, y_tr, x_val, y_val, x_te, y_te = data
y_tr=y_tr.reshape(-1,1); y_val=y_val.reshape(-1,1); y_te=y_te.reshape(-1,1)

# Budget table: (label, K, L, poly_degree)
# params_per_edge = K*L for LUT, degree+1 for poly
BUDGETS = [
    # label            K    L    poly_degree   note
    ("tiny",           1,   4,   3,            "4 params/edge"),
    ("small",          2,   8,   15,           "16 params/edge"),
    ("medium",         4,   16,  63,           "64 params/edge"),
    ("large",          8,   32,  255,          "256 params/edge"),
    ("H8_lut_budget",  16,  32,  511,          "512 params/edge — original H8 LUT budget"),
    ("poly_d20_budget",16,  32,  20,           "fair LUT budget vs poly_d20 (original H8 poly)"),
]

# Also: fixed best-LUT config, sweep poly degree to find crossover
POLY_SWEEP_DEGREES = [1, 2, 3, 4, 5, 7, 10, 15, 20]

all_results = []


def run_lut(K, L, seed, ep=EPOCHS):
    m = LUTKANStack(dims=[2, 4, 1], K=K, L=L, smooth_norms=False)
    cfg = StackTrainConfig(
        cheby_init=True, cheby_scale=1.5, cheby_noise=0.05,
        lr=1e-2, epochs=ep, batch_size=64, seed=seed,
        eval_every_epochs=100, lambda_2=0,
        freeze_norms=True, ema_alpha=0.01, recalibrate_every_epochs=0,
    )
    r = train_lut_stack(m, x_tr, y_tr, x_val, y_val, x_te, y_te, cfg)
    return r.mse_test_at_best, r.best_epoch, m.n_lut_params()


def run_poly(degree, hidden_dim, seed, ep=EPOCHS):
    m = PolyKAN2Layer(in_dim=2, hidden_dim=hidden_dim, out_dim=1, degree=degree)
    r = train_poly_kan2(m, x_tr, y_tr, x_val, y_val, x_te, y_te,
                        lr=1e-2, epochs=ep, batch_size=64, seed=seed)
    return r["mse_test_at_best"], r["best_epoch"]


print("=" * 70)
print("H9: LUT vs Poly at matched parameter budgets")
print("=" * 70)
print()

# ── Part 1: matched budget comparison ────────────────────────────────────────
print("Part 1: Matched budgets (one seed each for quick survey)")
print(f"{'Config':<28} {'params/edge':>12}  {'MSE':>10}  {'best_ep':>8}")
print("-" * 65)

for label, K, L, poly_deg, note in BUDGETS:
    params = K * L

    # LUT
    t0 = time.time()
    mse_lut, ep_lut, n_params = run_lut(K, L, seed=0, ep=EPOCHS)
    t_lut = time.time() - t0

    # Poly at same budget
    t0 = time.time()
    mse_poly, ep_poly = run_poly(poly_deg, hidden_dim=4, seed=0, ep=EPOCHS)
    t_poly = time.time() - t0

    print(f"  LUT  K={K:2d},L={L:2d} [{label:12s}]  {params:>12}  {mse_lut:>10.4e}  {ep_lut:>8}")
    print(f"  Poly deg={poly_deg:<4} [{label:12s}]  {poly_deg+1:>12}  {mse_poly:>10.4e}  {ep_poly:>8}")
    print(f"  {'LUT wins' if mse_lut < mse_poly else 'Poly wins':>50}  "
          f"ratio LUT/poly={mse_lut/mse_poly:.2f}x")
    print()

    all_results.append({
        "budget": label, "K": K, "L": L, "poly_degree": poly_deg,
        "params_per_edge": params,
        "lut_mse": round(mse_lut, 6), "lut_best_ep": ep_lut,
        "poly_mse": round(mse_poly, 6), "poly_best_ep": ep_poly,
        "lut_wins": mse_lut < mse_poly,
    })


# ── Part 2: poly degree sweep vs best LUT ────────────────────────────────────
print()
print("Part 2: Poly degree sweep — where is the crossover vs best LUT [2,4,1]?")
print("(Using K=16, L=32 for LUT — best config from H8)")

mse_best_lut, _, _ = run_lut(16, 32, seed=0, ep=EPOCHS)
print(f"  Best LUT K=16,L=32: MSE={mse_best_lut:.4e}")
print()
print(f"  {'degree':>8}  {'params/edge':>12}  {'MSE':>10}  {'vs best LUT':>12}")

crossover_degree = None
for degree in POLY_SWEEP_DEGREES:
    mse_p, ep_p = run_poly(degree, hidden_dim=4, seed=0, ep=EPOCHS)
    ratio = mse_p / mse_best_lut
    marker = " ← LUT wins!" if mse_p > mse_best_lut else ""
    print(f"  {degree:>8}  {degree+1:>12}  {mse_p:>10.4e}  {ratio:>12.3f}×{marker}")
    if mse_p > mse_best_lut and crossover_degree is None:
        crossover_degree = degree

if crossover_degree:
    print(f"\n  Crossover at degree={crossover_degree}: below this poly loses to LUT K=16,L=32")
else:
    print(f"\n  Poly wins at ALL tested degrees — no crossover found")


# ── Save ──────────────────────────────────────────────────────────────────────
out_path = os.path.join(OUT_DIR, "summary.json")
with open(out_path, "w") as f:
    json.dump({
        "config": {"epochs": EPOCHS},
        "budget_comparison": all_results,
        "poly_sweep": {"best_lut_mse": round(mse_best_lut, 6),
                       "crossover_degree": crossover_degree},
    }, f, indent=2)

print(f"\nResults saved to {out_path}")
