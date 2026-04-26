"""
H10: Does depth help when K,L are correctly set?

== What we know before this experiment ==

CORRECT K,L criterion: N / (K*L) >= 50 (examples per cell)
  N=800 → K*L <= 16. Winner so far: K=2, L=8 (16 cells, 50 ex/cell)

K=2 specifically beats K=4 at the same cell count because:
  - K=2 has 2 segment boundaries (2 gradient discontinuities)
  - K=4 has 4 boundaries → harder optimisation landscape
  - K=2 acts like a smooth cubic spline; K=4 like a jagged piecewise

Best known results (1-layer [2→4→1], cheby_init, 2000 epochs, seed=0):
  K=2, L=8   → MSE = 3.86e-4   (best; still converging at ep=900/2000)
  K=2, L=4   → MSE = 6.74e-4
  poly d=3   → MSE = 1.52e-3
  poly d=20  → MSE = 2.15e-3
  poly d=15  → MSE = 1.54e-2   (matched K=2,L=8 budget: both 16 params/edge)

Gradient imbalance with K=2,L=8 (3-block stack):
  block ratio = 2×   (vs 3609× for K=16,L=32)
  → uniform LR works fine; no need for block_lr_scales

== What this experiment tests ==

Part A  Depth sweep  [2→4→1]    1-layer (no norm)       ← baseline
                     [2→4→4→1]  2-norm stack
                     [2→4→4→4→1] 3-norm stack
        all with K=2,L=8, cheby_init, 3500 epochs, 3 seeds
        Purpose: is depth strictly helpful, neutral, or harmful at correct K,L?

Part B  Width sweep  [2→4→1]  hidden=4  ← already known
                     [2→8→1]  hidden=8
                     [2→16→1] hidden=16
        all K=2,L=8, 1-layer, no norm, 3500 epochs
        Purpose: how much does width help vs depth?

Part C  K,L sweep + depth
        [2→4→4→1] at K=2,L=4 and K=2,L=8
        Purpose: does optimal K,L carry over to multi-layer?

Part D  Poly crossover sweep (corrected baseline)
        poly degree 1..10 vs K=2,L=8 [2→4→1]
        Purpose: at what degree does poly catch LUT? (H9 poly_sweep had
        wrong baseline K=16,L=32; this corrects it)

== Runtime estimate ==
~60 min on CPU: 3×3×3500 + 3×3500 + 4×3500 + 10×3500 epochs ≈ 80 configs * ~8s/100ep
With early stopping (patience=300) typical runs finish in 1000-1500 epochs.
"""

import sys, time, json, os
sys.path.insert(0, "src")

import torch
import numpy as np
from lut_native.targets import generate_data_2d
from lut_native.kan_stack import LUTKANStack
from lut_native.training_stack import auto_block_lr_scales
from lut_native.poly_kan2 import PolyKAN2Layer, train_poly_kan2
from lut_native.training_stack import StackTrainConfig, train_lut_stack
from lut_native.coverage_stack import compute_stack_coverage

# ─── Config ───────────────────────────────────────────────────────────────────
MAX_EPOCHS = 3500       # upper bound; early stopping often terminates earlier
PATIENCE   = 400        # stop if val MSE doesn't improve for this many epochs
SEEDS      = [0, 1, 2]
OUT_DIR    = os.path.join("results", "H10_depth_correct_KL")
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Data ─────────────────────────────────────────────────────────────────────
data = generate_data_2d("feynman_2d", n_train=800, n_val=300, n_test=400, seed=42)
x_tr, y_tr, x_val, y_val, x_te, y_te = data
y_tr  = y_tr.reshape(-1, 1)
y_val = y_val.reshape(-1, 1)
y_te  = y_te.reshape(-1, 1)
xt = torch.from_numpy(x_tr)
yt = torch.from_numpy(y_tr)


# ─── LUT training helper ──────────────────────────────────────────────────────
def run_lut(dims, K, L, seed, epochs=MAX_EPOCHS, patience=PATIENCE,
            smooth=False, scales=None):
    """
    Train a LUTKANStack with all corrections applied:
      - cheby_init        (fixes dead-cell init)
      - freeze_norms+EMA  (fixes norm collapse; not applicable for 1-block stacks)
      - uniform LR        (K=2,L=8 has 2× grad ratio — no need for scaling)
      - early stopping    (patience-based on val MSE)

    Returns dict with mse_test, best_epoch, n_lut_params, coverage info.
    """
    torch.manual_seed(seed)
    m = LUTKANStack(dims=dims, K=K, L=L, smooth_norms=smooth)
    cfg = StackTrainConfig(
        cheby_init=True, cheby_scale=1.5, cheby_noise=0.05,
        lr=1e-2, epochs=epochs, batch_size=64, seed=seed,
        eval_every_epochs=50, lambda_2=0,
        freeze_norms=True, ema_alpha=0.01, recalibrate_every_epochs=0,
        block_lr_scales=scales,
    )
    t0 = time.time()
    r  = train_lut_stack(m, x_tr, y_tr, x_val, y_val, x_te, y_te, cfg,
                         patience=patience)
    elapsed = time.time() - t0

    norm_uni = []
    if len(m.norms) > 0:
        try:
            cov     = compute_stack_coverage(m, x_tr)
            norm_uni = [round(n.uniformity_mean, 4) for n in cov.norm_reports]
        except Exception:
            pass

    return {
        "dims":       dims,
        "K": K, "L": L,
        "seed":       seed,
        "mse_test":   round(r.mse_test_at_best, 6),
        "mse_val":    round(r.mse_val_at_best, 6),
        "best_epoch": r.best_epoch,
        "n_lut":      m.n_lut_params(),
        "norm_uni":   norm_uni,
        "elapsed_s":  round(elapsed, 1),
    }


def run_poly(degree, hidden, seed, epochs=MAX_EPOCHS):
    m = PolyKAN2Layer(in_dim=2, hidden_dim=hidden, out_dim=1, degree=degree)
    r = train_poly_kan2(m, x_tr, y_tr, x_val, y_val, x_te, y_te,
                        lr=1e-2, epochs=epochs, batch_size=64, seed=seed)
    return {
        "degree":     degree,
        "hidden":     hidden,
        "seed":       seed,
        "mse_test":   round(r["mse_test_at_best"], 6),
        "best_epoch": r["best_epoch"],
        "n_params":   (degree + 1) * (2 * hidden + hidden),
    }


# ─── Early stopping shim ──────────────────────────────────────────────────────
# train_lut_stack doesn't have native patience; we patch it here via epochs.
# Strategy: run with eval_every_epochs=50 and check trace for no-improvement.
# Simpler: just set MAX_EPOCHS=3500 and let val-best tracking handle it.
# (The function already snapshots best model; if val plateaus, waste is minimal.)


# ─── Print helpers ────────────────────────────────────────────────────────────
def fmt(r):
    tag = f"[{','.join(map(str,r['dims']))}] K={r['K']},L={r['L']}"
    return (f"{tag:<30} seed={r['seed']}  mse={r['mse_test']:.3e}  "
            f"best_ep={r['best_epoch']:4d}  t={r['elapsed_s']:.0f}s")


# ─── Part A: Depth sweep ──────────────────────────────────────────────────────
print("=" * 68)
print("Part A: Depth sweep — K=2,L=8, hidden=4, 3 seeds")
print("=" * 68)

DEPTH_CONFIGS = [
    [2, 4, 1],             # 1-layer: 2 blocks (2→4, 4→1), 1 norm
    [2, 4, 4, 1],          # 2-layer: 3 blocks, 2 norms
    [2, 4, 4, 4, 1],       # 3-layer: 4 blocks, 3 norms
]

depth_results = []
for dims in DEPTH_CONFIGS:
    for seed in SEEDS:
        t0 = time.time()
        print(f"  dims={dims} seed={seed} ...", end=" ", flush=True)
        r = run_lut(dims, K=2, L=8, seed=seed)
        depth_results.append(r)
        print(fmt(r).split("seed")[1])  # just the metrics

# Aggregate
from collections import defaultdict
by_depth = defaultdict(list)
for r in depth_results:
    by_depth[str(r["dims"])].append(r["mse_test"])

print("\n  Summary (mean ± std over seeds):")
print(f"  {'dims':<20} {'MSE mean':>10}  {'MSE std':>9}  {'vs 1-layer':>10}")
baseline_1l = float(np.mean(by_depth[str([2, 4, 1])]))
for dims in DEPTH_CONFIGS:
    vals = by_depth[str(dims)]
    m, s = float(np.mean(vals)), float(np.std(vals))
    ratio = m / baseline_1l
    print(f"  {str(dims):<20} {m:>10.3e}  {s:>9.3e}  {ratio:>10.3f}×")


# ─── Part B: Width sweep (1-layer) ────────────────────────────────────────────
print("\n" + "=" * 68)
print("Part B: Width sweep — K=2,L=8, 1-layer, 3 seeds")
print("=" * 68)

WIDTH_CONFIGS = [
    [2, 4, 1],   # hidden=4  (known from Part A)
    [2, 8, 1],   # hidden=8
    [2, 16, 1],  # hidden=16
]

width_results = []
for dims in WIDTH_CONFIGS[1:]:   # skip [2,4,1] — already in Part A
    for seed in SEEDS:
        print(f"  dims={dims} seed={seed} ...", end=" ", flush=True)
        r = run_lut(dims, K=2, L=8, seed=seed)
        width_results.append(r)
        print(fmt(r).split("seed")[1])

# Add Part A [2,4,1] results for comparison
for r in depth_results:
    if r["dims"] == [2, 4, 1]:
        width_results.append(r)

by_width = defaultdict(list)
for r in width_results:
    by_width[str(r["dims"])].append(r["mse_test"])

print("\n  Summary:")
print(f"  {'dims':<20} {'MSE mean':>10}  {'MSE std':>9}  {'n_lut params':>13}")
for dims in WIDTH_CONFIGS:
    vals = by_width[str(dims)]
    m, s = float(np.mean(vals)), float(np.std(vals))
    # n_params: only first seed
    n = next((r["n_lut"] for r in width_results if r["dims"] == dims), "?")
    print(f"  {str(dims):<20} {m:>10.3e}  {s:>9.3e}  {n:>13}")


# ─── Part C: K,L × depth ─────────────────────────────────────────────────────
print("\n" + "=" * 68)
print("Part C: K,L × depth — does K=2,L=8 carry over to multi-layer?")
print("=" * 68)

KL_DEPTH = [
    ([2, 4, 4, 1], 2, 4,  "K=2,L=4 3-block"),
    ([2, 4, 4, 1], 2, 8,  "K=2,L=8 3-block"),  # already in Part A
    ([2, 4, 4, 1], 4, 8,  "K=4,L=8 3-block"),  # violates criterion (32 cells)
]

kl_depth_results = []
for dims, K, L, label in KL_DEPTH[:2]:  # skip K=4,L=8 (known bad)
    for seed in SEEDS:
        print(f"  {label} seed={seed} ...", end=" ", flush=True)
        r = run_lut(dims, K=K, L=L, seed=seed)
        r["label"] = label
        kl_depth_results.append(r)
        print(fmt(r).split("seed")[1])

# Add Part A 3-block results for K=2,L=8
for r in depth_results:
    if r["dims"] == [2, 4, 4, 1]:
        r2 = dict(r); r2["label"] = "K=2,L=8 3-block"
        kl_depth_results.append(r2)

by_kl_depth = defaultdict(list)
for r in kl_depth_results:
    by_kl_depth[r["label"]].append(r["mse_test"])

print("\n  Summary:")
for label in ["K=2,L=4 3-block", "K=2,L=8 3-block"]:
    vals = by_kl_depth[label]
    m, s = float(np.mean(vals)), float(np.std(vals))
    print(f"  {label:<22} MSE={m:.3e} ± {s:.3e}")


# ─── Part D: Poly crossover (corrected baseline) ──────────────────────────────
print("\n" + "=" * 68)
print("Part D: Poly crossover — vs correct LUT baseline K=2,L=8")
print("=" * 68)

best_lut_mse = float(np.mean(by_depth[str([2, 4, 1])]))
print(f"  LUT baseline K=2,L=8 [2→4→1]: MSE = {best_lut_mse:.3e}")
print()

poly_crossover_results = []
crossover_degree = None
for degree in [1, 2, 3, 4, 5, 6, 7, 8, 10]:
    r = run_poly(degree, hidden=4, seed=0, epochs=2000)
    poly_crossover_results.append(r)
    wins = "LUT wins" if best_lut_mse < r["mse_test"] else "poly wins"
    marker = " ← CROSSOVER" if best_lut_mse < r["mse_test"] and crossover_degree is None else ""
    print(f"  poly deg={degree:2d}  ({degree+1:3d} params/edge)  "
          f"MSE={r['mse_test']:.3e}  {wins}{marker}")
    if best_lut_mse < r["mse_test"] and crossover_degree is None:
        crossover_degree = degree

if crossover_degree:
    print(f"\n  → LUT K=2,L=8 beats poly up to degree={crossover_degree-1}")
    print(f"    Poly first beats LUT at degree={crossover_degree} ({crossover_degree+1} params/edge)")
else:
    print(f"\n  → LUT K=2,L=8 beats ALL tested poly degrees")


# ─── Save ─────────────────────────────────────────────────────────────────────
all_lut = depth_results + width_results + kl_depth_results
out_path = os.path.join(OUT_DIR, "summary.json")
with open(out_path, "w") as f:
    json.dump({
        "config": {
            "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "seeds": SEEDS,
            "note": "K=2,L=8 optimal per N/KL>=50 criterion"
        },
        "part_a_depth":     depth_results,
        "part_b_width":     width_results,
        "part_c_kl_depth":  kl_depth_results,
        "part_d_poly_sweep": poly_crossover_results,
        "summary": {
            "depth": {k: {"mean": round(float(np.mean(v)),6),
                          "std":  round(float(np.std(v)), 6)}
                      for k, v in by_depth.items()},
            "width": {k: {"mean": round(float(np.mean(v)),6),
                          "std":  round(float(np.std(v)), 6)}
                      for k, v in by_width.items()},
            "poly_crossover_degree": crossover_degree,
            "lut_baseline_mse": round(best_lut_mse, 6),
        }
    }, f, indent=2)

print(f"\nResults saved to {out_path}")
print()
print("=" * 68)
print("FINAL SUMMARY")
print("=" * 68)
print(f"  1-layer K=2,L=8:           MSE={baseline_1l:.3e}")
deep = float(np.mean(by_depth[str([2,4,4,1])]))
deeper = float(np.mean(by_depth[str([2,4,4,4,1])]))
print(f"  2-norm stack K=2,L=8:      MSE={deep:.3e}  "
      f"({'better' if deep < baseline_1l else 'worse'} {abs(deep/baseline_1l-1)*100:.0f}%)")
print(f"  3-norm stack K=2,L=8:      MSE={deeper:.3e}  "
      f"({'better' if deeper < baseline_1l else 'worse'} {abs(deeper/baseline_1l-1)*100:.0f}%)")
if crossover_degree:
    print(f"  Poly crossover at degree:  {crossover_degree}  ({crossover_degree+1} params/edge)")
