"""
H8 corrected experiment — all three fixes applied:
  1. cheby_init          (dead init fixed)
  2. freeze_norms + EMA  (norm collapse fixed)
  3. block_lr_scales     (gradient imbalance fixed)

Runtime: ~30 min on CPU (5 configs × 5 seeds × 1500 epochs)

Results written to results/H8_corrected/
"""
import sys, time, json, os
sys.path.insert(0, "src")

import torch
import numpy as np
from lut_native.targets import generate_data_2d
from lut_native.kan_stack import LUTKANStack
from lut_native.kan2 import LUTKAN2Layer
from lut_native.poly_kan2 import PolyKAN2Layer, train_poly_kan2
from lut_native.training_stack import StackTrainConfig, train_lut_stack
from lut_native.training_kan2 import KAN2TrainConfig, train_kan2
from lut_native.coverage_stack import compute_stack_coverage

# ── Config ────────────────────────────────────────────────────────────────────
K, L    = 16, 32
EPOCHS  = 1500
SEEDS   = [0, 1, 2, 3, 4]
OUT_DIR = os.path.join("results", "H8_corrected")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Data ──────────────────────────────────────────────────────────────────────
data = generate_data_2d("feynman_2d", n_train=800, n_val=300, n_test=400, seed=42)
x_tr, y_tr, x_val, y_val, x_te, y_te = data
y_tr  = y_tr.reshape(-1, 1)
y_val = y_val.reshape(-1, 1)
y_te  = y_te.reshape(-1, 1)


# ── Architecture definitions ──────────────────────────────────────────────────
def make_cfg(epochs=EPOCHS, scales=None, smooth=False, seed=0):
    return StackTrainConfig(
        cheby_init=True, cheby_scale=1.5, cheby_noise=0.05,
        lr=1e-2, epochs=epochs, batch_size=64, seed=seed,
        eval_every_epochs=50, lambda_2=0,
        freeze_norms=True, ema_alpha=0.01, recalibrate_every_epochs=0,
        block_lr_scales=scales,
    )


ARCHS = [
    # (tag,  kind,   kwargs,                               lr_scales)
    ("stack_2x4_cheby",   "stack", {"dims":[2,4,1]},          None),
    ("stack_3x4_balanced","stack", {"dims":[2,4,4,1]},         [0.1, 0.1, 1.0]),
    ("stack_3x4_uniform", "stack", {"dims":[2,4,4,1]},         None),
    ("kan2_noise_baseline","kan2", {"hidden_dim":4},             None),   # old baseline
    ("poly_4_d20",         "poly", {"hidden_dim":4, "degree":20},None),
]

# ── Run ───────────────────────────────────────────────────────────────────────
all_results = []
total = len(SEEDS) * len(ARCHS)
run_idx = 0

for seed in SEEDS:
    for tag, kind, kwargs, scales in ARCHS:
        run_idx += 1
        t0 = time.time()
        print(f"[{run_idx:2d}/{total}] {tag:<30} seed={seed} ...", end=" ", flush=True)

        try:
            if kind == "stack":
                m = LUTKANStack(K=K, L=L, **kwargs, smooth_norms=False)
                cfg = make_cfg(EPOCHS, scales, seed=seed)
                r = train_lut_stack(m, x_tr, y_tr, x_val, y_val, x_te, y_te, cfg)
                cov = compute_stack_coverage(m, x_tr) if len(m.norms) > 0 else None
                uni = [round(n.uniformity_mean, 4) for n in cov.norm_reports] if cov else []
                row = {
                    "tag": tag, "seed": seed, "kind": kind,
                    "mse_test":  round(r.mse_test_at_best, 6),
                    "best_epoch": r.best_epoch,
                    "norm_uni":  uni,
                    "elapsed_s": round(time.time() - t0, 1),
                }

            elif kind == "kan2":
                m = LUTKAN2Layer(in_dim=2, out_dim=1, K=K, L=L, **kwargs)
                cfg_k = KAN2TrainConfig(
                    lr=1e-2, epochs=EPOCHS, batch_size=64,
                    init_noise_std_absolute=0.05, seed=seed,
                    eval_every_epochs=50, lambda_2=0,
                )
                r = train_kan2(m, x_tr, y_tr, x_val, y_val, x_te, y_te, cfg_k)
                row = {
                    "tag": tag, "seed": seed, "kind": kind,
                    "mse_test":  round(r.mse_test_at_best, 6),
                    "best_epoch": r.best_epoch,
                    "elapsed_s": round(time.time() - t0, 1),
                }

            else:  # poly
                m = PolyKAN2Layer(in_dim=2, out_dim=1, **kwargs)
                r = train_poly_kan2(m, x_tr, y_tr, x_val, y_val, x_te, y_te,
                                    lr=1e-2, epochs=EPOCHS, batch_size=64, seed=seed)
                row = {
                    "tag": tag, "seed": seed, "kind": kind,
                    "mse_test":  round(r["mse_test_at_best"], 6),
                    "best_epoch": r["best_epoch"],
                    "elapsed_s": round(time.time() - t0, 1),
                }

            all_results.append(row)
            print(f"mse={row['mse_test']:.4e}  best_ep={row['best_epoch']}  ({row['elapsed_s']}s)")

        except Exception as e:
            print(f"ERROR: {e}")
            all_results.append({"tag": tag, "seed": seed, "error": str(e), "mse_test": float("nan")})


# ── Summary ───────────────────────────────────────────────────────────────────
from collections import defaultdict
by_tag = defaultdict(list)
for r in all_results:
    if "error" not in r:
        by_tag[r["tag"]].append(r["mse_test"])

print()
print("=" * 60)
print("H8 corrected — summary (mean MSE over seeds)")
print("=" * 60)
baseline_mse = float(np.mean(by_tag.get("kan2_noise_baseline", [0.648])))
for tag in ["poly_4_d20", "stack_2x4_cheby", "stack_3x4_balanced",
            "stack_3x4_uniform", "kan2_noise_baseline"]:
    vals = by_tag.get(tag, [])
    if not vals:
        continue
    m, s = float(np.mean(vals)), float(np.std(vals))
    ratio = m / baseline_mse
    print(f"  {tag:<30} {m:.4e} ± {s:.2e}  ratio={ratio:.3f}")

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = os.path.join(OUT_DIR, "summary.json")
with open(out_path, "w") as f:
    json.dump({
        "config": {"K": K, "L": L, "epochs": EPOCHS, "seeds": SEEDS},
        "raw_results": all_results,
        "summary": {
            tag: {
                "mse_mean": round(float(np.mean(v)), 6),
                "mse_std":  round(float(np.std(v)), 6),
                "n_seeds":  len(v),
            }
            for tag, v in by_tag.items()
        },
    }, f, indent=2)

print(f"\nResults saved to {out_path}")
