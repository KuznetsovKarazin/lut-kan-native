#!/usr/bin/env python3
"""
Experiment H4: multi-edge KAN — single-edge findings do NOT transfer.

Honest null result. We test whether direct-LUT training, which achieves
100-1000x advantage in the single-edge setting, carries over to a two-layer
KAN [in_dim -> hidden_dim -> out_dim].

Setup:
    - 2D target:  y = sin(pi*x1) + 0.5*cos(2*pi*x1*x2)
      Not decomposable; genuinely requires multi-edge.
    - Three models at comparable layout:
        (a) Polynomial KAN2 (Chebyshev deg=8, trained via Adam)
        (b) Direct-LUT KAN2 (K=16, L=32, best single-edge lambdas)
        (c) Direct-LUT KAN2 + identity-like initialization

Findings (written up in docs/METHODOLOGY.md § 6):
    - Polynomial KAN2 reaches MSE ~1e-4.
    - Direct-LUT KAN2 does NOT train successfully on this task — test MSE
      remains ~5e-1, regardless of lambda, hidden_dim, or init strategy.
    - The likely cause is sparse-gradient pathology: each input sample
      touches only 2 LUT cells per edge, and in a 12-edge 2-layer model
      the update signal is diluted beyond what Adam can cope with.

This experiment does NOT find a working direct-LUT KAN2 configuration.
We publish it as a documented open problem. Paper claims are scoped to
single-edge only. Follow-up work needed:
    - Dense coordinate-wise gradient updates (not just the 2 visited cells)
    - Shared LUT atoms across edges (reduces effective parameter count)
    - Alternative initializations inspired by spline KANs
    - Curriculum / layer-wise training

Output: results/H4_multiedge/summary.json and log.txt

Runtime: ~4 min CPU.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lut_native import (  # noqa: E402
    KAN2TrainConfig,
    LUTKAN2Layer,
    PolyKAN2Layer,
    generate_data_2d,
    sample_polynomial_to_lut,
    train_kan2,
    train_poly_kan2,
)


def aggregate(vals):
    vals = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(vals.mean()),
        "std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
        "min": float(vals.min()),
        "max": float(vals.max()),
    }


def run_poly(x_tr, y_tr, x_v, y_v, x_te, y_te, hidden, degree, seeds, epochs):
    mses = []
    for seed in seeds:
        m = PolyKAN2Layer(in_dim=2, hidden_dim=hidden, out_dim=1, degree=degree)
        r = train_poly_kan2(m, x_tr, y_tr, x_v, y_v, x_te, y_te,
                            lr=5e-3, epochs=epochs, seed=seed)
        mses.append(r["mse_test_at_best"])
    return mses, m.memory_bytes_float32()


def run_lut_kan2(x_tr, y_tr, x_v, y_v, x_te, y_te, hidden, K, L,
                 l2, seeds, epochs, init_from_identity: bool):
    mses = []
    for seed in seeds:
        model = LUTKAN2Layer(in_dim=2, hidden_dim=hidden, out_dim=1, K=K, L=L)
        if init_from_identity:
            # Initialize each LUT from y = 0.5*x (scaled identity)
            id_coeffs = np.array([0.0, 0.5] + [0.0] * 19, dtype=np.float32)
            id_lut = sample_polynomial_to_lut(id_coeffs, K=K, L=L)
            with torch.no_grad():
                for i in range(2):
                    for h in range(hidden):
                        model.lut_l1.data[i, h] = torch.from_numpy(id_lut)
                for h in range(hidden):
                    model.lut_l2.data[h, 0] = torch.from_numpy(id_lut)
            noise = 0.05
        else:
            noise = 0.1

        cfg = KAN2TrainConfig(
            lambda_1=0.0, lambda_2=l2, lr=1e-2, epochs=epochs,
            batch_size=64, init_noise_std_absolute=noise,
            seed=seed, eval_every_epochs=30,
        )
        res = train_kan2(model, x_tr, y_tr, x_v, y_v, x_te, y_te, cfg)
        mses.append(res.mse_test_at_best)
    return mses, model.memory_bytes_uint8()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="results/H4_multiedge")
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"H4: multi-edge KAN  (target=feynman_2d, {len(args.seeds)} seeds)")
    print("=" * 72)

    x_tr, y_tr, x_v, y_v, x_te, y_te = generate_data_2d(
        "feynman_2d", n_train=1000, n_val=400, n_test=400, seed=42,
    )
    target_std2 = float(np.var(y_te))
    print(f"  target std^2 (trivial baseline MSE): {target_std2:.3e}\n")

    records = {}

    # Polynomial KAN2 at three hidden sizes
    print("Polynomial KAN2 (deg=8):")
    for h in [4, 8, 16]:
        t0 = time.time()
        mses, mem = run_poly(x_tr, y_tr, x_v, y_v, x_te, y_te,
                              hidden=h, degree=8,
                              seeds=args.seeds, epochs=args.epochs)
        agg = aggregate(mses)
        records[f"poly_h{h}_deg8"] = {
            "method": "PolyKAN2", "hidden_dim": h, "degree": 8,
            "memory_bytes": mem,
            "mse_values": mses, "mse_summary": agg,
        }
        print(f"    h={h:<3} mem={mem} B  MSE={agg['mean']:.3e}±{agg['std']:.1e}  dt={time.time()-t0:.1f}s")

    # LUT KAN2 at various lambdas and inits - we expect none of them to work well
    print("\nLUT KAN2 (K=16, L=32):")
    configs = [
        (False, "random", 0.0),
        (False, "random", 1e-3),
        (True,  "identity", 0.0),
        (True,  "identity", 1e-3),
    ]
    for (init_id, label, l2) in configs:
        t0 = time.time()
        mses, mem = run_lut_kan2(x_tr, y_tr, x_v, y_v, x_te, y_te,
                                  hidden=4, K=16, L=32, l2=l2,
                                  seeds=args.seeds, epochs=args.epochs,
                                  init_from_identity=init_id)
        agg = aggregate(mses)
        key = f"lut_h4_K16_L32_l2={l2}_init={label}"
        records[key] = {
            "method": "LUTKAN2", "hidden_dim": 4, "K": 16, "L": 32,
            "lambda_2": l2, "init": label,
            "memory_bytes": mem,
            "mse_values": mses, "mse_summary": agg,
        }
        print(f"    init={label:<8} l2={l2:<6}  mem={mem} B  MSE={agg['mean']:.3e}±{agg['std']:.1e}  dt={time.time()-t0:.1f}s")

    # Determine headline finding
    poly_best = min(records[k]["mse_summary"]["mean"] for k in records if k.startswith("poly_"))
    lut_best = min(records[k]["mse_summary"]["mean"] for k in records if k.startswith("lut_"))

    print(f"\nPolynomial KAN2 best: {poly_best:.3e}")
    print(f"LUT KAN2 best:        {lut_best:.3e}")
    print(f"Ratio LUT/Poly:       {lut_best/poly_best:.1f}x  "
          f"(>> 1 means LUT loses; < 1 means LUT wins)")
    print(f"Is LUT better than trivial predictor (MSE=std^2={target_std2:.3e}) ? "
          f"{'yes' if lut_best < target_std2 * 0.5 else 'NO — LUT does not learn usefully'}")

    with open(out_dir / "summary.json", "w") as f:
        json.dump({
            "target": "feynman_2d",
            "config": {"epochs": args.epochs, "seeds": args.seeds},
            "trivial_baseline_mse": target_std2,
            "records": records,
            "headline": {
                "poly_best_mse": poly_best,
                "lut_best_mse": lut_best,
                "lut_over_poly_ratio": lut_best / poly_best,
                "verdict": ("LUT-KAN2 does not train usefully on this task"
                            if lut_best > target_std2 * 0.5
                            else "LUT-KAN2 shows partial learning but loses to Poly-KAN2"),
            },
        }, f, indent=2)
    print(f"\nSaved: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
