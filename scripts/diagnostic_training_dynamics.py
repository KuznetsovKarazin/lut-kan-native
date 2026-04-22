#!/usr/bin/env python3
"""
Supporting diagnostic: shows why val-based best-model selection matters.

Without it, we'd report the final-epoch MSE, which can be 10-100x worse than
the best-val MSE due to late-training overfitting. This plot is referenced
in docs/METHODOLOGY.md section 2.2.

Output: results/diagnostics/training_dynamics.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lut_native import (  # noqa: E402
    TrainConfig,
    fit_chebyshev_ls,
    generate_data,
    sample_polynomial_to_lut,
    train_lut_edge,
)


def main():
    out_dir = Path("results/diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)

    x_tr, y_tr, x_v, y_v, x_te, y_te = generate_data("sine", seed=42)
    coeffs = fit_chebyshev_ls(x_tr, y_tr, degree=20)
    lut_init = sample_polynomial_to_lut(coeffs, K=16, L=32)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Left: trace at best lambda, 3 seeds
    ax = axes[0]
    for seed in [0, 1, 2]:
        cfg = TrainConfig(lambda_1=0.0, lambda_2=1.0, lr=1e-2, epochs=300,
                          batch_size=64, init_noise_std_rel=0.01, seed=seed,
                          eval_every_epochs=5)
        r = train_lut_edge(lut_init=lut_init, x_train=x_tr, y_train=y_tr,
                           x_val=x_v, y_val=y_v, x_test=x_te, y_test=y_te,
                           x_min=-1.0, x_max=1.0, cfg=cfg)
        eps = np.array(r.trace["epoch"])
        train_mse = [m if m is not None else np.nan for m in r.trace["mse_train"]]
        val_mse = r.trace["mse_val"]
        ax.plot(eps, train_mse, "-", color=f"C{seed}", alpha=0.4,
                label=f"seed {seed} train" if seed == 0 else None)
        ax.plot(eps, val_mse, "-", color=f"C{seed}", linewidth=1.5,
                label=f"seed {seed} val")
        ax.axvline(r.best_epoch, color=f"C{seed}", linestyle=":", alpha=0.6)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title("Training dynamics ($\\lambda_2=1.0$): val-best comes early\n"
                 "(dashed vertical lines = best-val epochs)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    # Right: final MSE vs best-val MSE, at different lambdas
    ax = axes[1]
    lambdas = [0.0, 0.01, 0.1, 1.0]
    final_mses = []
    best_mses = []
    for l2 in lambdas:
        finals, bests = [], []
        for seed in [0, 1, 2]:
            cfg = TrainConfig(lambda_1=0.0, lambda_2=l2, lr=1e-2, epochs=300,
                              batch_size=64, init_noise_std_rel=0.01, seed=seed,
                              eval_every_epochs=5)
            r = train_lut_edge(lut_init=lut_init, x_train=x_tr, y_train=y_tr,
                               x_val=x_v, y_val=y_v, x_test=x_te, y_test=y_te,
                               x_min=-1.0, x_max=1.0, cfg=cfg)
            # Final-epoch val MSE (what a naive run would report)
            finals.append(r.mse_val_final)
            bests.append(r.mse_val_at_best)
        final_mses.append(np.mean(finals))
        best_mses.append(np.mean(bests))

    x = np.arange(len(lambdas))
    width = 0.35
    ax.bar(x - width/2, final_mses, width, label="Final-epoch (naive)", color="tab:red")
    ax.bar(x + width/2, best_mses, width, label="Best-val (used)", color="tab:green")
    ax.set_xticks(x)
    ax.set_xticklabels([f"$\\lambda_2$={l}" for l in lambdas])
    ax.set_ylabel("Val MSE")
    ax.set_yscale("log")
    ax.set_title("Naive final-epoch vs val-best reporting\n(3-seed mean; lower is better)")
    for i, (f, b) in enumerate(zip(final_mses, best_mses)):
        ax.text(i - width/2, f * 1.1, f"{f:.1e}", ha="center", fontsize=7.5)
        ax.text(i + width/2, b * 1.1, f"{b:.1e}", ha="center", fontsize=7.5)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "Why val-based best-model selection matters  (sine, K=16, L=32)",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(out_dir / "training_dynamics.png", dpi=130)
    plt.close()
    print(f"Saved: {out_dir / 'training_dynamics.png'}")


if __name__ == "__main__":
    main()
