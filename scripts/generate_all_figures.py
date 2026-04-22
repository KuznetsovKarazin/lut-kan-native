#!/usr/bin/env python3
"""
Run all experiments end-to-end. Regenerates results/ from scratch.

Usage:
    python scripts/generate_all_figures.py                 # full run (~12 min)
    python scripts/generate_all_figures.py --quick         # reduced for CI (~2 min)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def run(cmd, descr):
    print("\n" + "=" * 72)
    print(f"# {descr}")
    print(f"$ {' '.join(cmd)}")
    print("=" * 72)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        print(f"FAILED: {descr}")
        sys.exit(r.returncode)
    print(f"  completed in {time.time() - t0:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="Reduced setting for CI/smoke test")
    ap.add_argument("--skip", type=str, nargs="+", default=[],
                    help="Experiment ids to skip: H1a, H1b, H1c, H2")
    args = ap.parse_args()

    if args.quick:
        epochs = 200
        seeds = ["0", "1"]
    else:
        epochs = 1500
        seeds = ["0", "1", "2", "3", "4"]

    py = sys.executable

    if "H1a" not in args.skip:
        run([py, str(HERE / "exp_H1_lambda_sweep.py"),
             "--epochs", str(epochs), "--seeds", *seeds],
            "H1a: lambda sweep")

    if "H1b" not in args.skip:
        run([py, str(HERE / "exp_H1_memory_sweep.py"),
             "--epochs", str(epochs), "--seeds", *seeds],
            "H1b: memory (L) sweep")

    if "H1c" not in args.skip:
        run([py, str(HERE / "exp_H1_targets.py"),
             "--epochs", str(epochs), "--seeds", *seeds],
            "H1c: target sweep")

    if "H2" not in args.skip:
        # H2 runs faster with 3 seeds even in full mode
        run([py, str(HERE / "exp_H2_effective_rank.py"),
             "--epochs", str(epochs), "--seeds", "0", "1", "2"],
            "H2: effective rank analysis")

    if "H4" not in args.skip:
        run([py, str(HERE / "exp_H4_kan2_2d.py"),
             "--poly-epochs", str(max(epochs, 800) if not args.quick else 200),
             "--lut-epochs", str(max(epochs // 2, 200) if not args.quick else 100),
             "--seeds", *seeds, "--lambda-2", "0.0"],
            "H4: multi-edge KAN on 2D target")

    if "H5" not in args.skip:
        run([py, str(HERE / "exp_H5_resources.py"),
             "--poly-epochs", str(400 if not args.quick else 100),
             "--n-reps", str(20 if not args.quick else 5)],
            "H5: resource benchmark")

    print("\n" + "=" * 72)
    print("All experiments complete.")
    print("Results: results/")
    print("=" * 72)


if __name__ == "__main__":
    main()
