#!/usr/bin/env python3
"""
generate_all_figures.py — Запуск всех экспериментов с нуля.

Воспроизводит все результаты из results/ одной командой.

Использование:
    python scripts/generate_all_figures.py              # полный прогон (~20 мин)
    python scripts/generate_all_figures.py --quick      # smoke-test для CI (~3 мин)
    python scripts/generate_all_figures.py --skip H1a H1b  # пропустить эксп.
    python scripts/generate_all_figures.py --only SENSOR   # только sensor study

Эксперименты:
    H1a  — lambda sweep (TNNLS Fig. 2)
    H1b  — memory (L) sweep
    H1c  — target sweep
    H2   — effective rank analysis
    H4   — multi-edge KAN (2D)
    H5   — resource benchmark
    SENSOR — multi-sensor calibration v14 (TIM paper)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def run(cmd: list, descr: str):
    """Запускает команду, выводит прогресс, завершается при ошибке."""
    print("\n" + "=" * 72)
    print(f"# {descr}")
    print(f"$ {' '.join(str(c) for c in cmd)}")
    print("=" * 72)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT)
    elapsed = time.time() - t0
    if r.returncode != 0:
        print(f"\nОШИБКА: {descr}")
        sys.exit(r.returncode)
    print(f"  завершено за {elapsed:.1f}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true",
                    help="Уменьшенные параметры для CI/smoke-test")
    ap.add_argument("--skip", type=str, nargs="+", default=[],
                    metavar="EXP",
                    help="Эксперименты для пропуска: H1a H1b H1c H2 H4 H5 SENSOR")
    ap.add_argument("--only", type=str, nargs="+", default=[],
                    metavar="EXP",
                    help="Запустить только эти эксперименты")
    args = ap.parse_args()

    py = sys.executable

    # Определяем параметры в зависимости от --quick
    if args.quick:
        epochs, seeds = 200, ["0", "1"]
    else:
        epochs, seeds = 1500, ["0", "1", "2", "3", "4"]

    def should_run(name: str) -> bool:
        if args.only:
            return name in args.only
        return name not in args.skip

    # ── TNNLS experiments ────────────────────────────────────────────────────

    if should_run("H1a"):
        run([py, str(HERE / "exp_H1_lambda_sweep.py"),
             "--epochs", str(epochs), "--seeds", *seeds],
            "H1a: lambda sweep (TNNLS Fig. 2 left)")

    if should_run("H1b"):
        run([py, str(HERE / "exp_H1_memory_sweep.py"),
             "--epochs", str(epochs), "--seeds", *seeds],
            "H1b: memory sweep → results/H1b_memory_sweep/")

    if should_run("H1c"):
        run([py, str(HERE / "exp_H1_targets.py"),
             "--epochs", str(epochs), "--seeds", *seeds],
            "H1c: target sweep → results/H1c_targets/")

    if should_run("H2"):
        run([py, str(HERE / "exp_H2_effective_rank.py"),
             "--epochs", str(epochs), "--seeds", "0", "1", "2"],
            "H2: effective rank → results/H2_effective_rank/")

    if should_run("H4"):
        run([py, str(HERE / "exp_H4_kan2_2d.py"),
             "--poly-epochs", str(max(epochs, 800) if not args.quick else 200),
             "--lut-epochs",  str(max(epochs // 2, 200) if not args.quick else 100),
             "--seeds", *seeds, "--lambda-2", "0.0"],
            "H4: multi-edge KAN → results/H4_kan2_2d/")

    if should_run("H5"):
        run([py, str(HERE / "exp_H5_resources.py"),
             "--poly-epochs", str(400 if not args.quick else 100),
             "--n-reps", str(20 if not args.quick else 5)],
            "H5: resource benchmark → results/H5_resources/")

    # ── TIM sensor calibration experiment ───────────────────────────────────

    if should_run("SENSOR"):
        run([py, str(HERE / "exp_sensor_calib_v14.py")],
            "SENSOR: multi-sensor calibration v14 → results/sensor_calib_v14/")

    print("\n" + "=" * 72)
    print("Все эксперименты завершены.")
    print("Результаты: results/")
    print("=" * 72)


if __name__ == "__main__":
    main()
