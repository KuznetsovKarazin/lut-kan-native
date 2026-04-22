"""
Target functions for the benchmark suite.

Each entry has:
  - name (string)
  - fn(x: np.ndarray) -> np.ndarray
  - a short description of what it stresses
  - recommended polynomial degree (for honest poly baseline)

Why these three:
  - sine     : smooth, band-limited. Polynomial is the theoretical best.
  - cusp     : C^0 but not C^1. Polynomials have a Gibbs-like floor here.
  - saturating : smooth but strongly non-linear (hard tanh). Tests how LUT
                 handles nearly-flat regions with sharp transitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np


@dataclass
class Target:
    name: str
    fn: Callable[[np.ndarray], np.ndarray]
    description: str
    recommended_poly_degree: int  # degree at which polynomial approximation is "saturated"


def _sine(x: np.ndarray) -> np.ndarray:
    """Multi-frequency sine, band-limited, smooth."""
    return (np.sin(2.0 * np.pi * x) + 0.5 * np.sin(4.0 * np.pi * x)).astype(np.float32)


def _cusp(x: np.ndarray) -> np.ndarray:
    """|x - 0.3| + 0.2 sin(3*pi*x): C^0 with a cusp at 0.3."""
    return (np.abs(x - 0.3) + 0.2 * np.sin(3.0 * np.pi * x)).astype(np.float32)


def _saturating(x: np.ndarray) -> np.ndarray:
    """tanh(4x) + 0.15 x: nearly-flat wings with a sharp central transition."""
    return (np.tanh(4.0 * x) + 0.15 * x).astype(np.float32)


def _piecewise_smooth(x: np.ndarray) -> np.ndarray:
    """Two smooth branches stitched at x=0.

    For x < 0:   f(x) = 0.5 * sin(3*pi*x)
    For x >= 0:  f(x) = 0.8 * (1 - exp(-3x)) - 0.2 * cos(2*pi*x)

    Both branches are individually smooth (C^inf), but they meet at x=0
    with a finite slope discontinuity:
      f(0-)  = 0,                             f'(0-) = 1.5 pi
      f(0+)  = 0.8*(1 - 1) - 0.2*cos(0) = -0.2,  [deliberate jump]
    The jump is small (0.2) so this stresses "local inhomogeneity":
    global polynomials struggle with the transition, LUT can place more
    effective resolution around it.
    """
    x = np.asarray(x, dtype=np.float32)
    left = 0.5 * np.sin(3.0 * np.pi * x)
    right = 0.8 * (1.0 - np.exp(-3.0 * x)) - 0.2 * np.cos(2.0 * np.pi * x)
    out = np.where(x < 0.0, left, right).astype(np.float32)
    return out


def _local_sharp_asymmetric(x: np.ndarray) -> np.ndarray:
    """Locally-sharp transition embedded in a smooth baseline; asymmetric.

    Baseline: gentle cubic 0.3*x + 0.1*x^3 (smooth everywhere)
    Plus:     narrow high-slope sigmoidal bump centered at x = -0.4

    Effective width of the sharp feature is ~0.08 (very local);
    elsewhere the function is lazy. This is the regime where a uniform-K
    polynomial wastes capacity on easy regions and underfits the sharp one.
    Asymmetric about x=0 by construction.
    """
    x = np.asarray(x, dtype=np.float32)
    baseline = 0.3 * x + 0.1 * x ** 3
    bump_center = -0.4
    bump_width = 0.08
    bump = 0.6 * np.tanh((x - bump_center) / bump_width)
    return (baseline + bump).astype(np.float32)


TARGETS: Dict[str, Target] = {
    "sine": Target(
        name="sine",
        fn=_sine,
        description="sin(2*pi*x) + 0.5*sin(4*pi*x); smooth, band-limited",
        recommended_poly_degree=20,
    ),
    "cusp": Target(
        name="cusp",
        fn=_cusp,
        description="|x - 0.3| + 0.2*sin(3*pi*x); C^0 with cusp at x=0.3",
        recommended_poly_degree=20,
    ),
    "saturating": Target(
        name="saturating",
        fn=_saturating,
        description="tanh(4x) + 0.15x; smooth but strongly non-linear (sharp transition)",
        recommended_poly_degree=20,
    ),
    "piecewise_smooth": Target(
        name="piecewise_smooth",
        fn=_piecewise_smooth,
        description="Two smooth branches meeting at x=0 with slope+value jump",
        recommended_poly_degree=24,
    ),
    "local_sharp": Target(
        name="local_sharp",
        fn=_local_sharp_asymmetric,
        description="Gentle cubic + narrow tanh bump at x=-0.4 (locally sharp, asymmetric)",
        recommended_poly_degree=24,
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# 2D targets (for multi-edge KAN benchmarks)
# ─────────────────────────────────────────────────────────────────────────────

def _feynman_2d(xy: np.ndarray) -> np.ndarray:
    """A 2D target with genuine cross-variable structure.

    f(x, y) = sin(pi * x) + 0.5 * cos(2 * pi * x * y)

    This cannot be decomposed into g(x) + h(y) — the product x*y in the
    second term means a univariate approximation of x alone (or y alone)
    is fundamentally insufficient. A 2-layer KAN [2 -> hidden -> 1] is
    a reasonable fit here; a single edge [1 -> 1] is not.
    """
    xy = np.asarray(xy, dtype=np.float32)
    x = xy[..., 0]
    y = xy[..., 1]
    return (np.sin(np.pi * x) + 0.5 * np.cos(2.0 * np.pi * x * y)).astype(np.float32)


def generate_data_2d(
    target_name: str,
    n_train: int = 1000,
    n_val: int = 400,
    n_test: int = 400,
    x_min: float = -1.0,
    x_max: float = 1.0,
    seed: int = 42,
):
    """Generate 2D input data for multi-edge KAN experiments.

    Train/val: uniform random samples.
    Test: deterministic grid (sqrt(n_test) per axis) for reproducibility.
    """
    if target_name == "feynman_2d":
        f = _feynman_2d
    else:
        raise KeyError(f"Unknown 2D target '{target_name}'. Available: feynman_2d")

    rng_tr = np.random.RandomState(seed)
    rng_v = np.random.RandomState(seed + 10_000)

    x_train = rng_tr.uniform(x_min, x_max, (n_train, 2)).astype(np.float32)
    x_val = rng_v.uniform(x_min, x_max, (n_val, 2)).astype(np.float32)

    # Test: deterministic grid
    side = int(np.sqrt(n_test))
    a = np.linspace(x_min, x_max, side, endpoint=False).astype(np.float32)
    xg, yg = np.meshgrid(a, a)
    x_test = np.stack([xg.ravel(), yg.ravel()], axis=-1)

    y_train = f(x_train)
    y_val = f(x_val)
    y_test = f(x_test)
    return x_train, y_train, x_val, y_val, x_test, y_test


def generate_data(
    target_name: str,
    n_train: int = 500,
    n_val: int = 200,
    n_test: int = 200,
    x_min: float = -1.0,
    x_max: float = 1.0,
    seed: int = 42,
):
    """Generate (x_train, y_train, x_val, y_val, x_test, y_test) for a named target.

    - Train: uniform random on [x_min, x_max)
    - Val  : uniform random on [x_min, x_max), distinct seed
    - Test : DETERMINISTIC grid on [x_min, x_max), identical across calls with same args
             -> enables apples-to-apples comparison across regimes.
    """
    if target_name not in TARGETS:
        raise KeyError(
            f"Unknown target '{target_name}'. Available: {list(TARGETS)}"
        )
    f = TARGETS[target_name].fn

    rng_train = np.random.RandomState(seed)
    rng_val = np.random.RandomState(seed + 10_000)

    x_train = rng_train.uniform(x_min, x_max, n_train).astype(np.float32)
    x_val = rng_val.uniform(x_min, x_max, n_val).astype(np.float32)
    x_test = np.linspace(x_min, x_max, n_test, endpoint=False).astype(np.float32)

    y_train = f(x_train)
    y_val = f(x_val)
    y_test = f(x_test)

    return x_train, y_train, x_val, y_val, x_test, y_test
