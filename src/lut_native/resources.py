"""
Resource accounting for LUT and polynomial evaluators.

Three axes:
  1. Static memory footprint (bytes, assuming MCU-style storage).
  2. Per-sample operation count (int ops, float ops) for forward pass.
  3. Measured CPU latency (timeit).

Intended for comparing three evaluators at matched *memory budget*:
  - Chebyshev polynomial (float32 coefficients)
  - Post-training LUT (uint8 per-segment asymmetric quantization)
  - Direct-trained LUT (same storage layout as post-training after quantization)

Operation counts are deliberately simple to match the MCU kernel implementation
in the main lut-kan repo (src/main.cpp, Variant A mixed kernel).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Memory accounting
# ─────────────────────────────────────────────────────────────────────────────

def polynomial_memory_bytes(degree: int, dtype: str = "float32") -> int:
    """Bytes to store a Chebyshev polynomial of given degree.

    Stored as (degree + 1) coefficients in the given dtype.
    """
    bytes_per = {"float32": 4, "float16": 2, "float64": 8}
    return (degree + 1) * bytes_per[dtype]


def lut_memory_bytes(
    K: int,
    L: int,
    dtype: str = "uint8",
    meta_dtype: str = "float16",
) -> int:
    """Bytes to store a segment-wise LUT, matching the main repo's layout.

    q_table: K * L * sizeof(dtype)
    scale:   K * sizeof(meta_dtype)  (one per segment)
    y_min:   K * sizeof(meta_dtype)  (one per segment)
    """
    bytes_per = {"uint8": 1, "int8": 1, "float16": 2, "float32": 4}
    b_q = bytes_per[dtype]
    b_m = bytes_per[meta_dtype]
    return K * L * b_q + 2 * K * b_m


def multi_edge_lut_memory_bytes(
    n_edges: int,
    K: int,
    L: int,
    dtype: str = "uint8",
    meta_dtype: str = "float16",
    share_knots: bool = True,
) -> int:
    """
    Multi-edge LUT memory. Each edge has its own (q_table, scale, y_min).
    Knots are shared by construction (K segments on same [x_min, x_max]).
    """
    per_edge = lut_memory_bytes(K, L, dtype=dtype, meta_dtype=meta_dtype)
    total = n_edges * per_edge
    if not share_knots:
        total += n_edges * (K + 1) * 4  # float32 knots per edge
    return total


# ─────────────────────────────────────────────────────────────────────────────
# Operation counts (per input sample)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OpCount:
    """Integer + floating-point op counts for evaluating one input sample."""
    int_ops: int = 0
    float_ops: int = 0
    memory_reads_bytes: int = 0  # approximate DRAM/flash touch

    def __add__(self, other: "OpCount") -> "OpCount":
        return OpCount(
            int_ops=self.int_ops + other.int_ops,
            float_ops=self.float_ops + other.float_ops,
            memory_reads_bytes=self.memory_reads_bytes + other.memory_reads_bytes,
        )

    def total_ops(self) -> int:
        return self.int_ops + self.float_ops


def polynomial_ops_per_sample(degree: int) -> OpCount:
    """Chebyshev 3-term recurrence: T_n = 2x*T_{n-1} - T_{n-2}.

    For degree d:
      - d-1 recurrence steps, each: 1 mul (2x * T_{n-1}) + 1 sub = 2 float ops
      - final dot product with coefficients: (d+1) muls + d adds = 2d+1 float ops
    Plus 1 normalization mul for 2x (reused across steps; count once).
    """
    if degree < 2:
        # Trivial cases
        return OpCount(float_ops=(degree + 1), memory_reads_bytes=(degree + 1) * 4)
    recurrence = 2 * (degree - 1) + 1  # +1 for 2x precompute
    dot = 2 * degree + 1
    return OpCount(
        float_ops=recurrence + dot,
        memory_reads_bytes=(degree + 1) * 4,  # read coefficients
    )


def lut_ops_per_sample(K: int, L: int, dtype: str = "uint8") -> OpCount:
    """Segment-wise LUT forward (matches v2.1 Variant A mixed kernel).

    Per sample:
      Segment index:
        - 1 float sub (x - x_min)
        - 1 float div (/ seg_width)  -> or 1 float mul (* inv_seg_width)
        - 1 float->int cast (floor)
        - 1 int clip
      LUT interpolation:
        - 1 float sub for u (fractional)
        - 1 float mul (* (L-1))
        - 1 float->int cast
        - 1 int add (r0+1) + 1 int clip
        - 2 int gather (read q_table) -> 2 bytes for uint8
        - 2 dequant multiplies:  y = y_min + scale * q  -> 2 float mul + 2 float add
        - 1 float lerp: (1-w)*v0 + w*v1 -> 2 mul + 1 add + 1 sub
      Also read scale[k], y_min[k] once each (2 reads)

    Approximate: ~3 float + 4 int for segment index/lut index math,
    +2 int gather + 2 float dequant + 3 float lerp = ~8 float, ~6 int.
    """
    bytes_per = {"uint8": 1, "int8": 1, "float16": 2, "float32": 4}
    b = bytes_per[dtype]
    # Segment selection: 3 float + 2 int
    # LUT indices: 2 float + 3 int
    # Gather: 2 reads of b bytes each; scale + y_min: 2 reads of 2 bytes each (float16)
    # Dequant of 2 values: 4 float
    # Lerp: 4 float
    return OpCount(
        float_ops=3 + 2 + 4 + 4,
        int_ops=2 + 3 + 2,  # 2 gathers counted as int ops
        memory_reads_bytes=2 * b + 2 * 2,  # 2 q values + scale + y_min (f16)
    )


def multi_edge_kan_ops(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    eval_ops_per_edge: OpCount,
) -> OpCount:
    """
    Operation count for a KAN [in_dim -> hidden_dim -> out_dim].

    Layer 1: in_dim * hidden_dim edges
    Layer 2: hidden_dim * out_dim edges
    Plus per-hidden-unit summation (in_dim - 1 adds) and activation
    normalization (we'll assume tanh-like: 1 float op).
    """
    n_edges_l1 = in_dim * hidden_dim
    n_edges_l2 = hidden_dim * out_dim
    base = OpCount(
        float_ops=(n_edges_l1 + n_edges_l2) * eval_ops_per_edge.float_ops,
        int_ops=(n_edges_l1 + n_edges_l2) * eval_ops_per_edge.int_ops,
        memory_reads_bytes=(n_edges_l1 + n_edges_l2) * eval_ops_per_edge.memory_reads_bytes,
    )
    # Summation at hidden nodes: hidden_dim nodes, each sums in_dim values -> (in_dim - 1) adds
    # Plus tanh activation: 1 op per hidden unit (approximate)
    base.float_ops += hidden_dim * (in_dim - 1 + 1)
    # Summation at output nodes
    base.float_ops += out_dim * (hidden_dim - 1)
    return base


# ─────────────────────────────────────────────────────────────────────────────
# CPU latency (measured)
# ─────────────────────────────────────────────────────────────────────────────

def time_forward(
    fn: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    n_warmup: int = 3,
    n_reps: int = 20,
) -> Dict[str, float]:
    """Measure per-sample wall-clock latency via repeated runs.

    Returns dict with mean/median/std/min of per-sample microseconds.
    """
    # Warmup
    for _ in range(n_warmup):
        fn(x)
    times = []
    N = len(x)
    for _ in range(n_reps):
        t0 = time.perf_counter()
        fn(x)
        times.append((time.perf_counter() - t0) / N * 1e6)  # us per sample
    arr = np.asarray(times, dtype=np.float64)
    return {
        "mean_us_per_sample": float(arr.mean()),
        "median_us_per_sample": float(np.median(arr)),
        "std_us_per_sample": float(arr.std(ddof=1)),
        "min_us_per_sample": float(arr.min()),
        "n_samples": int(N),
        "n_reps": int(n_reps),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Combined summary
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ResourceProfile:
    method: str
    memory_bytes: int
    ops_per_sample: OpCount
    latency: Dict[str, float] = field(default_factory=dict)
    extra: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "method": self.method,
            "memory_bytes": self.memory_bytes,
            "ops_int": self.ops_per_sample.int_ops,
            "ops_float": self.ops_per_sample.float_ops,
            "ops_total": self.ops_per_sample.total_ops(),
            "memory_reads_bytes": self.ops_per_sample.memory_reads_bytes,
            "latency_us_per_sample": self.latency.get("median_us_per_sample"),
            "latency_std_us": self.latency.get("std_us_per_sample"),
            **self.extra,
        }
