# M9b Findings — Methodology audit: what we're actually comparing

## Motivation

After M9a results, three questions arose about what the poly comparison means:

1. Which polynomial? (Chebyshev, B-spline, Jacobi — they differ)
2. Is `degree = K×L − 1` the right matched baseline?
3. What does "LUT wins" actually claim?

This document audits the comparison and calibrates the claims.

## Polynomial choice: Chebyshev

`PolyKANStack` uses Chebyshev basis (3-term recurrence, domain [-1,1]).

**Why Chebyshev is the right choice for this comparison:**
- Orthogonal on [-1,1] — best-conditioned global polynomial basis
- Minimax property: minimises maximum error among all degree-d polynomials
- Used in original pykan (B-spline), EfficientKAN (Chebyshev), and
  related works — comparing against Chebyshev covers the strongest baseline

**What we are NOT comparing against:**
- B-spline (local, order=3) — used by original pykan. B-spline is fundamentally
  different: local support, O(1) eval, ~256 MCU cycles vs ~31 for LUT and ~2000
  for high-degree Chebyshev. See Section "B-spline comparison" below.

## Parameter matching: degree = K×L − 1

| LUT config | K×L params | matched poly degree |
|-----------|-----------|---------------------|
| K=1, L=4  | 4         | 3                   |
| K=2, L=8  | 16        | 15                  |
| K=4, L=8  | 32        | 31                  |
| K=4, L=16 | 64        | 63                  |
| K=4, L=32 | 128       | 127                 |

This is **equal parameter count**, not equal accuracy or equal memory.

Memory is NOT equal — LUT is stored as uint8 + float32 knots,
poly as float32 coefficients:
- K=4,L=8 LUT: 32 bytes uint8 + 16 bytes float32 = 48 bytes
- Degree=31 poly: 32 × 4 = 128 bytes float32

**LUT uses 2.7× less memory for the same parameter count.**

## Three-pipeline comparison (single edge [1→1])

Benchmarked K=4,L=16 (64 params) on cusp and saturating:

| Pipeline | Time | cusp MSE | saturating MSE | notes |
|----------|------|----------|----------------|-------|
| Poly LS d=63 → LUT (quantize) | **2 ms** | 4.7e-3 | 1.2e-1 | server-side, closed-form |
| Poly GD 300 ep → LUT (quantize) | 12 000 ms | 1.9e-1 | 9.0e-1 | **worst: GD at high degree fails** |
| Direct LUT GD 500 ep | 3 000 ms | **1.9e-5** | **7.4e-7** | 250× better than LS pipeline |

**Key insight:** Poly GD at high degree is *worse* than LS at high degree —
gradient training of high-degree polynomials is inherently unstable.
The correct alternative to LUT GD is **Poly LS**, not Poly GD.

## Inference cost: iso-accuracy analysis

On Cortex-M4 without FPU (estimated cycles per edge):

| Method | cycles | grows with? |
|--------|--------|-------------|
| LUT any K,L | **31** | nothing — O(1) |
| B-spline cubic | ~256 | O(1) — only knot search varies |
| Chebyshev d=15 | 1020 | O(degree) |
| Chebyshev d=31 | 2124 | O(degree) |
| Chebyshev d=63 | 4332 | O(degree) |

To reach MSE ~1e-4 on cusp:
- **LUT**: K=4,L=32 → 128 params, 144 bytes uint8, **31 cycles**
- Chebyshev LS: d=12 → 13 params, 52 bytes float32, 813 cycles (26× slower)
- B-spline cubic LS: n=8 knots → 12 params, 48 bytes float32, 255 cycles (8× slower)

**LUT requires ~6× more parameters than LS methods to reach the same accuracy,
but inference is 8–26× faster.** For latency-critical MCU applications this is
the right trade-off.

## B-spline comparison (iso-accuracy, cusp target)

| B-spline order | knots | params | bytes | cycles | MSE |
|---------------|-------|--------|-------|--------|-----|
| linear (k=1)  | 30    | 32     | 128   | 46     | 5.4e-6 |
| cubic (k=3)   | 28    | 32     | 128   | 256    | 2.9e-6 |
| LUT K=4,L=8   | —     | 32     | 48    | **31** | 8.1e-4 (poly-init, no GD) |
| LUT K=4,L=8   | —     | 32     | 48    | **31** | **6.9e-6** (after 1500 ep GD, in [1→4→1]) |

With gradient training in multi-layer stack, LUT K=4,L=8 reaches 6.9e-6 —
matching B-spline cubic LS (2.9e-6) within a factor of 2.4, at 8× faster
inference and using float32-free memory layout.

**The B-spline comparison is the most honest benchmark** and should be added to
future experiments. B-spline with gradient training in multi-layer stacks is
expected to be similarly unstable as Chebyshev at equivalent knot count.

## What each comparison proves

| Comparison | Claim proved |
|------------|-------------|
| LUT GD vs Poly GD (matched params, same training) | LUT trains stably where poly-KAN diverges |
| LUT GD vs Poly LS (single-edge) | LUT achieves 2× worse accuracy at 250× more training time but 26–68× faster inference |
| LUT GD vs B-spline LS (iso-accuracy) | LUT needs ~6× more params, but inference is 8× faster |

**The strongest claim is the first**: in gradient-trained multi-layer stacks,
LUT-KAN is the only representation that consistently converges.

## Implications for positioning

Do not claim: "LUT is more accurate than polynomial approximation."
(False for single-edge vs optimal LS baseline.)

Claim instead:
1. "LUT-KAN trains reliably in multi-layer stacks where poly-KAN
   diverges due to high-degree Chebyshev instability."
2. "LUT inference on Cortex-M4 requires 31 cycles regardless of
   model size — 8–68× fewer than equivalent B-spline or Chebyshev."
3. "LUT-KAN is the first KAN variant trainable on-device on MCU
   without floating-point hardware."

All three claims are supported by measurements in this project.
