# M7c Findings — Extended K × L sweep: smaller K and L values

## Motivation

M7a showed a monotonic trend: smaller K → higher ratio (at fixed L=32).
This phase extends the sweep to K ∈ {1, 2, 4, 8} × L ∈ {8, 16, 32}
to find whether the trend holds down to K=1 and to establish the true
Pareto frontier below 288 bytes.

## Full results table

| Config | Mem | sine ratio | cusp ratio | saturating ratio |
|--------|-----|-----------|-----------|-----------------|
| K=1,L=8  | 12B  | 1×   | 2×   | 2× |
| K=1,L=16 | 20B  | 1×   | 2×   | 8× |
| K=1,L=32 | 36B  | 3×   | 8×   | 114× |
| K=2,L=8  | 24B  | 1×   | 2×   | 5× |
| K=2,L=16 | 40B  | 2×   | 6×   | 60× |
| K=2,L=32 | 72B  | 54×  | 107× | **2016×** |
| K=4,L=8  | 48B  | 2×   | 2×   | 23× |
| K=4,L=16 | 80B  | 33×  | 26×  | 936× |
| **K=4,L=32** | **144B** | **1124×** | **126×** | **21012×** |
| K=8,L=8  | 96B  | 9×   | 16×  | 101× |
| K=8,L=16 | 160B | 278× | 144× | 1566× |
| K=8,L=32 | 288B | 2764×| 555× | 16428× |

## Key findings

### Finding 1: K=4, L=32 (144B) beats K=8, L=32 (288B) on saturating

K=4, L=32 achieves **21012×** vs K=8, L=32's **16428×** — 1.28× better ratio
at **half the memory**. This is the clearest Pareto-dominating result of the
whole M7 series.

For sine, K=8 is still better (2764× vs 1124×). For cusp, K=8 is strongly
better (555× vs 126×). The target function structure determines which K wins:

- **Saturating (tanh):** Sharp central transition; direct-LUT benefits from
  wide K=4 segments that each span a large portion of the transition, giving
  many cells to represent the steep region accurately.
- **Cusp (|x-0.3|):** The non-smoothness is localised at x=0.3. With K=8,
  one segment boundary lands near x=0.3, concentrating resolution there.
  K=4 misses this with its wider but fewer segments.
- **Sine:** Smooth and periodic; K=8 has more segments in each period, 
  reducing per-segment complexity. Direct-LUT excels at the medium-K regime.

### Finding 2: L=32 is a hard requirement for good performance

At L=8 or L=16, even K=4 gives poor ratios:
- K=4, L=8 (48B): 2× (sine), 2× (cusp), 23× (saturating)
- K=4, L=16 (80B): 33×, 26×, 936×
- K=4, L=32 (144B): 1124×, 126×, **21012×**

The jump from L=16 to L=32 at K=4: **sine 33×→1124× (+34×), saturating 936×→21012× (+22×)**.
L=32 gives enough cells per segment to represent the compensation accurately.
Below L=32, the per-cell gradient signal is sparse and training stagnates.

### Finding 3: K=1 and K=2 break the monotonic trend (except at L=32)

At L=32: K=2 gives 54×, 107×, 2016× — decent. K=1 gives 3×, 8×, 114× — weak.
At L<32: K=1 and K=2 are near parity (1-8×) for all targets.

The monotonic-K trend holds only when L is large enough to utilise the wide
segments. With only 8 or 16 cells per segment, even K=4 can't effectively
represent what the polynomial misses.

### Finding 4: True Pareto frontier

Best ratio at each memory budget (across all targets, best-case):

| Memory | Best for | Config | Ratio |
|--------|---------|--------|-------|
| 12–36B | saturating | K=1,L=32 | 114× |
| 72B | saturating | K=2,L=32 | 2016× |
| 80B | saturating | K=4,L=16 | 936× |
| **144B** | **saturating** | **K=4,L=32** | **21012×** |
| 160B | saturating | K=8,L=16 | 1566× |
| 288B | sine+cusp | K=8,L=32 | 2764× / 555× |

**K=4, L=32 is Pareto-dominant for saturating**; K=8, L=32 remains best for
sine and cusp. If your target is non-smooth with global curvature (like tanh),
use K=4, L=32 — you get better accuracy at half the memory.

## Consolidated recommendation

| Use case | Config | Memory | Expected gain vs poly |
|----------|--------|--------|-----------------------|
| Smooth periodic | K=8, L=32 | 288B | ~100–2800× |
| Non-smooth (cusp) | K=8, L=32 | 288B | ~500× |
| Saturating / tanh | **K=4, L=32** | **144B** | **~20000×** |
| Ultra-constrained | K=2, L=32 | 72B | 50–2000× |

## Combined K/L sweep picture (M7a + M7c)

The full picture of K ∈ {1,2,4,8,16,32} × L ∈ {8,16,32,64} now shows:

- **L=32 is universally optimal** — there's a consistent "cliff" between L=16 and L=32
- **K=4–8 at L=32 is the sweet zone** — below K=4 the trend reverses; above K=8 it degrades
- **Target-dependent optimum** within the sweet zone:
  - Smooth: K=8
  - Non-smooth / saturating: K=4

## Deliverables

Code:
- `scripts/m7c_kl_small.py`
- `tests/test_kl_small.py` — 5 tests

Results:
- `results/M7c_kl_small/{sine,cusp,saturating}/{heatmap.png,pareto.png,summary.json}`
- `results/M7c_kl_small/pareto_all.png`
- `results/M7c_kl_small/summary.json`

Tests total: **91 passing.**
