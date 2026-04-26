# H8 — Multilevel LUT-KAN experiment results

Target: `feynman_2d` — f(x,y) = sin(πx) + 0.5 cos(2·π·xy)
K=16, L=32, 3 seeds (except stack_3x8: 1 seed), 400 epochs (under-converged).

## Test MSE summary

| Config | Budget (B) | MSE mean | MSE std | vs kan2_4 | Verdict |
|---|---|---|---|---|---|
| `poly_4_d20` | 3 686 | **1.40e-02** | 5.9e-03 | 0.022× | best |
| `poly_8_d20` | 6 400 | 2.01e-02 | 4.1e-03 | 0.031× | — |
| `stack_2x4` | 5 376 | 6.39e-01 | 1.8e-02 | 0.986× | 1.4% > baseline |
| `stack_3x4` | 9 728 | 6.39e-01 | 1.3e-02 | 0.986× | same as 2-layer |
| `stack_3x8` | 22 528 | 6.39e-01 | — | 0.986× | 1 seed only |
| `kan2_4` | 5 376 | 6.48e-01 | 5.1e-03 | 1.000× | baseline (diverges!) |

## Key results

- **H8 rejected**: depth does not help. All LUT-KAN stacks land at ~0.64.
- **Poly-KAN gap**: 45× advantage for poly_4_d20 over best LUT-KAN stack.
- **kan2_4 pathology**: best_epoch=0 (all seeds). Training diverges.
- **Norm-collapse fix**: freeze_norms=True restores uniformity=0.73 (narrow models).

## How to reproduce (full run)

```bash
cd lut-kan-native-v15
PYTHONPATH=src python3 scripts/exp_h8_multilevel.py
```

Writes summary.json and overwrites this README with full 5-seed results.
