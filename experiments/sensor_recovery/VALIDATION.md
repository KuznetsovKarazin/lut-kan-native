# Validation performed on 2026-11-07

Environment: Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0,
PyTorch 2.15.0+cpu. Single-thread BLAS/OMP for the pilot and original tests.

| Check | Observed outcome |
|---|---|
| `python -m unittest discover -s experiments/sensor_recovery -p 'test_*.py' -v` | 11 passed |
| `PYTHONPATH=src pytest tests/ -q --tb=short` | 159 passed in 84.17 s |
| Full declared synthetic sweep | 144 cases / 7,488 evaluations completed; no missing groups |
| Bootstrap aggregation | Every summary group includes all 8 virtual units |
| Reproduction after gzip storage change | References, raw numeric results, summaries and paired/optimizer diagnostics match the initial run byte-for-byte; all recorded source hashes match |
| C inference parity | float32, uint8, int16 agree with host reference within 3e-7 normalized absolute tolerance on boundaries and interior inputs |
| `git diff --check` | Passed |

The original suite emitted one warning at `tests/test_stack.py:66` about
converting a gradient-tracking tensor to a Python scalar. No test failed.
The recovery test was also added to the GitHub Actions workflow, but this
document records local execution; it does not assert a remote CI outcome.

Physical-sensor tests, MCU execution/timing, online training, adaptive-grid
KANs, and generalization to new measured units have **not** been performed.
