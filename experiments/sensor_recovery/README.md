# Sensor calibration recovery

This isolated experiment revisits the sensor paper with equal labelled-data
budgets, strong classical/KAN-edge comparators and real coefficient encoding.
All pilot data are **synthetic**. No MCU timing or physical accuracy is claimed.

- [Protocol and next stages](PROTOCOL.md)
- [Numerical findings and limitations](results/pilot_20261107/RESULTS.md)
- [Complete sweep](results/pilot_20261107/summary.csv)
- [Paired unit-level comparisons](results/pilot_20261107/paired_comparisons.json)
- [Run metadata](results/pilot_20261107/run_info.json)

## Run

From the repository root, Python >=3.10 with NumPy/SciPy/Matplotlib:

```bash
python -m unittest discover -s experiments/sensor_recovery -p 'test_*.py' -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/sensor_recovery/run_pilot.py --out /tmp/sensor-recovery-new --units 8 --repeats 3
python experiments/sensor_recovery/make_report.py /tmp/sensor-recovery-new
PYTHONPATH=src pytest tests/ -q --tb=short
```

The new pilot does not need PyTorch; the original repository tests do.
Completed run directories are not silently overwritten. On Windows set the
thread environment variables in PowerShell before running the same Python
commands. `cc` is needed for the native C parity test; this is a host test.

## Contents and scope

- `calibration.py`: synthetic sensors, budgeted calibration, regularized least
  squares, physical/polynomial/cubic/scalar KAN models, encoding and metrics.
- `run_pilot.py`: retained references/holdouts, complete sweep, paired unit
  bootstrap, selected hyperparameters and provenance.
- Raw evaluations and references use `.jsonl.gz` (standard gzip compression);
  no records are removed. Summary CSV/JSON remain directly readable.
- `lut_kernel.c`: portable float32, uint8 and int16 inference. Input must be
  finite and normalized; n >= 2. External normalization and output conversion
  use the four stored float32 transform constants. This is not an MCU benchmark
  or a complete on-device training implementation.
- `test_calibration.py`: spline identity, no test-label access in selection,
  real one-point baseline, quantization bounds, optimizer reference, C parity.
- `deployment_examples_*.json`: representative encoded LUTs, transforms and
  source hyperparameters. Total payload excludes firmware/stack/training RAM.

Uniform fixed-grid 1D LUT interpolation is a degree-one B-spline. The KAN
comparator is one cubic edge plus SiLU; its fitted function class can be
optimized linearly when the grid is fixed. There is no new spline theorem
and no claim about multilayer/adaptive-grid KAN performance here.

The legacy one-point API now requires an independently defined
`factory_predict` callback; the old all-data fitted-polynomial approximation
was scientifically incorrect. Historical saved results have not been rewritten.
The old v14 script retains other historical choices and is not the revised
publication pipeline. Use this directory for new evidence.
