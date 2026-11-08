# Sensor calibration recovery

This isolated experiment revisits the sensor paper with equal labelled-data
budgets, strong classical/KAN-edge comparators and real coefficient encoding.
All calibration data are **synthetic**. User-supplied Mega2560 and ESP32-C3
captures now provide measured execution times on the same numerical fixture;
physical sensor accuracy remains unmeasured.

- [Protocol and next stages](PROTOCOL.md)
- [Numerical findings and limitations](results/pilot_20261107/RESULTS.md)
- [Complete sweep](results/pilot_20261107/summary.csv)
- [Paired unit-level comparisons](results/pilot_20261107/paired_comparisons.json)
- [Run metadata](results/pilot_20261107/run_info.json)
- [Monotonicity follow-up protocol](FOLLOWUP_PROTOCOL.md)
- [New-unit follow-up results](results/robust_20261108/RESULTS.md)
- [MCU benchmark and Windows instructions (Russian)](mcu/README_RU.md)
- [Compiled MCU sizes and validation](results/mcu_build_20261107/RESULTS.md)
- [First Mega2560 execution and duplicate-capture audit](results/mega_capture_20261107/RESULTS.md)
- [Current ESP32-C3 USB v3, failed v2 capture and build validation](results/usb_v3_build_20261108/RESULTS.md)
- [Previous USB v2 build audit](results/usb_v2_build_20261107/RESULTS.md)
- [Supplied Mega repeat and session-identity clarification](results/mega_repeat_review_20261107/RESULTS.md)
- [First complete ESP32-C3 session and comparison with Mega](results/c3_capture_20261108/RESULTS.md)
- [Two complete ESP32-C3 sessions and repeatability](results/c3_repeat_20261108/RESULTS.md)
- [Optimized cubic comparison protocol](OPTIMIZED_SPLINE_PROTOCOL.md)
- [Optimized cubic host results and interpretation (Russian)](results/optimized_spline_20261108/REPORT_RU.md)
- [Optimized cubic MCU build sizes](results/optimized_spline_build_20261108/RESULTS.md)
- [Two physical C3 optimized-suite runs (Russian)](results/c3_optimized_20261108/REPORT_RU.md)
- [Two Mega acquisitions and completed Mega/C3 comparison (Russian)](results/mega_optimized_20261108/REPORT_RU.md)
- [One-upload, two-capture Windows procedure](mcu/OPTIMIZED_RUN_RU.md)
- [Minimal physical NTC shopping/setup guide (Russian)](PHYSICAL_SETUP_RU.md)

Two complete C3 USB v3 command sessions validate all eight model checks, all
ten timing batches per method and all ten monotone-update batches. The raw
logs, parsed results, distinct acquisition IDs, sidecars, hashes and startup
acknowledgements agree. Reported accuracy is identical across sessions; the
largest relative range of session timing medians is 0.001535%. This measures
repeatability on one C3, not variation across boards or cold boots.
Uniform LUT32/float32 takes 10.9517 µs/call versus 25.7686 for the selected
degree-nine polynomial (2.353×), with MAE 0.0524224 versus 0.0476196 °C and
payload 144 versus 56 bytes. Mega gave a 2.098× ratio for the same comparison.
Both sessions give these gross within-capture medians on a synthetic fixture,
and the projected-update step costs 18.2422 µs in both. The preliminary MCU
execution/capture checkpoint is complete. The optimized cubic follow-up now
has two numerically validated implementations, three host-process repetitions
at each of -O2/-Os, and sixteen MCU cross-compilations. The unrolled version
keeps the 80-byte payload; compiled Horner uses 224 bytes. Horner and LUT are
close in host time, with ordering dependent on compiler settings. Two new
physical C3 sessions now give Horner 12.138184 µs/call versus LUT32/float32
10.981934: LUT time is 9.526% lower, with 144 versus 224 bytes of model data
and MAE 0.0524224 versus 0.0465563 °C. Both new cubic methods pass the same
numerical gate. All accuracy fields match across the two sessions, and the
largest range of method timing medians is 0.0007844%. The batch/upload/source
chain is audited against the measured commit, accounting for Windows CRLF.
Two subsequent Mega host acquisition records also pass all ten checks:
LUT32/float32 takes 155.023438 µs/call versus Horner 165.250000 (6.189% lower
LUT time), with the same payloads and MAEs. The logs/JSON are byte-identical,
but distinct host sidecars and sequential timestamps are retained separately.
The reviewer now separates acquisition identity from content identity and
rejects conflicting metadata for a repeated ID. Mega host IDs are not echoed
by the device; neither these records nor the C3 sessions establish statistical
independence or between-board variation. The current MCU checkpoint is complete.
Two optimized KAN rearrangements failed the existing float32 parity gate and
remain host-only diagnostics. Physical NTC validation remains a separate stage.

The USB v3 transport is retained. Optional `esp32c3_opt`/`mega2560_opt`
environments add two cubic methods and identify the ten-model suite explicitly.
The default eight-model environments remain available. The batch runner
uploads once, captures twice and packages source/image hashes, upload logs and
all raw/parsed/session files; no USB reconnection is needed between captures.
The one-upload/two-capture path is now supported by the supplied Windows/C3
and Mega batches. Continue with matched-storage comparison, complete sequential
adaptation and physical NTCs. The current check set passes 57 tests and eight
model subtests; no firmware or model coefficients changed in the Mega review.

Reproduce the stored capture audit into a fresh directory:

```bash
python experiments/sensor_recovery/review_device_runs.py experiments/sensor_recovery/results/c3_repeat_20261108/captures/esp32c3_capture01.log experiments/sensor_recovery/results/c3_repeat_20261108/captures/esp32c3_capture02.log --reference-log experiments/sensor_recovery/results/mega_repeat_review_20261107/captures/mega2560_capture01.log --out runs/c3_review
```

## Run

From the repository root, Python >=3.10 with NumPy/SciPy/Matplotlib/pytest:

```bash
python -m pytest experiments/sensor_recovery/test_*.py -q
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/sensor_recovery/run_pilot.py --out /tmp/sensor-recovery-new --units 8 --repeats 3
python experiments/sensor_recovery/make_report.py /tmp/sensor-recovery-new
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/sensor_recovery/run_robust.py --out /tmp/sensor-recovery-robust --units 8 --repeats 3
python experiments/sensor_recovery/report_followup.py /tmp/sensor-recovery-robust
PYTHONPATH=src pytest tests/ -q --tb=short
```

The new pilot does not need PyTorch; the original repository tests do.
Completed run directories are not silently overwritten. On Windows set the
thread environment variables in PowerShell before running the same Python
commands. `cc` and `g++` are needed for the native kernel parity tests;
these are host tests. MCU build/capture dependencies have a separate
`mcu/requirements.txt`; no numerical Python packages are needed to flash a board.

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
