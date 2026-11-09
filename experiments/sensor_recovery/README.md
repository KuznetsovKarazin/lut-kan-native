# Sensor calibration recovery

## Current V2 result: 9 November 2026

The seven-model common-fixture acquisition is complete: two validated sessions
each on Mega2560, ESP32-C3 and ESP32-S3. [Results and audit](results/v2_boards_20261109/RESULTS.md)
include 480 timing records, 42 accuracy checks, source/image/sidecar hashes and
captured-S3 native float32 opcode verification. LUT32 is fastest on Mega/C3;
polynomial Horner is fastest on S3 and improves payload and fixed-case error
over LUT32. This is a physical execution result with synthetic accuracy inputs.
No new hyperparameter search or simulation was performed. Zenodo 1.0.0 predates
these captures; new raw evidence needs a subsequent deposit version.

## Completed second-review refinement

The stronger 32-pair block OLS baseline and extended tau grid are complete.
[Results and reproduction](results/refinement_20261108/README.md).
The common-fixture captures required for the unified frontier are complete.
[mcu/RUN_REFERENCES.txt](mcu/RUN_REFERENCES.txt) retains the acquisition procedure.
Seven implementations include Clenshaw and power-basis Horner for the same
polynomial. Historical measurements remain separate. The earlier evidence is
published at https://doi.org/10.5281/zenodo.24062383; the new captures postdate it.
No physical sensor campaign has been completed. The completed 9 November V2 plan
includes S3-A through native USB Serial/JTAG; see
[mcu/V2_ACQUISITION_PROTOCOL.txt](mcu/V2_ACQUISITION_PROTOCOL.txt).
The decay grid is fixed: tau=2048 reduces eta only from 0.5 to 0.421 over 384
references. No third tuning cycle on the prior seeds is planned.

## Study and evidence index

This isolated experiment revisits the sensor paper with equal labelled-data
budgets, strong classical/KAN-edge comparators and real coefficient encoding.
All calibration data are **synthetic**. User-supplied Mega2560, ESP32-C3 and ESP32-S3
captures now provide measured execution times on the same numerical fixture;
physical sensor accuracy remains unmeasured.

The [revised IEEE Sensors Journal manuscript](../../manuscripts/ieee_sensors_tradeoff/README.md)
presents conditional accuracy, storage and update-cost trade-offs: eight main
pages, five scientific figures, a separate graphical abstract, a twelve-page
supplement and 23 references. It includes all seven implementations on one
common fixture, with a separate frontier per board. Earlier hardware fixtures
remain in the supplement. Post-review analyses and immutable evidence are indexed
in [revision results](results/revision_20261108/README.md); they add model-selection
sensitivity and causal recalibration baselines. The restricted LUT/Horner
ranking reversal is no longer the headline. Physical sensor uncertainty remains
unmeasured; the article requires scientific author review before submission.

- [Locally frozen revision protocol](REVIEW_REVISION_PROTOCOL.md)
- [Recovered hardware evidence and protocol addendum](REVIEW_REVISION_ADDENDUM.md)
- [New analyses, validation, hashes and optional hardware builds](results/revision_20261108/README.md)
- [Completed unified-reference/FPU acquisition procedure](mcu/RUN_REFERENCES.txt)

- [Protocol and next stages](PROTOCOL.md)
- [Numerical findings and limitations](results/pilot_20261107/RESULTS.md)
- [Complete sweep](results/pilot_20261107/summary.csv)
- [Paired unit-level comparisons](results/pilot_20261107/paired_comparisons.json)
- [Run metadata](results/pilot_20261107/run_info.json)
- [Monotonicity follow-up protocol](FOLLOWUP_PROTOCOL.md)
- [New-unit follow-up results](results/robust_20261108/RESULTS.md)
- [MCU benchmark and Windows instructions](mcu/README.md)
- [Compiled MCU sizes and validation](results/mcu_build_20261107/RESULTS.md)
- [First Mega2560 execution and duplicate-capture audit](results/mega_capture_20261107/RESULTS.md)
- [Current ESP32-C3 USB v3, failed v2 capture and build validation](results/usb_v3_build_20261108/RESULTS.md)
- [Previous USB v2 build audit](results/usb_v2_build_20261107/RESULTS.md)
- [Supplied Mega repeat and session-identity clarification](results/mega_repeat_review_20261107/RESULTS.md)
- [First complete ESP32-C3 session and comparison with Mega](results/c3_capture_20261108/RESULTS.md)
- [Two complete ESP32-C3 sessions and repeatability](results/c3_repeat_20261108/RESULTS.md)
- [Optimized cubic comparison protocol](OPTIMIZED_SPLINE_PROTOCOL.md)
- [Optimized cubic host results and interpretation](results/optimized_spline_20261108/RESULTS.md)
- [Optimized cubic MCU build sizes](results/optimized_spline_build_20261108/RESULTS.md)
- [Two physical C3 optimized-suite runs](results/c3_optimized_20261108/RESULTS.md)
- [Two Mega acquisitions and completed Mega/C3 comparison](results/mega_optimized_20261108/RESULTS.md)
- [One-upload, two-capture Windows procedure](mcu/OPTIMIZED_RUN.md)
- [Minimal physical NTC shopping/setup guide](PHYSICAL_VALIDATION.md)
- [Matched-storage and sequential-adaptation protocol](MATCHED_ADAPTATION_PROTOCOL.md)
- [Matched-storage and adaptation results](results/matched_adaptation_20261108/RESULTS.md)
- [Interpretation and continuation plan](results/matched_adaptation_20261108/RESULTS.md)
- [Matched MCU measurement protocol](MATCHED_MCU_PROTOCOL.md)
- [Matched MCU host validation and cross-compiled sizes](results/matched_mcu_20261108/RESULTS.md)
- [Reproduce the completed matched C3 and Mega suite](mcu/MATCHED_RUN.md)
- [First matched C3 attempt: successful upload, Windows inventory failure](results/c3_matched_failed_20261108/RESULTS.md)
- [Two complete matched C3 sessions: equal memory and full adaptation](results/c3_matched_20261108/RESULTS.md)
- [First matched Mega attempt and bounded startup recovery](results/mega_matched_failed_20261108/RESULTS.md)
- [Two complete matched Mega sessions and verified startup recovery](results/mega_matched_20261108/RESULTS.md)
- [Completed matched Mega/C3 comparison, figures and manuscript limits](results/matched_boards_20261108/RESULTS.md)

The matched C3 and Mega checkpoint is complete: four 144-byte inference representations
and complete float32 LUT/cubic NLMS trajectories, with and without full PAVA.
All 20 cross-compilations pass; host prediction/state gaps stay below 0.000016 °C
against the frozen reference. The new suites use device-acknowledged sessions
on both boards and package firmware images with captures. Two complete C3
captures give LUT32 12.051025 µs/call versus Horner8 13.984131 at equal payload:
13.824% less time, with MAE 0.0557110 versus 0.0455196 °C. Full projected LUT
NLMS takes 43.781250 µs/update versus cubic's 71.861979 on the step trajectory;
cubic is more accurate. PAVA runs without needing block merges on this fixture,
so these are not worst-case projection times. Two complete Mega sessions give
LUT32 167.279297 µs/call versus Horner8 192.788086: 13.232% less time with the
same accuracy trade-off. Full projected updates take 797.156250 µs for LUT
and 1149.541667 for cubic on the step trajectory. Source/image/raw/session
checks pass on both boards; the comparison retains separate session medians
and does not infer independent-device replication. The first Windows C3 attempt
uploaded successfully but stopped at the PlatformIO package inventory with a
CP1251 Unicode encoding error, before any capture. The runner now configures
UTF-8 for Python subprocesses and tolerates legacy console output; the firmware
is unchanged. The first Mega upload also passed, but its initial H greeting
was rejected before any run was requested. A Mega-only two-second greeting
delay and bounded pre-READY recovery now preserve startup evidence without
retrying Q/R or a benchmark. Both successful Mega sessions required one such
retry; the delay alone did not eliminate the initial rejection. Recovery is
now physically verified, while the cause of that rejection remains unknown.
C3 behavior and numerical/timing code are unchanged. The original checkpoint passed 120 tests and eight model subtests; the revision
suite passes 126 tests and eight subtests. No additional board acquisition is
needed for that completed checkpoint. The revised manuscript also uses plain
NLMS update timings and the recovered smaller-reference pilot; scientific
author review and physical sensor validation remain open.

The matched-storage and sequential-adaptation stages are now complete in a new,
explicit reconstruction run: 3312 static evaluations, development-only tuning
on 3168 candidate trajectories, then 3888 confirmation trajectories with no
numerical failures. At 144 bytes, the eight-piece Horner spline has lower mean
MAE than both float32 LUT32 and int16 LUT60 in all six scenario/budget groups.
Projected LUT NLMS corrects the injected drift, but projected cubic NLMS is
more accurate at the same 148-byte persistent-state count. Both can slightly
degrade already good calibration when no drift is present. These are synthetic
recalibration results with labelled references, not physical sensor validation.
The old 224-byte Horner MCU timings do not measure the new 144-byte comparator.

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
and Mega batches. The numerical matched-storage and adaptation follow-up
originally added no firmware changes. The subsequent matched MCU environments
have now been measured on both C3 and Mega.
Physical NTC accuracy remains a separate, unmeasured stage.

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

Reproduce the new campaign into fresh directories (the last two commands can
also verify/regenerate the committed results without rerunning the campaign):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/sensor_recovery/run_matched_adaptation.py static --out runs/matched/static
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/sensor_recovery/run_matched_adaptation.py development --out runs/matched/development
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/sensor_recovery/run_matched_adaptation.py confirmation --out runs/matched/confirmation --selection runs/matched/development/selected_parameters.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/sensor_recovery/verify_matched_adaptation.py runs/matched
python experiments/sensor_recovery/report_matched_adaptation.py runs/matched
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
