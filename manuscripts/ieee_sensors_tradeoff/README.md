# Accuracy, Storage, and Update Cost in Reference-Assisted Sensor Calibration

Oleksandr Kuznetsov. IEEE Sensors Journal submission preparation, 9 November 2026.
This is an implementation study. Accuracy and drift use controlled synthetic
responses; physical Mega2560, ESP32-C3 and ESP32-S3 boards establish execution
cost and numerical agreement. No physical sensor-accuracy specification is claimed.

## Reproduce figures and PDFs

```sh
python -m pip install -r requirements.txt
python make_assets.py
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex
```

The generators read frozen JSON under `data/` and verify hashes; they do not
refit models or run new experiments. The portable source package includes the
same inputs. `ieeecolor.cls`, `jsen.sty` and `IEEEtran.bst` are the unmodified
publisher files; see TEMPLATE_PROVENANCE.json. BUILD_CHECKS.json records the
current validation. The original author photograph is `figures/mcu_boards.jpg`,
with provenance in `data/mcu_photo.json`; it was not retouched.

## Results and scope

The current seven-model timing frontier uses one fixed distorted-NTC fixture,
with two warm sessions per board. LUT32 is fastest on Mega/C3. For the recorded
S3 binary, polynomial Horner uses 4.2% less gross time than LUT32, with 56 versus
144 bytes and lower fixed-case MAE. Captured ELF metadata records GCC 8.4.0,
-Os, -fno-fast-math and -ffp-contract=off for C3/S3. Optimization-level and code-layout
robustness, FMA-enabled timing, fixed-point polynomial kernels and Cortex-M4F/M33
boards were not tested. The result is not a causal FPU-only comparison.

The 24-file firmware/fixture/capture source set and all recorded numerical
results remain unchanged. Historical fixtures and failed attempts retain their
own identities. The earlier Mega/C3 adaptation timing is distinct from V2
inference. No S3 update time, physical NTC/LDR campaign, energy, WCET or peak-stack
measurement is included. The decay grid is closed at tau=2048; no third selection
cycle or additional simulated trajectories accompanies this revision.

## Evidence

- [Experiment entry point](../../experiments/sensor_recovery/README.md)
- [Frozen V2 audit and public raw logs](https://github.com/KuznetsovKarazin/lut-kan-native/tree/df1906500c609bf8eabe41862432f79475548ba9/experiments/sensor_recovery/results/v2_boards_20261109)
- [Earlier published data, Zenodo 1.0.0](https://doi.org/10.5281/zenodo.24062383)
- New data version 1.1.0: DOI 10.5281/zenodo.24082229 is reserved; publication pending.

Zenodo 1.0.0 and tag `sensors-zenodo-v1.0.0` retain the earlier evidence at
45edb31. Do not imply that this old version contains the six V2 captures.
The submission package includes the paper, supplement, separate graphical
abstract, source ZIP and cover letter. Author working notes are kept separately.
