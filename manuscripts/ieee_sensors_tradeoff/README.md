# IEEE Sensors Journal: review-driven revision

**Accuracy, Storage, and Update Cost in Reference-Assisted Sensor Calibration**

Oleksandr Kuznetsov, 9 November 2026. Working revision for scientific author review;
not a submitted or accepted article. Main paper: eight pages, five numbered
scientific figures, three tables and 23 references. Supplement: see BUILD_CHECKS.json for the current page count.

## Second-review refinement

All 32 references per completed block now contribute to an affine OLS correction.
The four float32 sums use 16 B; gain/offset add 8 B, plus a counter and the frozen
model. Distorted NTC, B=75: final MAE 0.0484 C without drift and 0.0730 C after
step/ramp; integrated MAE 0.0966 C for step and 0.0692 C for ramp. The stronger
baseline substantially narrows the NLMS advantage.

The decay grid now includes 1024 and 2048. Both families select 2048, again at the
boundary. The step falls only from 0.5 to 0.421 over 384 references (15.8%):
the selected behavior is nearly constant over this horizon. The grid and seeds
are now closed to further selection. Additional MCU runs use frozen coefficients.
1440 development and 1728 paired-confirmation trajectories reuse the
previous seeds: this is post-review reanalysis, not an untouched new holdout.
Exact input snapshots are hash-checked under data/refinement_*.json. Historical
512/two-point results are retained in their original files. New plot labels and
explicit intermediate logarithmic ticks improve readability.

## Scientific claim and evidence boundaries

The result is a conditional accuracy–storage–computation trade-off. In the
original static study, compact cubics outperform float32 LUTs in 15/18 cells,
including all six 80-byte cells; no LUT is the most accurate equal-payload
candidate in any cell. The three compact-cubic exceptions occur at 224 bytes
with 30 references and reflect fitting/selection limitations. The restricted
C²-to-Horner expansion is no longer used to claim general LUT accuracy superiority.

The V2 common fixture now measures all seven models on Mega2560, C3 and S3.
LUT32 is fastest on Mega/C3, while polynomial Horner is faster, smaller and more
accurate than LUT32 on S3. The earlier pilot has a different seed and remains
in the supplement, outside the common frontier. Ordinary LUT NLMS updates take
48.7–53.3% less time than cubic updates on the original matched Mega/C3 fixture,
with higher residual error; S3 update time was not measured. The analytical AVR
acquisition-cycle example gives about 1.50% total reduction for LUT32 versus
piecewise Horner8, using the new V2 inference times.

New analyses add ideal/noisy/max error separation, an exact-generator noisy-input
inverse benchmark, fivefold model selection, generator sensitivity, decaying
NLMS, blockwise least-squares affine correction, causal all-history/recent-window refits, and extended
RLS forgetting-factor selection. Every negative result is retained. There are
3,488 additional static model evaluations, 2,304 development trajectories,
5,472 first-revision confirmation trajectories and an audit of 331,776 original
projected updates. Development and confirmation seeds are separate. The protocol
was frozen locally before calculation, not publicly preregistered.

Accuracy and drift remain synthetic. Physical boards establish execution cost,
not sensor uncertainty. Physical sensor validation, measured energy, WCET, peak
stack and between-board variation remain open. The supplement gives a concrete
prospective physical campaign and uncertainty components without invented values.

## Files and standalone build

- `main.tex`, `supplement.tex`, `references.tex`: editable article sources.
- `make_assets.py`, `make_revision_assets.py`: regenerate numerical assets from
  included hash-checked inputs; they do not synthesize hardware measurements.
- `data/source_hashes.json`: original immutable input identities;
  `data/revision_source_hashes.json`: five additional input hashes.
- `figures/`: vector PDF/SVG and PNG exports. The five main figures are
  `protocol`, `cv_capacity`, `revision_adaptation`, `sensitivity`, and
  `hardware_complete`. Historical plots remain for traceability and are not all
  used in the revised paper. The obsolete ranking-reversal graphical abstract
  was removed.
- `figures/gagraphic.png`: separate graphical abstract, 660 × 295 pixels,
  300 dpi metadata; PDF/SVG exports also included. A separate vector variant with readable labels is embedded in the JSEN colored abstract panel.
- `REVIEW_RESPONSE_RU.txt`: point-by-point analysis of the colleague's review,
  changes made, disagreements supported by data, and remaining work.
- `BUILD_CHECKS.json`: completed package checks and their limits.
- `zenodo_record.json`, `DEPOSIT_NOTES.txt`: published Zenodo record,
  version-specific DOI, archive checksums and anonymous-access verification.

In Overleaf select `main.tex` as the main document; build `supplement.tex`
separately. Figures and table inputs are included, so no repository checkout is
needed to compile either PDF.

```sh
python -m pip install -r requirements.txt
python make_assets.py
latexmk -pdf main.tex
latexmk -pdf supplement.tex
```

Asset regeneration was performed with Python 3.12.14, NumPy 2.3.5,
Matplotlib 3.10.8 and Pillow; numerical experiments also use SciPy 1.17.0.
The main document uses the publisher ieeecolor class and JSEN style parameters;
see TEMPLATE_PROVENANCE.txt for sources, working-copy differences and the
verified identity of the official May 2024 ZIP. The supplement retains its previous format.
The original and revision run manifests record exact calculation dependencies.

## Immutable experiment links

- [Second-review OLS, extended decay grid, audits and seven-kernel fixture](https://github.com/KuznetsovKarazin/lut-kan-native/tree/45edb31b3e67821a06e4131d0b82785248de5a5f/experiments/sensor_recovery)
- [Second-review results and reproduction commands](https://github.com/KuznetsovKarazin/lut-kan-native/tree/45edb31b3e67821a06e4131d0b82785248de5a5f/experiments/sensor_recovery/results/refinement_20261108)

- [Revision source, protocol and tests](https://github.com/KuznetsovKarazin/lut-kan-native/tree/0b828da03ce2132b672a869c37e0e1b0c86a75d5/experiments/sensor_recovery)
- [New summaries, per-unit results, hashes and reproduction commands](https://github.com/KuznetsovKarazin/lut-kan-native/tree/0b828da03ce2132b672a869c37e0e1b0c86a75d5/experiments/sensor_recovery/results/revision_20261108)
- [Recovered physical timings, including small references](https://github.com/KuznetsovKarazin/lut-kan-native/tree/0b828da03ce2132b672a869c37e0e1b0c86a75d5/experiments/sensor_recovery/results/revision_20261108/recovered_hardware.json)
- [Original immutable evidence](https://github.com/KuznetsovKarazin/lut-kan-native/tree/db39238c1467169b65ff1f3581945e57ee7ffd7a/experiments/sensor_recovery)
- [Original matched hardware results](https://github.com/KuznetsovKarazin/lut-kan-native/blob/db39238c1467169b65ff1f3581945e57ee7ffd7a/experiments/sensor_recovery/results/matched_boards_20261108/RESULTS.md)

The original large raw archive is 140,570,539 bytes, SHA-256
`c1594e1668d1c0c0f1bcd4e1c27ac91bfec2393a182fd468e37483e02ac6e1f3`.
The three additional compressed record streams total 77,720,317 bytes; individual
hashes are in `results/revision_20261108/raw_manifest.json`. Both are preserved
separately from Git. Public source, summaries, per-unit results and raw MCU logs
are linked above. All three raw evidence archives and the pinned research code
are publicly deposited in [Zenodo, version 1.0.0](https://doi.org/10.5281/zenodo.24062383). Gzip timestamps may change upon reproduction; compare decoded records.

## Completed V2 hardware extension

Six sessions are now audited: two each on Mega2560, ESP32-C3 and ESP32-S3.
All seven implementations use one frozen fixture. LUT32 is fastest on Mega/C3;
polynomial Horner is fastest on S3, where it uses 4.2% less time, 61.1% less
payload and has 20.8% lower synthetic MAE than LUT32. S3 uses native USB
Serial/JTAG at 240 MHz, 4 MB DIO flash at 80 MHz, and no PSRAM. The actual captured
ELF verifies native float32 instructions in all seven predictors. No fitted
coefficients, parameters or seeds changed. No new simulation was performed.

The evidence and reproducible audit are fixed at [df19065](https://github.com/KuznetsovKarazin/lut-kan-native/tree/df1906500c609bf8eabe41862432f79475548ba9/experiments/sensor_recovery/results/v2_boards_20261109).
data/v2_hardware.json supplies the new plot/table inputs, with a SHA-256 file.
make_v2_assets.py generates the three-board frontier and both supplemental tables.
The supplement retains the historical fixtures separately. No S3 update timings,
physical sensor measurements, energy, WCET or measured peak stack are claimed.

## Journal format and remaining checks

The working draft uses JSEN colored layout, a single-paragraph abstract, and
numbered figures/tables/equations. The official May 2024
JSEN ZIP has been inspected. The supplied ieeecolor.cls, jsen.sty and IEEEtran.bst
match its bytes exactly; provenance and working-copy adjustments are documented.
The evidence dataset is published in [Zenodo, version 1.0.0](https://doi.org/10.5281/zenodo.24062383).
This version-specific DOI identifies the earlier evidence only. The concept DOI
`10.5281/zenodo.24062382` refers to all versions. The complete deposit ZIP is
234,018,558 bytes; SHA-256:
`f0877b61eef306614e181e3d631a799d647d8fd7d0c332c31a1f6fff00755774`;
MD5: `3f6986c166c4911948fc6f4e3c657f35`. The server checksum matches, and
the DOI, public record and archive download return HTTP 200 without account authentication.
The evidence archive remains unchanged; historical draft-status wording inside
its original components records their preparation date. The evidence code is also fixed by the [sensors-zenodo-v1.0.0 release](https://github.com/KuznetsovKarazin/lut-kan-native/releases/tag/sensors-zenodo-v1.0.0),
pointing to the archived commit 45edb31. Real NTC measurements remain absent.

## Для автора

Главные методические замечания приняты; рецензия разобрана в отдельном файле.
Работа теперь показывает измеренный компромисс, а не общую победу LUT. Сильный
дополнительный результат — конфликт скорости отслеживания дрейфа, остаточной
ошибки и памяти для хранения эталонов. Для Sensors Journal полезнее небольшой
физический опыт с калиброванным эталоном, чем очередное увеличение числа симуляций.

## Acquisition plan and author checks

The two-session-per-board plan fixed on 9 November has been completed and audited.
See the unchanged MCU V2_ACQUISITION_PROTOCOL.txt for the plan and the V2 results
for the observations. Maximum host/device discrepancy is 0.00005341 C. All
480 timing and 42 accuracy records pass; raw batches and both session medians
are retained. Thirteen targeted parser/transport/numerical tests pass.
AUTHOR_REQUIREMENTS_20261109.txt documents the biography/photo requirement check.

The new V2 captures are NOT in the already published Zenodo 1.0.0. The matching
sensors-zenodo-v1.0.0 tag remains pinned to 45edb31. Add the separate V2 evidence
archive in a subsequent deposit version, preserving the earlier files. No new
version-specific DOI is claimed until publication.
