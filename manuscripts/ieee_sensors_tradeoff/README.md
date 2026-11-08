# IEEE Sensors Journal: review-driven revision

**Accuracy, Storage, and Update Cost in Reference-Assisted Sensor Calibration**

Oleksandr Kuznetsov, 8 November 2026. Working revision for scientific author review;
not a submitted or accepted article. Main paper: eight pages, five numbered
scientific figures, three tables and 23 references. Supplement: see BUILD_CHECKS.json for the current page count.

## Second-review refinement

All 32 references per completed block now contribute to an affine OLS correction.
The four float32 sums use 16 B; gain/offset add 8 B, plus a counter and the frozen
model. Distorted NTC, B=75: final MAE 0.0484 C without drift and 0.0730 C after
step/ramp; integrated MAE 0.0966 C for step and 0.0692 C for ramp. The stronger
baseline substantially narrows the NLMS advantage.

The decay grid now includes 1024 and 2048. Both families select 2048, again at the
boundary. 1440 development and 1728 paired-confirmation trajectories reuse the
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

The earlier physical MCU logs already include a selected 56-byte polynomial and
32-byte physical model. The polynomial has lower error and smaller payload but
longer inference time than LUT144 on that pilot fixture. Those four logs have
been reparsed and hashed. Their seed and errors differ from the later matched
fixture, so the two series occupy separate figure panels. Ordinary LUT NLMS
updates take 48.7–53.3% less time than compact-cubic updates in the matched
fixture, with higher residual error. A slow AVR acquisition-cycle example reduces
the end-to-end inference benefit to about 1.37%; this is an analytical budget.

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
  300 dpi metadata, less than 45 KB; PDF/SVG exports also included. A separate vector variant with readable labels is embedded in the JSEN colored abstract panel.
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

## Prospective hardware extension

The separate reference-run pack provides seven static implementations on the later matched
fixture, with strict capture validation and image/source hashes. Mega2560,
ESP32-C3 and optional ESP32-S3 builds passed. The new kernels passed portable C++
parity and parser rejection tests; all focused sensor tests pass (129 tests and
8 subtests). This is not additional board execution. The ESP32-S3 environment
targets ESP32-S3-DevKitC-1 at 240 MHz through its UART bridge; another board needs
its own verified configuration. Instructions are in `mcu/RUN_REFERENCES_RU.txt`.
New Mega/C3 acquisitions ARE required for one common-fixture frontier. The
earlier statement that they need not be rerun does not satisfy that comparison.
V2 adds power-basis Horner evaluation of the same degree-9 polynomial (56 B),
requires 80 timing records and seven parity records, and rejects V1 captures.

## Journal format and remaining checks

The working draft uses JSEN colored layout, a single-paragraph abstract, and
numbered figures/tables/equations. The official May 2024
JSEN ZIP has been inspected. The supplied ieeecolor.cls, jsen.sty and IEEEtran.bst
match its bytes exactly; provenance and working-copy adjustments are documented.
The evidence dataset is published in [Zenodo, version 1.0.0](https://doi.org/10.5281/zenodo.24062383).
Use this version-specific DOI for the paper. The concept DOI
`10.5281/zenodo.24062382` refers to all versions. The complete deposit ZIP is
234,018,558 bytes; SHA-256:
`f0877b61eef306614e181e3d631a799d647d8fd7d0c332c31a1f6fff00755774`;
MD5: `3f6986c166c4911948fc6f4e3c657f35`. The server checksum matches, and
the DOI, public record and archive download return HTTP 200 without account authentication.
The evidence archive remains unchanged; historical draft-status wording inside
its original components records their preparation date. No formal Git release
tag is claimed. Real NTC measurements remain absent.

## Для автора

Главные методические замечания приняты; рецензия разобрана в отдельном файле.
Работа теперь показывает измеренный компромисс, а не общую победу LUT. Сильный
дополнительный результат — конфликт скорости отслеживания дрейфа, остаточной
ошибки и памяти для хранения эталонов. Для Sensors Journal полезнее небольшой
физический опыт с калиброванным эталоном, чем очередное увеличение числа симуляций.
