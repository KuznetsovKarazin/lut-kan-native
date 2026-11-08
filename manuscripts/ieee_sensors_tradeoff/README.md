# IEEE Sensors Journal: review-driven revision

**Accuracy, Storage, and Update Cost in Reference-Assisted Sensor Calibration**

Oleksandr Kuznetsov, 8 November 2026. Working revision for scientific author review;
not a submitted or accepted article. Main paper: seven pages, five numbered
scientific figures, three tables and 22 references. Supplement: ten pages.

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
NLMS, two-point correction, causal all-history/recent-window refits, and extended
RLS forgetting-factor selection. Every negative result is retained. There are
3,488 additional static model evaluations, 2,304 development trajectories,
5,472 independent-confirmation trajectories and an audit of 331,776 original
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
  300 dpi metadata, less than 45 KB; PDF/SVG exports also included. It is not
  embedded after the article title.
- `REVIEW_RESPONSE_RU.txt`: point-by-point analysis of the colleague's review,
  changes made, disagreements supported by data, and remaining work.
- `BUILD_CHECKS.json`: completed package checks and their limits.
- `zenodo_metadata_draft.json`, `DEPOSIT_NOTES.txt`: proposed deposit metadata;
  no publication, release tag, or DOI has been created.

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
The document build uses pdfTeX/TeX Live 2023, IEEEtran v1.8b and latexmk 4.83.
The original and revision run manifests record exact calculation dependencies.

## Immutable experiment links

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
are linked above; the large raw synthetic archives are not claimed to have a
public DOI. Gzip timestamps may change upon reproduction; compare decoded records.

## Prospective hardware extension

The separate reference-run pack provides six static models on the later matched
fixture, with strict capture validation and image/source hashes. Mega2560,
ESP32-C3 and optional ESP32-S3 builds passed. The new kernels passed portable C++
parity and parser rejection tests; all focused sensor tests pass (126 tests and
8 subtests). This is not additional board execution. The ESP32-S3 environment
targets ESP32-S3-DevKitC-1 at 240 MHz through its UART bridge; another board needs
its own verified configuration. Instructions are in `mcu/RUN_REFERENCES_RU.txt`.
There is no need to rerun Mega/C3 merely to establish polynomial/physical timings,
since the earlier measured series already supplies them.

## Journal format and remaining checks

The draft uses the standard IEEE journal two-column class, a single-paragraph
150–250-word abstract, numbered figures/tables/equations, and a separate graphical
abstract. It contains no invented publication dates, journal volume, or DOI.
The official interactive selector did not provide a downloadable current Sensors
package in this environment; exact conformity to the current journal-specific
class still needs checking before submission. The editable source and vector
assets are portable. Official pages checked on 8 November 2026:

- [Sensors Journal author instructions](https://ieee-sensors.org/ieee-sensors-journal/for-authors/)
- [IEEE templates](https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/authoring-tools-and-templates/tools-for-ieee-authors/ieee-article-templates/)
- [Supplementary/graphical abstract guidance](https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/prepare-supplementary-materials/)

The AI-assistance statement includes experiment-design refinement, simulation,
analysis, firmware, acquisition, validation, visualization and drafting. Authorship,
affiliation/contact details, funding and final scientific claims need author review.
No unconfirmed coauthor or funding acknowledgment is added. A public DOI deposit
and release tag remain pending; metadata preparation is not publication.

## Для автора

Главные методические замечания приняты; рецензия разобрана в отдельном файле.
Работа теперь показывает измеренный компромисс, а не общую победу LUT. Сильный
дополнительный результат — конфликт скорости отслеживания дрейфа, остаточной
ошибки и памяти для хранения эталонов. Для Sensors Journal полезнее небольшой
физический опыт с калиброванным эталоном, чем очередное увеличение числа симуляций.
