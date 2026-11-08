# IEEE Sensors Journal: first manuscript draft

**Accuracy–Memory–Latency Trade-Offs in Embedded Sensor Calibration and Adaptation**

Author: Oleksandr Kuznetsov. Draft dated 8 November 2026. This is an unpublished
manuscript for author review, not an accepted article or a submitted version.

The contribution is a controlled engineering comparison: exact serialized
storage, identical labelled references, development-only adaptation settings,
and validated execution on Mega2560 and ESP32-C3. Sensor accuracy and drift
are synthetic. Physical boards supply execution times, not physical sensor
calibration results.

## Read and reproduce

- `main.tex`: seven-page IEEE journal-style draft, seven numbered scientific
  figures, a graphical abstract, two tables, and twelve references.
- `supplement.tex`: full seven-model static and nine-method sequential tables,
  paired unit bootstrap intervals, complete update measurements, build sizes,
  and reproduction commands.
- `references.tex`: manually checked IEEE-style reference entries with DOI or
  primary-source links; it is intentionally editable without a BibTeX dependency.
- `figures/`: PDF vector figures and editable SVG / 300 dpi PNG exports.
  `graphical_abstract.*` is also available as a separate submission asset.
- `make_assets.py`: regenerates every figure and numerical table from the saved
  inputs. This does not rerun simulations or fabricate hardware timings.
- `data/source_hashes.json`: immutable evidence commit and SHA-256 of every
  figure input. A mismatch aborts generation.

The standalone source archive can be uploaded to Overleaf; select `main.tex`
as the main document, or `supplement.tex` to build the supplement. All numerical
figure inputs are included, so the archive also works outside the full repo.

```sh
python -m pip install -r requirements.txt
python make_assets.py
latexmk -pdf main.tex
latexmk -pdf supplement.tex
```

Python 3.12.14, NumPy 2.3.5, Matplotlib 3.10.8, pdfTeX / TeX Live 2023 and
latexmk 4.83 were used for this draft. Numerical experiment reproduction also
requires SciPy 1.17.0; see Section S7 of the supplement.

## Exact evidence

The experimental snapshot is
[`db39238c1467169b65ff1f3581945e57ee7ffd7a`](https://github.com/KuznetsovKarazin/lut-kan-native/tree/db39238c1467169b65ff1f3581945e57ee7ffd7a).
The manuscript commit adds presentation assets and leaves experiment results,
firmware, and acquisition protocols unchanged.

- [Experiment source and index](https://github.com/KuznetsovKarazin/lut-kan-native/tree/db39238c1467169b65ff1f3581945e57ee7ffd7a/experiments/sensor_recovery)
- [Static/adaptation summaries and per-unit statistics](https://github.com/KuznetsovKarazin/lut-kan-native/tree/db39238c1467169b65ff1f3581945e57ee7ffd7a/experiments/sensor_recovery/results/matched_adaptation_20261108)
- [Combined matched-board results](https://github.com/KuznetsovKarazin/lut-kan-native/blob/db39238c1467169b65ff1f3581945e57ee7ffd7a/experiments/sensor_recovery/results/matched_boards_20261108/RESULTS.md)
- [C3 raw captures](https://github.com/KuznetsovKarazin/lut-kan-native/tree/db39238c1467169b65ff1f3581945e57ee7ffd7a/experiments/sensor_recovery/results/c3_matched_20261108)
- [Mega raw captures](https://github.com/KuznetsovKarazin/lut-kan-native/tree/db39238c1467169b65ff1f3581945e57ee7ffd7a/experiments/sensor_recovery/results/mega_matched_20261108)

Large raw synthetic streams and complete trajectory states are preserved in a
separate 140,570,539-byte archive. Its file hashes and regeneration commands are
public; the archive itself is **not** in the Git tree. The manuscript does not
claim an existing public data DOI. Before submission, a public archive/release
deposit would make this part of the evidence more convenient to access.

## Journal format and author review

The draft uses the unmodified IEEEtran v1.8b journal class supplied with the
author's earlier manuscript: 10 pt, two columns, US letter. It includes a
150–250-word single-paragraph abstract, alphabetical index terms, numbered
figures/tables/equations, an integrated graphical abstract, and linked evidence.
There are no invented received/accepted dates, journal volume, or article DOI.

Official requirements checked on 8 November 2026:

- [IEEE Sensors Journal author instructions](https://ieee-sensors.org/ieee-sensors-journal/for-authors/): double-column manuscript, normal eight-page length, required graphical abstract.
- [IEEE article templates](https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/authoring-tools-and-templates/tools-for-ieee-authors/ieee-article-templates/).
- [IEEE submission and peer-review policies, including AI-generated content](https://journals.ieeeauthorcenter.ieee.org/become-an-ieee-journal-author/publishing-ethics/guidelines-and-policies/submission-and-peer-review-policies/).

The interactive IEEE template selector did not provide a downloadable journal
package in this environment. This draft uses standard IEEEtran, not a claimed
latest journal-specific `ieeecolor` package. Check the live submission template
when preparing the final version; the source and vector assets are portable.

The acknowledgment identifies ChatGPT assistance in drafting and visualization
code, as required for substantive AI-generated content. It does not claim that
the author has already completed scientific review of this first draft.

Author review should confirm authorship/contact details and any funding statement,
assess the framing and scientific claims, and decide on a public raw-data deposit.
No coauthors or grant acknowledgments were inferred. Physical sensor accuracy,
reference uncertainty, energy, peak stack, multiple-board variation, and active
PAVA worst-case timing remain outside the evidence. The initial calibration fit
is offline; small online state does not establish a low-memory initialization
method. No further capture is needed to reproduce the completed hardware stage.

## Кратко для автора

Главный результат — зависимость выбора представления от ресурсов. При 80 байтах
LUT float32 точнее развернутого Horner во всех шести группах; при 144 и 224 байтах
порядок меняется. Это сравнение именно этих двух представлений, а не превосходство
LUT над всеми сплайнами. На реальных платах выигрыш LUT по времени относительно
Horner составляет 13.2–13.8%, при более высокой ошибке. Для обновления с PAVA
выигрыш составляет 30.7–39.1% относительно компактного кубического сплайна.

В тексте сохранены сильные физические и полиномиальные baselines, случаи
ухудшения без дрейфа и ограничения по физическим измерениям. Все рисунки построены
из числовых данных. Для подачи сначала нужна авторская научная редактура; данный
пакет — первый полноценный драфт, а не автоматически подготовленная submission.
