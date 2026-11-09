# Post-review analyses, 8 November 2026

These are additional host simulations and a re-audit of existing physical MCU logs. No new board or physical sensor readings are included. The locally frozen protocol and dated addendum distinguish planned analyses from the subsequent discovery of earlier small-reference timings.

- `audit/`: all 432 original streams, 331,776 projected updates; exact LUT/cubic projection activity.
- `static/`: 608 nominal/stress cases and 3,488 fitted models; fivefold versus original holdout selection, exact-generator noisy inverse, one-factor sensitivity.
- `development/`: 144 streams, 2,304 candidate trajectories; decaying NLMS and extended forgetting-factor grid. `selected_parameters.json` fixes confirmation settings.
- `confirmation/`: 288 new-seed streams, 5,472 trajectories, 19 methods, no numerical failures.
- `summary.json` and `curves.json`: aggregate and per-unit results, retained negative results, finite-grid maxima, checkpoint curves.
- `recovered_hardware.json`: the earlier pilot fixture, independently reparsed from four raw MCU logs, including timed polynomial and physical baselines. Never merge these points with the later matched fixture into one frontier.
- `builds/`: successful cross-compilations for the prospective six-model Mega2560, ESP32-C3 and ESP32-S3 suite. Compilation and host parity checks are not new hardware timings.
- `raw_manifest.json`: SHA-256 and sizes of the three large `records.jsonl.gz` streams, preserved in the separate revision evidence ZIP and reproducible using the protocol. They are excluded from Git. No public archive DOI is claimed.

From repository root, use Python 3.12 with NumPy 2.3.5 and SciPy 1.17.0, and fresh output directories:

```sh
python experiments/sensor_recovery/revision_analysis.py audit --out reproduced_revision/audit
python experiments/sensor_recovery/revision_analysis.py static --out reproduced_revision/static
python experiments/sensor_recovery/revision_analysis.py development --out reproduced_revision/development
python experiments/sensor_recovery/revision_analysis.py confirmation --out reproduced_revision/confirmation --selection reproduced_revision/development/selected_parameters.json
python experiments/sensor_recovery/revision_analysis.py summarize --out reproduced_revision
python experiments/sensor_recovery/recover_reference_timings.py
python -m pytest experiments/sensor_recovery/test_*.py -q
```

The recovery command rewrites only `recovered_hardware.json` at this results path from archived raw logs. It does not acquire data. Raw gzip timestamps may differ on regeneration; compare decoded records. The old algorithms' full covariance histories are regenerated from the saved streams; not every original-method internal update state is serialized in the revision records.

Physical sensor uncertainty, energy, active-projection worst-case timing, measured stack, and between-board variation remain outside this evidence. See the manuscript supplement for a prospective physical validation protocol. The ESP32-S3 run is optional; the two original boards already provide measured polynomial/physical baselines on the earlier fixture.
