# Second-review paired refinement — 8 November 2026

This is controlled numerical reanalysis, not new physical sensor data or MCU timing.
`REFINEMENT_PROTOCOL.md` was frozen locally before the calculations. Seeds and
reference streams are reused from the first revision; the confirmation sample is
not an untouched new test set after the methodological review.

- Development: 144 streams, 1440 paths, tau in {32,128,512,1024,2048}.
- Confirmation: 288 streams, 1728 paths: two decaying methods, two float32 block
  OLS corrections and two float64-accumulation audits.
- Both decay schedules select 2048, still at the boundary. Constant-step NLMS
  and the historical 512 results are preserved in the first-revision archive.
- Block OLS uses ALL 32 reference pairs and four normalized float32 sums (16 B),
  plus float32 gain/offset (8 B), a counter and the frozen model. No degenerate
  block occurred; independent least-squares and causal perturbation tests pass.
- Distorted NTC, B=75, cubic OLS: final MAE 0.0484012 C without drift and
  0.0729765 C after both step and ramp; integrated 0.0965766 C for step and
  0.0691763 C for ramp. The stronger baseline narrows the NLMS advantage.
- V2 common MCU fixture adds power-basis Horner evaluation of the SAME fitted
  degree-9 polynomial, alongside Clenshaw (56 B each), physics and four matched
  representations. Three target builds and host parity pass. New Mega/C3
  acquisitions are required to construct a single frontier; no new captures
  or S3/FPU timings are present in this directory.

Reproduce from experiments/sensor_recovery, creating fresh output directories:

```sh
python refinement_analysis.py development --out results/refinement_new/development
python refinement_analysis.py confirmation --out results/refinement_new/confirmation --selection results/refinement_new/development/selection.json
python audit_refinement.py --results results/refinement_new
python -m pytest test_*.py -q
```

Run environment: Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0. The raw gzip streams
are preserved in IEEE_Sensors_Refinement_Evidence_2026-11-08.zip, separately from
Git. run_info.json records source/output hashes; raw_manifest.json covers archived
results. A gzip container timestamp need not match on reproduction: compare
decoded records. Original first-revision inputs and results are unchanged.
