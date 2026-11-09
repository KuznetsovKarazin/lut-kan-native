# Seven-model common-fixture acquisition, 9 November 2026

All six requested sessions passed independent reparsing and the frozen protocol.
No new fitting, hyperparameter selection or synthetic trajectories were added.

| Model | Payload B | Mega us | C3 us | S3 us | MAE C | Maximum C |
|---|---:|---:|---:|---:|---:|---:|
| LUT32 float32 | 144 | 164.324219 | 11.980957 | 0.802246 | 0.0557110 | 0.4702148 |
| LUT60 int16 | 144 | 204.137207 | 15.713867 | 0.864746 | 0.0478595 | 0.3202744 |
| Compact cubic32 | 144 | 363.855469 | 28.392578 | 1.213867 | 0.0444038 | 0.2678528 |
| Piecewise Horner8 | 144 | 192.096680 | 13.982910 | 0.843750 | 0.0455196 | 0.2676544 |
| Polynomial Clenshaw | 56 | 326.235840 | 25.922485 | 1.155762 | 0.0441309 | 0.2489777 |
| Polynomial Horner | 56 | 223.661133 | 17.645508 | 0.768555 | 0.0441311 | 0.2489777 |
| Fitted physics | 32 | 311.640625 | 30.985352 | 2.212402 | 0.3298833 | 0.8232117 |

Times are medians of two session medians; each session median uses ten batches
of 2048 calls. They are gross times, without subtracting the overhead control.
Extra decimal places above support audit, not metrological accuracy. Main paper
values use three significant figures. MAE is from one fixed synthetic distorted
NTC case, seed 2026110801, unit/repetition 0, B=75. Each board executes the same
401 test inputs. Maximum error is the largest on that finite grid.

LUT32 is fastest on Mega and C3. Relative to polynomial Horner, it reduces time
by 26.53% and 32.10%, but has higher error and larger payload. Polynomial Horner
is fastest on S3: 4.20% less time, 20.79% lower MAE and 61.11% less payload than
LUT32. The same polynomial is 31.44%, 31.93% and 33.50% faster in Horner than
Clenshaw form on Mega, C3 and S3. Clenshaw's MAE is only 0.0000002 C lower after
float32 evaluation/accumulation: this is numerical rounding, not a meaningful
accuracy advantage. The strict two-axis frontiers retain both forms.

Piecewise Horner dominates LUT60 int16; polynomial Horner dominates the compact
cubic on every board in this fixed case. This does not undo the multi-unit accuracy and sensitivity
results: different response shapes, budgets and fits can change the ranking.
The physical equation is misspecified because it omits front-end distortion.

## Acquisition and audit

- Three boards: Mega2560 16 MHz; C3 160 MHz; S3 240 MHz with native USB Serial/JTAG,
  4 MB DIO flash at 80 MHz and no PSRAM. One board per architecture.
- Six unique session IDs, 480 TIME records, 42 ACCURACY records. All raw logs
  reproduce their complete stored JSON. Batch/sidecar hashes and all 24 frozen
  source identities match; source-before equals source-after. Firmware images
  are present and their recorded sizes and SHA-256 hashes match the originals.
- Maximum host/device discrepancy: 0.00005341 C. Actual captured S3 ELF opcode
  bytes match the previously disassembled native float32 instructions in all
  seven predictors; see s3_captured_elf_fpu.json and the frozen preparation.
- Mega has one INVALID_COMMAND line before READY in each session, within the
  previously tested allowance for reset/startup. No rejection occurs after
  handshake. Both complete sessions are retained; no retry was requested.
- Gross overhead medians: 11.635254, 0.698242 and 0.135254 us. These are reported
  separately and are not subtracted.
- Maximum relative range of the two model session medians: 0.00144% Mega,
  0.000942% C3 and 0% S3 at logged resolution. Raw batches do vary; identical
  medians do not establish zero timing variation or a confidence interval.

## Reproduce

Obtain the three unmodified ZIPs named in summary.json, then from repository root:

```sh
python experiments/sensor_recovery/analyze_v2_captures.py --archives /path/to/raw_archives --out reproduced_v2
```

The script validates exact archive identities, rechecks all records and image
hashes, verifies captured S3 float instruction bytes, and regenerates summary.json.
The public Git copy retains raw serial logs, parsed JSON, sidecars, package
inventories and batch records. The original complete ZIPs additionally preserve
firmware images and upload diagnostics. SHA256.json covers the imported evidence
and generated summary. The protocol was fixed in commit 755ffe0 before receipt.

## Scope and deposit

These are real MCU execution measurements on synthetic calibration inputs.
Cross-board comparisons also change ISA, clock, memory and toolchain, so no
causal FPU-only speedup is inferred. No physical NTC/LDR experiment, uncertainty
budget, energy, WCET, peak stack or board-to-board spread was measured.
The earlier update timings on Mega/C3 are unchanged; no S3 update times exist.

Zenodo 1.0.0 (10.5281/zenodo.24062383) predates these captures and does not contain
them. Preserve that version and the sensors-zenodo-v1.0.0 tag unchanged. Deposit
the new V2 evidence as a later version; no new DOI is claimed until publication.
