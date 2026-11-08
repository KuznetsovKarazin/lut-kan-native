# Equal-memory inference and complete adaptation measured on ESP32-C3

Two device-acknowledged C3 sessions pass the frozen matched-MCU protocol.
At the same 144-byte payload, LUT32 takes 12.051025 µs/call and Horner8 takes
13.984131: LUT time is 13.824% lower, while Horner is more accurate. Full
projected LUT NLMS takes about 43.78 µs/update versus 71.86 for projected
cubic NLMS, again with higher error. These are computational measurements
on the predetermined synthetic fixture, not physical NTC accuracy.

## Evidence and validation

Original archive: `c3_matched_20261108_101534_200.zip`, 2,217,390 bytes,
SHA-256 `53ceb2ffa59b2a626647d7880c3453fa72168bb9b043b2369f777ef746657b54`.
Measured commit: `12c6415467b3138c0af7e5b49828a01c0f1066ec`; its logging fix
does not change firmware from 98452eb. ESP32-C3 runs at 160 MHz with USB
Serial/JTAG, one upload and two sequential acquisitions on 8 November 2026:

| Session ID | Start UTC | End UTC |
|---|---|---|
| 646fff2706d34a228f77cc5fc0b4348d | 09:15:47.106806 | 09:15:52.346202 |
| 62438f47b06a4c158d6e6d17d7861d74 | 09:15:52.461845 | 09:15:57.712062 |

All ZIP CRCs, 19 LF-normalized source hashes and four image hashes pass.
Before/after sources match. Upload succeeds with four flash-data hash
verifications; package inventory now completes. H/Q/R commands, device
ECHO/ACCEPTED/SESSION, sidecars, timestamps, byte lengths and hashes agree.
Raw re-parsing reproduces both JSONs, and metrics independently match the
saved host reference within the predeclared gate.

Each session contains four static accuracy rows, 50 inference/overhead timing
batches, 96 checkpoints, 12 final accuracy rows, 120 timed update trajectories
and 30 update-overhead batches. Maximum reported gaps are 0.00001526 °C for
predictions, 0.00001490 °C for checkpoint states and 0.00001192 °C for timed
final states, all below 0.002 °C. All projected checkpoints are monotone.

`audit.json` retains entry hashes, source/image checks and per-session timing
ranges. Raw/parsed/session/console evidence and packages are preserved here.
Public batch/upload copies redact the device MAC and host username; redacted
hashes are recorded. Original unredacted bytes and actual firmware images
remain in the submitted ZIP. This does not cryptographically attest the device.

## Static inference: 144 bytes for every representation

Times are within-session medians of ten batches of 2048 calls. Both sessions
have the same medians and error values at recorded precision. Gross time
includes fixture reads and checksums. Overhead is 0.699707 µs/call, retained
separately rather than subtracted.

| Representation | µs/call | MAE, °C | RMSE, °C | Maximum error, °C |
|---|---:|---:|---:|---:|
| LUT32 float32 | 12.051025 | 0.0557110 | 0.0837515 | 0.4702148 |
| LUT60 int16 | 15.714844 | 0.0478595 | 0.0687363 | 0.3202744 |
| Compact cubic32, unrolled | 28.543457 | 0.0444039 | 0.0639407 | 0.2678528 |
| Eight-piece Horner | 13.984131 | 0.0455196 | 0.0658783 | 0.2676544 |

LUT32 is fastest; its MAE exceeds Horner8 by 0.0101914 °C (22.39%). The denser
quantized LUT improves its error but takes longer than Horner8 here. Compact
cubic32 gives the lowest MAE. This supports a speed/accuracy trade-off, not
universal LUT superiority. Models use the same label/payload constraints but
represent different fitted functions. Earlier LUT144/Horner224 captures used
a different fixture; their absolute times are not a before/after comparison.

## Complete adaptation

Each timing repeat processes 384 supplied references; the median of ten full
trajectory times is divided by 384. Normalization, basis, prediction, residual,
NLMS, coefficient writes and any full PAVA are included. Reset, initial
projection, validation and serial output are excluded. Fixture reads/checksums
are included; their matching overhead is 1.208333 µs/update for every schedule.

| Method | Persistent bytes | PAVA workspace bytes | None, µs/update | Step, µs/update | Ramp, µs/update |
|---|---:|---:|---:|---:|---:|
| LUT NLMS | 148 | 0 | 24.611979 | 24.614583 | 24.617188 |
| LUT NLMS + PAVA | 148 | 160 | 43.776042 | 43.781250 | 43.781250 |
| Cubic NLMS | 148 | 0 | 52.656250 | 52.664062 | 52.666667 |
| Cubic NLMS + PAVA | 148 | 160 | 71.820312 | 71.861979 | 71.856771 |

For the step trajectory, cubic/LUT time ratios are 2.1395 without projection
and 1.6414 with projection; projected LUT time is 39.076% lower. These memory
counts omit local stack scratch and the harness; they are not peak-RAM totals.

Final MAEs use each schedule's separate 401-point noisy ADC grid. Frozen
baselines are host replays of the same float32 kernel; adapted values come
from the C3. This grid differs from the static test grid above.

| Schedule | Frozen LUT, °C (host) | Adapted LUT, °C (C3) | Frozen cubic, °C (host) | Adapted cubic, °C (C3) |
|---|---:|---:|---:|---:|
| None | 0.0566563 | 0.0566184 | 0.0474990 | 0.0436144 |
| Step | 0.5663979 | 0.0589095 | 0.5662535 | 0.0446434 |
| Ramp | 0.5663979 | 0.0587070 | 0.5662535 | 0.0443521 |

Projected/unprojected errors coincide here. Host replay of every float32 update
finds no monotonicity violations in the unprojected trajectories and identical
projected/unprojected states throughout. Full PAVA executes without requiring
block merges. Its measured time therefore describes that path, not a merging-
heavy or worst-case workload. Broader error conclusions remain based on the
separate 3888-trajectory campaign, where projected cubic wins all 18 groups.

For the step case, adaptation reduces LUT MAE by 89.60% and cubic MAE by
92.12% relative to their frozen baselines. This requires labelled references
across the range; it is not autonomous calibration without reference values.

## Repeatability, scope and continuation

All session medians agree, but three inference timing rows, 25 update timing
rows and one overhead row differ. The records are not duplicates. Accuracy
and state checks agree. These are two warm sessions on one board, not two
independent devices; identical medians do not imply zero timing uncertainty.

The Windows full harness reports 282832 bytes Flash and 14252 bytes static
RAM. These are harness sizes. Physical NTC accuracy, real ADC/reference
acquisition costs, energy, peak stack, board/cold-boot variation and matched
Mega times remain unmeasured. Seven new review tests include rejection of a
forged passing MAE with recomputed hashes. The focused suite passes 101 tests
and eight model subtests.

Reproduce from the repository root with the original archive and measured
commit available in Git history; output paths must be new:

```bash
python experiments/sensor_recovery/review_matched_batch.py /path/to/c3_matched_20261108_101534_200.zip --out runs/c3_matched_review_new
python experiments/sensor_recovery/analyze_matched_projection.py --out runs/c3_matched_review_new/projection_activity.json
```

The review uses the standard library; projection replay needs NumPy and g++.
`comparison.json` records ratios and frozen/adapted comparisons. Next, run
the same frozen suite twice on Mega2560 using [the instructions](../../mcu/MATCHED_RUN_RU.md).
