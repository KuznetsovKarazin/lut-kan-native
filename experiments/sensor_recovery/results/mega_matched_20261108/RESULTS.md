# Mega2560 matched inference and complete adaptation: two validated sessions

The successful Mega archive completes the matched hardware checkpoint.
At the same 144-byte model payload, LUT32 takes 167.279297 µs/call and
Horner8 takes 192.788086 µs/call: LUT time is 13.232% lower. Complete
projected LUT NLMS takes 797.156250 µs/update under step drift, versus
1149.541667 for projected cubic NLMS. The cubic models have lower error.
See the [joint Mega/C3 report and figures](../matched_boards_20261108/RESULTS.md).

## Evidence and startup recovery

Original archive: `mega_matched_20261108_104432_873.zip`, 108659 bytes,
SHA-256 `90245afc27176c7c7b66049b6d83d1bc21ef120e9f33c72e8281825873a8b2ae`.
Measured commit: `a86116f87e8c04b9cee54eb7766ecf62b9484075`.
The Mega2560 ran at 16 MHz, with one upload and two sequential acquisitions
on 8 November 2026. Times below are UTC; local Italian time was one hour later.

| Session ID | Start UTC | End UTC | Rejections before READY |
|---|---|---|---:|
| 4b8c535d310f4115a71678cedf6c28d1 | 09:44:53.701475 | 09:45:58.788536 | 1 |
| ad691e200e3540d09de10b2d6117c356 | 09:45:58.878840 | 09:47:03.998544 | 1 |

Each initial H still received `ERROR,INVALID_COMMAND`. The bounded greeting
retry recovered both sessions; each command sequence was H, H, Q, R.
The two-second delay alone did not eliminate the first rejection. No Q, R or
measurement trajectory was retried. Raw logs and sidecars retain the rejection
and timestamps. This validates the recovery on two actual acquisitions;
it does not establish the cause of the initial command rejection.

The audit passes ZIP CRCs, all 19 LF-normalized source hashes, before/after
source equality, both firmware-image hashes, upload verification, session
identity and command order. Re-parsing raw bytes reproduces the JSON results.
The ELF and HEX images match those in the earlier failed Mega capture; the
startup change is in the host capture code, not the firmware.

| Image | Bytes | SHA-256 |
|---|---:|---|
| firmware.elf | 67224 | 0f9ce921f54eae2034f49aae963b3e612d20ee72b59792ecd653d7645c7e714b |
| firmware.hex | 139391 | 3366faeb1cdfce0ad6caf8d17274b442044d5f685762d7cf06799bb5f4a27045 |

Each session contains four static accuracy rows, 50 inference/overhead timing
rows, 96 state checkpoints, 12 final accuracy rows, 120 complete update
timings and 30 update-overhead timings. Maximum reported gaps are 0.00001526 °C
for predictions, 0.00001490 °C for checkpoint states and 0.00001192 °C for
timed final states, below the predeclared 0.002 °C gate. All projected
checkpoints are monotone. Final metrics also agree with the host reference.

`audit.json` retains hashes and per-session ranges. Raw logs, parsed JSON,
sidecars, console output, package inventory and upload log are preserved here.
Only the public batch copy redacts a Windows username, with its new hash
recorded in the audit. The original archive retains unmodified metadata and
the actual images. This evidence does not cryptographically attest the device.

## Static inference

Each session median uses ten batches of 2048 calls. The summary column is the
median of the two session medians. These are gross times including fixture
reads and checksums. Overhead medians, 12.328125 and 12.327148 µs/call, remain
separate and are not subtracted. Accuracy is identical between the sessions.

| Model, all 144 B | Session 1, µs | Session 2, µs | Summary, µs | MAE, °C | RMSE, °C | Max error, °C |
|---|---:|---:|---:|---:|---:|---:|
| LUT32 float32 | 167.279297 | 167.279297 | 167.279297 | 0.0557110 | 0.0837515 | 0.4702148 |
| LUT60 int16 | 204.830078 | 204.830078 | 204.830078 | 0.0478595 | 0.0687363 | 0.3202744 |
| Cubic32, unrolled | 369.023438 | 369.022461 | 369.022949 | 0.0444038 | 0.0639407 | 0.2678528 |
| Horner8 | 192.787109 | 192.789062 | 192.788086 | 0.0455196 | 0.0658783 | 0.2676544 |

## Complete adaptation

Each repeat includes all 384 supplied references. Times include normalization,
basis evaluation, prediction, residual, NLMS, writes and any full PAVA,
including fixture reads/checksums. Reset, initial projection, validation and
serial output are excluded. The following are medians of two session medians.

| Method | Persistent B | PAVA workspace B | None, µs/update | Step, µs/update | Ramp, µs/update |
|---|---:|---:|---:|---:|---:|
| LUT NLMS | 148 | 0 | 371.427083 | 371.177083 | 371.072917 |
| LUT NLMS + PAVA | 148 | 160 | 797.401042 | 797.156250 | 797.059896 |
| Cubic NLMS | 148 | 0 | 723.997396 | 723.583333 | 723.635417 |
| Cubic NLMS + PAVA | 148 | 160 | 1149.953125 | 1149.541667 | 1149.593750 |

Matching overhead is 22.239583, 22.252604 and 22.229167 µs/update for none,
step and ramp respectively. For step drift, cubic/LUT time ratios are 1.9494
without projection and 1.4421 with projection. Projected LUT uses 30.654%
less time. The 148/160-byte counts exclude local stack and harness storage.

| Schedule | Final LUT MAE, °C | Final cubic MAE, °C |
|---|---:|---:|
| None | 0.0566184 | 0.0436144 |
| Step | 0.0589095 | 0.0446434 |
| Ramp | 0.0587070 | 0.0443521 |

Projected and unprojected results coincide on this fixture. The shared
float32 host replay finds no monotonicity violations before projection and
no PAVA block merges. These measurements cover that execution path, not
worst-case projection. Broader adaptation accuracy conclusions come from
the separate 3888-trajectory confirmation campaign.

## Scope and reproduction

The maximum relative range of the two model-inference medians is 0.0010131%;
for update methods it is 0.0013063%. These are descriptive same-board
repeatability figures, not uncertainty estimates or independent-device
replications. The uploaded full harness uses 49552 bytes Flash and 1949
bytes static RAM. Those counts are not model-only or peak-RAM measurements.

Calibration data are synthetic; execution times are measured on physical
hardware. Physical NTC accuracy, reference/ADC acquisition costs, energy,
peak stack and between-board variation remain unmeasured.

From the repository root, with the measured commit in Git history and a
fresh output directory:

```bash
python experiments/sensor_recovery/review_matched_batch.py /path/to/mega_matched_20261108_104432_873.zip --out runs/mega_matched_review_new
```

Both boards now have two complete validated matched sessions. No additional
acquisition is required to close this predefined hardware checkpoint.
