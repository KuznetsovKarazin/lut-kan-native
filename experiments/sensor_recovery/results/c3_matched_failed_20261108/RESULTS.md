# First matched C3 attempt: upload passed; acquisition did not start

The supplied `c3_matched_20261108_095740_558.zip` contains **no benchmark
captures**. Upload completed successfully, then `platformio pkg list` raised
`UnicodeEncodeError` while writing its dependency-tree characters through
Windows CP1251. This is a host logging failure. It provides no new inference
times, adaptation times, MCU numerical checks or evidence of an MCU fault.

## Evidence

- Archive: 2,199,397 bytes, SHA-256
  `00f3d88b8b812966833e0791ff62ab7ced7b898f8e6bfc50ed1cf5d32d170ca8`.
- Seven entries; CRCs pass; no duplicate/unsafe paths or symlinks.
- Source commit: `98452eb173250aaa0f6347fa7235db5e767dcfc4`.
  All 19 LF-normalized source hashes match that commit and before/after records
  are identical. Windows raw CRLF hashes are retained in `audit.json`.
- Host: Windows Python 3.13.5; batch interval
  2026-11-08 08:57:42.451026–08:58:05.725364 UTC.
- `upload.log` reports success, four flash-data hash verifications and reset.
  Reported Windows build sizes: 282832 bytes Flash and 14252 bytes static RAM.
- All four archived images match their recorded sizes and SHA-256 hashes:
  firmware ELF/BIN, bootloader and partition table.
- Requested captures: two; recorded captures: zero. There are no session,
  raw benchmark or parsed benchmark files. The original runner left its
  `stage` at `upload`; the actual failed operation was package inventory.

`audit.json` records original entry hashes and individual source/image checks.
The original ZIP remains preserved with the submitted attachment. The two
logs here redact the device MAC and Windows user-directory name; original
and redacted hashes and replacement counts are recorded. Binary images are
preserved in that original ZIP and are not duplicated in this repository.

## Fix and verification

`run_suite.run_logged`, also used by the matched runner, now sets
`PYTHONIOENCODING=utf-8` and `PYTHONUTF8=1` in the child environment. A parent's
UTF-8 decoder alone cannot stop a child from failing to encode its output.
The complete UTF-8 log is written before console output; a legacy console
gets escaped unsupported characters rather than terminating the process.
Nonzero subprocess exits still fail the batch. `run_matched` now reports the
`package_inventory` stage explicitly and uses explicit UTF-8 for JSON I/O.
Firmware, fixtures, numerical gates and timing regions are unchanged.

The real PlatformIO inventory command was reproduced on Linux with CP1251
forced for redirected Python output: it failed before the fix and completed
with the fixed runner. `host_cp1251_before.log`, `host_cp1251_fixed.log` and
`host_validation.json` preserve that check. This is a host encoding test,
not a Windows hardware rerun.

Three new regression tests exercise real child/grandchild processes under
CP1251, exact preservation of Unicode logs despite a CP1251 parent console,
nonzero return-code propagation, and preservation of a failed inventory batch
without beginning acquisition. The complete focused suite passes **94 tests
and eight model subtests**. Mock images and timings stay in temporary test
directories and are never presented as device evidence.

## Next acquisition

Update the existing matched worktree using [the run instructions](../../mcu/MATCHED_RUN_RU.md),
then rerun C3 into a new output directory with `--runs 2`. Keep the failed ZIP.
Audit the new raw logs and device-acknowledged session IDs before interpreting
any timings. Mega measurements follow after the C3 result. All physical sensor
accuracy and equal-memory target timing claims remain pending.
