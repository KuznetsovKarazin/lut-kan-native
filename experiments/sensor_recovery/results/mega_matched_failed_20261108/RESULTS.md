# Mega upload verified; first greeting rejected before acquisition

The archive `mega_matched_20261108_103108_182.zip` contains **zero benchmark
captures**. Mega was flashed successfully, but its only captured response is
`ERROR,INVALID_COMMAND` to the host's first `H` greeting. There is no READY,
Q probe, R run request, benchmark header or numerical/timing result.

## Verified evidence

- ZIP: 89501 bytes; SHA-256
  `71330e87cc040a91fc302eebb926b34f746c305db030e80d99b5a33333e8f1fc`.
  Eight entries, valid CRCs, no duplicate or unsafe paths.
- Measured source commit: `12c6415467b3138c0af7e5b49828a01c0f1066ec`.
  All 19 canonical source hashes match; before/after source records agree.
- Upload succeeds on COM12 and avrdude verifies all 49552 flash bytes.
  The harness uses 1949 bytes static RAM. ELF and HEX bytes match their
  recorded lengths and SHA-256 hashes. UTF-8 package inventory completes.
- The failed session starts at 2026-11-08 09:31:30.743279 UTC, sends `H` at
  09:31:30.781443 and ends at 09:31:30.997277 in phase `hello`.
  Raw output is exactly 23 bytes, `ERROR,INVALID_COMMAND\r\n`; its sidecar
  length/hash match. No measurement was requested.

`audit.json` records original entry hashes, source/image checks and the exact
response. Non-image evidence is retained here, with the host username redacted
from the public batch record. The original ZIP and image bytes remain intact
in the submitted attachment. No failed data are presented as measurements.

## Interpretation and host-only recovery

The log establishes a rejected startup greeting. It does not reveal the bytes
received by the MCU and does not prove a reset, bootloader or framing cause.
A transient initial UART/framing problem is a working hypothesis. The previous
capture treated that pre-READY rejection as immediately fatal.

For matched Mega only, the host now waits two seconds before its first greeting.
The read loop remains active and retains all bytes during this interval. Before
READY, it permits at most three exact `ERROR,INVALID_COMMAND` responses and
continues the existing one-second H greeting schedule within the original
15-second startup deadline. A fourth rejection, a different error, a wrong
READY or any command error after READY fails the session. Q and R are sent
only once; a measurement is never retried automatically.

Recovered rejections and timestamps are written to the session sidecar; raw
bytes are never cleared. The saved-log parser permits this bounded prefix only
for Mega, only before READY, and the archive auditor cross-checks sidecar
records. It accepts repeated H greetings but rejects repeated Q/R commands.
The default C3 behavior, firmware, fixture, numerical gates and timing regions
are unchanged. Existing C3 evidence still passes the complete archive audit.

## Checks and next step

The focused suite passes **117 tests and eight model subtests**. New fault
tests cover delayed first greeting, zero/one/three startup rejections, fatal
fourth rejection, other errors, errors in later phases, retained raw bytes,
bounded saved-log prefixes and single Q/R commands. The serial fault tests use
mock devices and artificial time; they do not supply Mega measurements or
prove that this recovery resolves the actual board's startup problem.

Update the existing worktree and rerun Mega into a fresh output directory using
[the instructions](../../mcu/MATCHED_RUN.md). C3 does not need another run.
If startup still fails, retain the complete new archive for diagnosis. The
cross-board matched comparison remains pending real Mega results.
