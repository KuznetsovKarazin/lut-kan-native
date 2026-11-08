# Frozen-function optimized cubic deployment comparison

The initial MCU captures compared LUTs with a general de Boor evaluator.
This follow-up measures implementation cost while holding the learned
functions, calibration budget and evaluation fixture fixed. It introduces
no new training, coefficient tuning, sensor case or test-set selection.

## Representations

| Method | Learned coefficients | Stored floats | Payload including four transforms |
| --- | ---: | ---: | ---: |
| Original cubic16 | 16 | 16 | 80 B |
| Unrolled cubic16 | Same 16 | 16 | 80 B |
| Horner cubic16 | Same 16, compiled into 13 pieces | 52 | 224 B |
| Existing LUT32/float32 | 32 | 32 | 144 B |

The unrolled evaluator uses six de Boor interpolations in integer knot
coordinates, with explicit repeated-knot boundary cases and no knot divisions.
The Horner exporter reads the already rounded float32 coefficients in the
frozen header, constructs local power-basis coefficients in float64 through
SciPy PPoly, then stores float32 coefficients for t in [0,1]. It does not
fit a new curve. The original fixture/header is not regenerated.

## Checks and admission

Retain the existing 0.002 °C maximum host-gap gate on all 1005 saved parity
points and the unchanged 401-point device fixture. Test unit basis vectors,
arbitrary control vectors, both clamps and adjacent floating-point values
around every interval boundary. Compare an expanded grid against the exact
function defined by the frozen float32 coefficients and transforms; this is
an additional numerical diagnostic, not a replacement for the release gate.

During implementation, equivalent float32 rearrangements of the scalar KAN
model failed the existing gate. Its spline/SiLU terms have large opposing
magnitudes. Those two candidates are retained behind
`SENSOR_OPTIMIZED_DIAGNOSTICS` for host diagnostics only; they are excluded
from the MCU suite. Their numerical failures are reported with the timing
pilot. The original scalar KAN remains as a comparator. No relaxed admission
tolerance or coefficient refit is used to rescue the failed candidates.

## Host execution

Compile the same C++ kernels with g++ at -O2 and -Os, with floating-point
contraction disabled. Use 8192 warm-up calls per method, then 15 batches of
262144 calls in rotated method order on the fixed 401-point input array.
Keep function dispatch, input reads and checksum accumulation in the timed
region. Keep output printing outside it. Retain every raw batch and compiler
setting. Three separate process runs per profile check the close LUT/Horner
comparison; host scheduling is not controlled, so report the range of process
medians and avoid interpreting a small difference as a decisive advantage.
These numbers are not estimates of AVR or ESP32 execution time.

## MCU execution and sizes

The optional `mega2560_opt` and `esp32c3_opt` environments retain the original
eight methods and append `cubic16_unrolled` and `cubic16_horner`. The raw log
declares `SUITE,spline_optimized_v1`; the parser requires all ten models and
all batches. The default eight-model environments remain available.

Retain the same clocks, normalization, checksums, 10 × 2048 timing calls,
401-point numerical checks and ten projected-update batches. Build separate
footprint-only baseline, polynomial, original cubic, both LUT32 formats and
both new cubics, followed by the full harness on each board. Report absolute
Flash/static RAM and baseline deltas; parameter payload and peak RAM are
different quantities. Compiler-folded transforms may not appear as data.

Capture two identified command sessions on C3 after one upload, without
disconnecting between them. `mcu/run_suite.py` keeps the upload log, source
hashes before/after, compiled image hashes, capture sidecars and a single ZIP.
It stops on failure and never retries a started measurement automatically.
This records an upload/capture chain; the device protocol still does not
cryptographically attest its flashed ELF. Mega uses its usual serial-open
reset and needs no cable reconnection between captures.

## Interpretation

Compare latency, model payload, compiled Flash/RAM and error jointly. A large
speedup against a general evaluator alone cannot support a universal LUT
speed claim. Exact evaluation optimizations do not establish an advantage
in calibration sample efficiency, online-training quality or physical NTC
accuracy; those remain separate experiments.

Primary conversion reference: [SciPy PPoly.from_spline](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PPoly.from_spline.html).
