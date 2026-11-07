# MCU cross-compilation — 2026-11-07

Twenty successful builds: nine individual footprint applications and the full benchmark for each of Mega2560 and ESP32-C3. These are compiler/linker measurements, not execution on hardware.

## Full benchmark

| Board | Flash bytes | Static RAM bytes |
|---|---:|---:|
| mega2560 | 28990 | 1280 |
| esp32c3 | 288640 | 14004 |

The complete benchmark includes 401 test points, expected predictions, Serial logging, timing loops and the update-cost experiment. It is not a deployment footprint.

## Individual deployments

Delta = individual build minus the baseline built with the same framework and flags. Serial, test inputs and expected predictions are absent. Runtime input/output are volatile; prediction functions cannot be inlined.

| Method | Mega Flash delta B | Mega static RAM delta B | C3 Flash delta B | C3 static RAM delta B |
|---|---:|---:|---:|---:|
| Physics | 1648 | 0 | 1534 | 8 |
| Chebyshev degree 9 | 1148 | 40 | 266 | 0 |
| Cubic spline 16 | 1954 | 64 | 666 | 0 |
| KAN edge 16 + SiLU | 2522 | 68 | 1682 | 8 |
| Uniform LUT32 float32 | 1624 | 128 | 438 | 0 |
| Uniform LUT32 int16 | 1600 | 64 | 442 | 0 |
| Uniform LUT32 uint8 | 1546 | 32 | 406 | 0 |
| Nonuniform LUT16 int16 | 1610 | 96 | 530 | 0 |

Baseline sizes: Mega 730 B Flash / 17 B static RAM; C3 218848 B Flash / 13484 B static RAM. These framework baselines explain why total binaries should not be compared as if they contained only model data.

On AVR, coefficient arrays in this implementation reside in SRAM. On C3 they can remain in flash-backed read-only memory; a zero static-RAM delta does not imply zero stack or working memory. Some scalar constants are embedded in instructions. Payload bytes in fixture_manifest.json are a portable accounting convention, distinct from compiler section sizes.

Polynomial deployment is smaller than LUT32 in these builds. Whether LUT earns an execution-time advantage requires physical-board measurements. No latency, energy, stack high-water, or online-training accuracy is inferred from compilation.

## Validation

- 25 focused tests plus eight model subtests passed. They cover calibration isolation, quantization, constrained fitting, nonuniform grids, C/C++ parity, projected-update invariants and rejection of incomplete/corrupt UART captures.
- Exact C++ inference kernels match Python references within 0.002 °C, including endpoints and out-of-range inputs. This is host parity; the firmware repeats its own parity checks on the MCU.
- A two-node monotone SGD update preserves global monotonicity in 2000 random updates. A boundary regression checks projection order. This is a primitive cost benchmark, not a validated adaptive calibration procedure.
- All 20 builds completed; each ELF header was verified. ELF SHA256 and static sizes are in builds.json. Compiler/framework versions are in packages.txt; hashed sources are in source_hashes.json.
- Numerical follow-up has 4896 valid gzip records and 204 summary groups; all monotone float32/int16 runs have zero detected monotonicity violations. Numerical-run source hashes still match.
- The original core suite (159 tests) passed in the parent revision. Core learning code was not changed in this follow-up; it was not rerun.

## Next observation

Follow ../../mcu/README_RU.md: flash Mega2560 and ESP32-C3, capture two independent reset-to-END runs per board, and retain both .log and .json files. No sensors are needed. Report gross timing and overhead-adjusted estimates separately. Actual SuperMini USB/board compatibility must be confirmed by the user’s device run.
