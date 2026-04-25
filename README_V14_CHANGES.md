# v14 Changes

## New experiments
- `scripts/exp_sensor_calib_v14.py` — full v14 experiment (run this first)

## New sensor models (`src/lut_native/sensors.py`)
- `TypeKThermocoupleUnit` + `make_typek_units()` + `typek_calibration_data()`
  NIST ITS-90 polynomial model; Seebeck variation ±0.15%
- `LDRSensorUnit` + `make_ldr_units()` + `ldr_calibration_data()`
  GL55-series power-law R(lux); gamma ±10%, R10 ±30%

## New baselines (`src/lut_native/sensors.py`)
- `fit_shh_baseline(..., use_noisy=True/False)` — SHH with 3 ideal OR 50 noisy points
- `fit_onepoint_offset_baseline()` — single-point offset (factory poly + constant shift)

## Bug fixes
- Hardware latency units corrected: ns → μs everywhere
- Speedup corrected: 3.3× (not 6.25×)

## Documentation
- `docs/M_SENSOR_FINDINGS_v14.md` — updated sensor study results
- `docs/M_HARDWARE_FINDINGS.md` — corrected latency numbers

## How to reproduce
```bash
pip install -e .
python scripts/exp_sensor_calib_v14.py
# Results written to results/sensor_calib_v14/
```
