# Changelog

## v0.14.0 (2026-04-25)

### Новые датчики (`src/lut_native/sensors.py`)
- `TypeKThermocoupleUnit` — термопара Type K на основе NIST ITS-90
  - Полная полиномиальная модель с экспоненциальным членом коррекции
  - Вариация коэффициента Зеебека ±0.15% (IEC 60584-1, класс 1)
  - Точность NIST: отклонение от справочных значений < 0.001 мВ
- `LDRSensorUnit` — фоторезистор GL55-series
  - Степенная логарифмическая характеристика R = R₁₀·(10/E)^γ
  - Вариация: R₁₀ ±30%, γ ±10% (из datasheet GL5516)

### Новые baselines (`src/lut_native/sensors.py`)
- `fit_shh_baseline(..., use_noisy=True/False)` — честное сравнение Steinhart-Hart:
  - `use_noisy=False`: стандартный SHH по 3 идеальным точкам (лит. baseline)
  - `use_noisy=True`: SHH по тем же 50 зашумлённым точкам, что и LUT (справедливое)
- `fit_onepoint_offset_baseline()` — однонаправленный offset (factory poly + сдвиг)
  - Представляет минимальную полевую калибровку, используемую на практике

### Новые эксперименты
- `scripts/exp_sensor_calib_v14.py` — полный эксперимент v14:
  - Эксп. A: расширенное сравнение NTC (5 методов × 8 единиц)
  - Эксп. B: 6 типов датчиков в режимной таблице
  - Эксп. C: sweep по N_cal (NTC, TypeK, LDR)
  - Эксп. D: sweep по размеру L таблицы (NTC)
  - 5 новых рисунков автоматически

### Исправления
- **КРИТИЧНО**: единицы задержки ns → μs в документации и статье
  - Реальные значения: ~80 μs/call на AVR, ~5 μs на ESP32-C3 (K=1,L=32)
  - Speedup: 3.3× (не 6.25×)
- `docs/M_HARDWARE_FINDINGS.md` — исправлены цифры задержки

### Реорганизация
- `ntc_run.py` перемещён из корня в `scripts/ntc_run.py`
- `CHANGELOG.md` введён (вместо README_V14_CHANGES.md)
- `results/sensor_calib_v14/` создан с README
- `tests/test_sensor_v14.py` — новые тесты

## v0.13.0 (2026-04-24)
- Multi-sensor calibration study (NTC, MQ, pH)
- Coverage rule K×L < n_train validated
- Submitted as IEEE TIM v13

## v0.12.0 (2026-04-21)
- On-device training: SGD + fp16, zero init
- Hardware validation on AVR and RISC-V
- L-sweep analysis (M7d)

## v0.11.0 (2026-04-20)
- KAN2 multi-edge experiments (H4)
- Effective rank analysis (H2)
