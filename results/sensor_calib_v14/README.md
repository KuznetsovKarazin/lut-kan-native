# results/sensor_calib_v14/

Результаты эксперимента v14 (multi-sensor calibration study).

## Генерация
```bash
python scripts/exp_sensor_calib_v14.py
```

## Ожидаемые файлы после запуска
| Файл | Описание |
|------|----------|
| summary_v14.json | Все числовые результаты |
| per_sensor_table.csv | Таблица II статьи (6 типов датчиков) |
| ntc_extended_table.csv | Таблица I (5 методов для NTC) |
| sensor_curves.png | Кривые отклика всех датчиков |
| regime_map.png | Режимная карта LUT vs Poly |
| ntc_extended.png | Сравнение 5 методов для NTC |
| ncal_sweep.png | MAE vs число точек калибровки |
| l_sweep_ntc.png | MAE vs размер таблицы L |
