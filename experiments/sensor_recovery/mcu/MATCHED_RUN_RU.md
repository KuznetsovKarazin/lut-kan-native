# Новый аппаратный прогон: одинаковая память и полная адаптация

Два новых прогона C3 прошли полный аудит: LUT32 — 12.051 мкс против Horner8 —
13.984 мкс при одинаковых 144 байтах. Полный NLMS + PAVA: 43.781 мкс для LUT
против 71.862 мкс для cubic на скачке дрейфа. Сплайн точнее.
Следующий шаг — **Mega2560, раздел 3**. Используется та же прошивка/методика;
успешный C3 повторять не нужно. Физические датчики не нужны, обмен идёт по USB.
101 профильный тест + 8 под-тестов моделей прошли; 20 сборок подготовлены.

## Что измеряется

- Четыре статические модели по 144 байта: LUT32 float32, LUT60 int16,
  компактный cubic32 и Horner8. У каждого метода десять временных повторений.
- Полное NLMS-обновление LUT/cubic, с проекцией на монотонность и без неё.
  По 384 опорных измерения, три сценария дрейфа, десять повторений траектории.
- Реальная арифметика float32, состояние 148 байт; у проекции ещё 160 байт
  рабочей памяти. Инициализация и проверка точности исключены из времени
  обновления; чтение числовых данных и накопление контрольной суммы включены.
- Каждая плата подтверждает номер сессии. Все контрольные состояния должны
  пройти численный порог 0.002 °C. Это вычислительные тесты с синтетическими
  опорными значениями, а не измерение точности физического термистора.

## 1. Подготовить отдельную рабочую папку

В PowerShell из вашей папки репозитория `lut-kan-native`:

```powershell
git fetch origin research/sensor-calibration-recovery
$work = "..\sensors_matched_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
git worktree add --detach "$work" origin/research/sensor-calibration-recovery
Set-Location "$work"
py -m pip install -r experiments/sensor_recovery/mcu/requirements.txt
py -m platformio device list
```

Отдельная папка позволяет оставить другие эксперименты и их ветки на месте.
Если какая-либо команда завершилась ошибкой, следующие команды не запускать.
Для загрузки и захвата не нужны NumPy, SciPy или PyTorch: файлы моделей уже
подготовлены. Нужны Git, Python и указанные зависимости PlatformIO.

## 2. ESP32-C3

Первый архив `c3_matched_20261108_095740_558.zip` содержит успешную прошивку,
но не содержит измерений: список пакетов PlatformIO завершился ошибкой
`UnicodeEncodeError` в Windows CP1251. Исправленный запуск задаёт UTF-8
дочерним процессам и сохраняет Unicode в журнале даже при старой кодировке
консоли. Прошивка, модели и протокол измерений при этом не менялись.

Если рабочая папка уже создана предыдущей инструкцией, перед повтором
обновить её из этой папки:

```powershell
git fetch origin research/sensor-calibration-recovery
git switch --detach origin/research/sensor-calibration-recovery
```

При ошибке Git остановиться; не добавлять `--force`. Существующий ZIP
сохранить, а новый запуск выполнять с новым именем каталога ниже.

Подключить C3 и закрыть Serial Monitor. Посмотреть её COM-порт в списке выше
(раньше использовался COM11, но номер может измениться):

```powershell
$port = Read-Host "COM-порт ESP32-C3"
$batch = "runs/c3_matched_$(Get-Date -Format 'yyyyMMdd_HHmmss_fff')"
py experiments/sensor_recovery/mcu/run_matched.py --board esp32c3 --port "$port" --out "$batch" --runs 2
```

Скрипт собирает и прошивает один раз, затем последовательно выполняет два
прогона. Во время захвата не нажимать BOOT/RESET, не вынимать USB и не открывать
Serial Monitor. Между двумя прогонами переподключение не требуется.

Ожидаются `READY,SENSOR_MATCHED_V1,esp32c3`, `ECHO`, `ACCEPTED`, затем
`SUITE,matched_adaptation_v1`. После каждого прогона — `END_SENSOR_BENCH`.

## 3. Mega2560

После C3 подключить Mega, посмотреть её COM-порт и выполнить:

```powershell
py -m platformio device list
$port = Read-Host "COM-порт Mega2560"
$batch = "runs/mega_matched_$(Get-Date -Format 'yyyyMMdd_HHmmss_fff')"
py experiments/sensor_recovery/mcu/run_matched.py --board mega2560 --port "$port" --out "$batch" --runs 2
```

Новая Mega-прошивка ждёт команду компьютера и возвращает свой номер сессии.
Правила ручного RESET из старого прогона здесь не нужны. Ожидается
`READY,SENSOR_MATCHED_V1,mega2560`; дальше порядок тот же.

## 4. Что передать для анализа

Успешный итог каждой команды:

```text
Validated 2 complete matched captures. Board may be disconnected.
```

Передать два созданных ZIP: `c3_matched_....zip` и `mega_matched_....zip`.
Каждый содержит исходные логи, JSON с результатами, номера и время сессий,
журнал загрузки, версии пакетов, хеши исходников и **сами собранные прошивки**.
Скрипт повторно разбирает логи и проверяет совпадение JSON перед подтверждением.

Если выполнение остановилось, передать полученный ZIP целиком: частичные
результаты и причина ошибки сохраняются. Повторный запуск — с новым именем
каталога. Не изменять вручную строки FAIL и не объединять логи разных запусков.

## Воспроизведение подготовки без плат

```powershell
py experiments/sensor_recovery/export_matched_mcu.py
py -m pytest experiments/sensor_recovery/test_*.py -q
py experiments/sensor_recovery/mcu/build_matched.py --out runs/matched_build_new
```

Для этих проверок нужны NumPy/SciPy/pytest и g++; это отдельная задача от
обычной прошивки/захвата. Полный [протокол](../MATCHED_MCU_PROTOCOL.md) и
[отчёт подготовки](../results/matched_mcu_20261108/RESULTS.md) описывают границы
сравнения. Предыдущая серия LUT144 против Horner224 остаётся отдельным этапом.
