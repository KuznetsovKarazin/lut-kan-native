# Сравнение LUT с оптимизированным кубическим сплайном

В прошивке десять методов: прежние восемь и два варианта вычисления того же
кубического сплайна. `cubic16_unrolled` сохраняет 80 байт данных модели;
`cubic16_horner` использует 224 байта. Данные, обученные модели и число
вызовов прежние. Это вычислительный тест; датчики подключать не нужно.

## ESP32-C3: одна загрузка и два прогона подряд

Подключить C3 один раз. Закрыть Serial Monitor. В той же папке проекта:

```powershell
git switch research/sensor-calibration-recovery
git pull --ff-only
py -m platformio device list
$batch = "runs/c3_opt_$(Get-Date -Format 'yyyyMMdd_HHmmss_fff')"
py experiments/sensor_recovery/mcu/run_suite.py --board esp32c3 --port COM11 --out "$batch" --runs 2
```

Если порт изменился, заменить COM11 на порт C3 из списка. Скрипт собирает и
прошивает `esp32c3_opt`, ждёт возвращения порта, выполняет два захвата и
создаёт `$batch.zip`. **Между прогонами кабель не трогать, RESET и BOOT не
нажимать.** Повторной прошивки между захватами нет.

В каждом захвате должны быть `READY,SENSOR_USB_V3,esp32c3`, `ECHO`, `ACCEPTED`,
затем `SUITE,spline_optimized_v1`, десять строк `ACCURACY,...,PASS` и
`END_SENSOR_BENCH`. Итоговая строка скрипта:

```text
Validated 2 complete ten-model captures. Board may be disconnected.
```

Передать для анализа созданный ZIP. Он содержит шесть файлов результатов,
логи консоли и загрузки, а также `batch.json` с версиями исходников, хешами
собранных образов и идентификаторами сессий. Бинарные образы в ZIP не включены.
Если возникла ошибка, скрипт остановится и сохранит ZIP с частичными данными;
передать его целиком. Для нового запуска потребуется новое имя каталога.

USB должен возвращаться после автоматического перезапуска загрузчиком.
Если скрипт сообщает, что COM11 не появился, повторно посмотреть список
портов. Регулярно вынимать и вставлять кабель для этого сценария не требуется.

## Mega 2560

После результатов C3 аналогичный контроль можно выполнить на Mega. Подставить
её фактический порт вместо COM5:

```powershell
$batch = "runs/mega_opt_$(Get-Date -Format 'yyyyMMdd_HHmmss_fff')"
py experiments/sensor_recovery/mcu/run_suite.py --board mega2560 --port COM5 --out "$batch" --runs 2
```

На Mega открытие Serial обычно автоматически перезапускает прошивку. Если
заголовок не появился за пять секунд, допускается одно нажатие RESET до
начала измерений, как в прежнем протоколе Mega. Во время измерений RESET не
нажимать. Между прогонами кабель остаётся подключённым.

## Воспроизведение без плат

```powershell
py experiments/sensor_recovery/export_optimized_spline.py
py experiments/sensor_recovery/run_optimized_spline.py --out runs/optimized_host_new
py experiments/sensor_recovery/mcu/build_audit.py --out runs/optimized_build_new --boards mega2560_opt esp32c3_opt --models 0 2 3 5 6 9 10
```

Эти команды требуют научных Python-зависимостей и, для хостового пилота,
компилятора g++. Для обычной загрузки/захвата достаточно прежних зависимостей
из `mcu/requirements.txt`. Хостовые наносекунды не заменяют измерения MCU.
