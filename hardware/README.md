# hardware/

Benchmark scripts for flashing `lut_kan_hw_bench` to real MCU hardware.

Supported devices:
- **Arduino Mega 2560** (ATmega2560, 8-bit AVR, 16 MHz, 8 KB SRAM)
- **ESP32-C3 SuperMini** (RISC-V, 160 MHz, 400 KB SRAM)

---

## Quick start (PowerShell — recommended)

```powershell
# 1. Clone the repo
git clone https://github.com/<you>/lut-kan-native.git
cd lut-kan-native

# 2. Connect your device via USB

# 3. Flash and open serial monitor (auto-installs arduino-cli on first run)
.\hardware\flash.ps1 -Monitor

# Flash a specific device
.\hardware\flash.ps1 -Device mega    -Monitor
.\hardware\flash.ps1 -Device esp32c3 -Monitor

# Flash both (script will prompt between devices)
.\hardware\flash.ps1 -Device both -Monitor

# Explicit port
.\hardware\flash.ps1 -Device mega -Port COM3 -Monitor

# Skip install step (if arduino-cli already set up)
.\hardware\flash.ps1 -SkipInstall -Monitor
```

First run downloads arduino-cli and installs board cores (~300 MB for ESP32).
Subsequent runs use the cached installation and are fast (~15 s).

---

## Alternative: PlatformIO

```bash
pip install platformio
cd hardware

# Flash Arduino Mega
pio run -e mega2560 -t upload
pio device monitor -b 115200

# Flash ESP32-C3
pio run -e esp32c3 -t upload
pio device monitor -b 115200
```

---

## File structure

```
hardware/
├── flash.ps1                        ← main PowerShell script
├── platformio.ini                   ← PlatformIO alternative
├── README.md                        ← this file
└── lut_kan_hw_bench/
    └── lut_kan_hw_bench.ino         ← the sketch (same name as folder — Arduino requirement)
```

---

## Expected output

After flashing and opening the serial monitor at **115200 baud**:

```
========================================
  lut-kan-native hardware benchmark
  target: tanh(4x) + 0.15x (saturating)
========================================
Platform: ESP32-C3 (RISC-V 160 MHz, 400 KB SRAM)
Free SRAM at start: 312448 bytes
K=4  N_TRAIN=200  EPOCHS=800  LAMBDA2=1.0

── Inference latency (1000 calls each) ──────────────────
  LUT  K=4,L=32  1000 calls: 1840 µs
  Poly deg=16    1000 calls: 9120 µs
  Speed ratio (poly/lut): 4.96×

── On-device training sweep ─────────────────────────────

  ► K=4, L=8  (32 bytes uint8, 6.2 pts/cell)
    ep 0  val_mse=0.6821200  test_mse=0.5932100
    ep 100 val_mse=0.0002341 test_mse=0.0001508
    ...
    ── Results ──────────────────────────────────────────
    Training time: 38 s
    Best epoch: 228
    Direct-LUT MSE: 0.00015083
    Post-LUT   MSE: 0.00372010
    Ratio (post/direct): 24.7×
    Sim expected: ~25× (sim: 23× at n=500)
```

### Reference values (from PC simulation)

| Config | Post-LUT MSE | Expected direct-LUT MSE | Expected ratio |
|--------|-------------|------------------------|----------------|
| K=4, L=8  | 3.72e-3 | ~1.5e-4 | ~25×   |
| K=4, L=16 | 9.73e-4 | ~1.2e-6 | ~800×  |
| K=4, L=32 | 2.46e-4 | ~1.2e-8 | ~20000× |

Hardware results may differ by 2–3× due to soft-float arithmetic differences
and SGD stochasticity, but the order of magnitude should match.

---

## Validate before flashing (PC)

```bash
cd lut-kan-native
pip install torch numpy   # if not already installed
python hardware/validate_before_flash.py
```

This runs the exact same experiment in PyTorch on your PC and prints
expected values to compare against hardware output.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `avrdude: timeout` (Mega) | Wrong COM port. Check Device Manager. |
| `Failed to connect to ESP32` | Hold BOOT → press RESET → release BOOT → re-run script |
| Serial Monitor shows garbage | Set baud rate to 115200 |
| Serial Monitor empty (ESP32-C3) | Ensure `CDCOnBoot=cdc` in FQBN (already set in script) |
| Script can't find arduino-cli | Remove `-SkipInstall` flag |
| `execution of scripts is disabled` | Run: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |

---

## ESP32-C3 boot mode (if upload fails)

The ESP32-C3 SuperMini sometimes needs manual boot mode entry:

1. Hold the **BOOT** button (small button near USB connector)
2. Press and release **RESET**
3. Release **BOOT**
4. Re-run `flash.ps1` — upload should now succeed
