# Matched-payload inference and adaptation acquisition

This historical stage is complete on Mega2560 and ESP32-C3. See the
[combined results](../results/matched_boards_20261108/RESULTS.md).
The current seven-model inference comparison uses [V2](RUN_REFERENCES.txt).
Keep these fixture identities separate.

The matched suite measures four 144-byte static models and complete LUT/cubic
NLMS updates, with and without monotonicity projection. Each update trajectory
consumes 384 synthetic labelled references; initialization and validation are
outside timing. Prediction/state parity uses the fixed 0.002 C tolerance.
These measurements establish execution cost, not physical thermistor accuracy.

From a fresh repository checkout, close Serial Monitor and use the actual port:

```powershell
py -m pip install -r experiments/sensor_recovery/mcu/requirements.txt
py -m serial.tools.list_ports
py experiments/sensor_recovery/mcu/run_matched.py --board esp32c3 --port COM7 --out runs/c3_matched_new --runs 2
py experiments/sensor_recovery/mcu/run_matched.py --board mega2560 --port COM7 --out runs/mega_matched_new --runs 2
```

Run only the command for the connected board. It replaces that board's firmware,
then takes two warm captures after one upload. Use a new output name each time.
Do not press BOOT/RESET or disconnect USB during capture. The runner handles the
bounded pre-READY handshake retry documented in the transport tests.

Expected completion is `Validated 2 complete matched captures. Board may be disconnected.`
Keep the entire ZIP: raw logs, JSON, session IDs, upload diagnostics, package
versions, hashes and firmware images. Retain failed attempts with their own
identities; partial runs do not count as measurements. The earlier Windows
encoding and Mega startup failures remain in their original audit directories.

The frozen [protocol](../MATCHED_MCU_PROTOCOL.md) defines the scope and memory
accounting. LUT/cubic update state is 148 bytes; PAVA adds 160 bytes of workspace.
Neither payload nor a free-memory estimate is a measured peak-stack value.
