# Historical optimized-cubic suite

The two-session campaign is complete on [C3](../results/c3_optimized_20261108/RESULTS.md)
and [Mega](../results/mega_optimized_20261108/RESULTS.md). The following procedure
is retained for reproduction. The current seven-model comparison uses a different
fixture; see [V2](RUN_REFERENCES.txt).

This suite contains ten methods, including the same cubic in an 80-byte compact
form and a 224-byte expanded Horner form. Test pairs are synthetic.

From the repository root, close Serial Monitor and replace COM7 with the actual port:

```powershell
py -m pip install -r experiments/sensor_recovery/mcu/requirements.txt
py -m serial.tools.list_ports
py experiments/sensor_recovery/mcu/run_suite.py --board esp32c3 --port COM7 --out runs/c3_opt_new --runs 2
py experiments/sensor_recovery/mcu/run_suite.py --board mega2560 --port COM7 --out runs/mega_opt_new --runs 2
```

Run only the connected board's command. Firmware is replaced once, followed by
two warm sessions. Do not reset or disconnect during the benchmark. Use fresh
output names and preserve every complete or failed ZIP. A complete run includes
`SUITE,spline_optimized_v1`, ten passing accuracy records and `END_SENSOR_BENCH`.
The final message is `Validated 2 complete ten-model captures. Board may be disconnected.`

These historical ZIPs preserve source/image hashes, not the firmware images
included in the later V2 archives. Their assurance level must not be conflated.
An identical session median is not evidence of independent boards or zero jitter.
