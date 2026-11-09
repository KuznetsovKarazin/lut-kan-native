# Sensor-study history

The current IEEE Sensors Journal implementation study is in
[experiments/sensor_recovery](../experiments/sensor_recovery/README.md).
Its accuracy data are explicitly synthetic; physical boards measure execution.

The older `scripts/exp_sensor_calib_v14.py` and `results/sensor_calib_v14/`
remain as historical development code and outputs. Their earlier narrative,
figures, split and storage conventions do not support the current paper. In
particular, old wording about acquired NTC data, universal LUT-winning sensor
families and a necessary `K * L < N` rule must not be used as evidence. The
superseded note is available in Git history at d5b44de for provenance.

The revised study uses explicit label budgets, actual stored precision, strong
physical/polynomial references, selection/sensitivity checks and identified
hardware fixtures. Its published releases and immutable commits retain the
original records, including negative results and unsuccessful captures.
