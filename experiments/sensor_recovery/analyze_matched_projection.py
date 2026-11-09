"""Replay frozen float32 trajectories to inspect PAVA activity, without MCU timing."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

from matched_host import compile_library, state

HERE = Path(__file__).resolve().parent


def analyze():
    path = HERE / "mcu/matched_host_reference.json"
    reference = json.loads(path.read_text(encoding="utf8"))
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        lib, command = compile_library(tmp)
        for stream in reference["streams"]:
            for base, family in [(0, "lut"), (2, "cubic")]:
                lib.matched_reset(base)
                unprojected = [state(lib).copy()]
                violations = 0
                frozen = np.array([lib.matched_online_predict(base, x) for x in stream["final_adc"]])
                frozen_mae = float(np.mean(abs(frozen - np.array(reference["truth"]))))
                for x, y in zip(stream["adc"], stream["reference"]):
                    lib.matched_step(base, x, y)
                    coefficients = state(lib).copy()
                    unprojected.append(coefficients)
                    violations += int(np.any(np.diff(coefficients) > 0))
                lib.matched_reset(base + 1)
                gap = 0.0
                for step, (x, y) in enumerate(zip(stream["adc"], stream["reference"]), 1):
                    lib.matched_step(base + 1, x, y)
                    gap = max(gap, float(np.max(abs(state(lib) - unprojected[step]))))
                rows.append(dict(drift=stream["mode"], family=family, checked_updates=384,
                                 unprojected_steps_violating_monotonicity=violations,
                                 max_projected_vs_unprojected_state_gap_C=gap * reference["transform"]["y_half"],
                                 frozen_final_grid_mae_C=frozen_mae))
    return dict(reference_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                kernel_sha256=hashlib.sha256((HERE / "mcu/include/matched_kernels.h").read_bytes()).hexdigest(),
                compiler_flags=command[1:command.index("-I")], rows=rows,
                note="Host replay of actual float32 kernels; not new device timing. Frozen baselines use each drift's final ADC grid, not the static test grid.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with args.out.open("x", encoding="utf8") as output:
        json.dump(analyze(), output, indent=2)
        output.write("\n")
