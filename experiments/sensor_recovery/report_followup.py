import argparse,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument("directory",type=Path);args=p.parse_args();d=args.directory
    rows=json.loads((d/"summary.json").read_text())
    info=json.loads((d/"run_info.json").read_text())
    def get(s,b,m,l=32):return next(r for r in rows if r["scenario"]==s and r["budget"]==b and r["method"]==m and r["length"]==l and r["encoding"]=="int16")
    lines=["# Shape-constrained LUT follow-up", "", f"Synthetic confirmation seed {info['seed']}; {info['cases']} cases, {info['evaluations']:,} model evaluations; {info['units']} new virtual units × {info['repeats']} repeats × 2 budgets × 3 scenarios. No physical sensor or MCU timings are contained in this report.","",
      "## Equal-payload uniform LUT comparison", "", f"L32/int16: 88 bytes for either method. Max is the largest absolute error across all held-out inputs and {info['units']*info['repeats']} calibration/noise cases in a row.","",
      "| Scenario | Budget | Uniform MAE | Monotone MAE | Uniform max | Monotone max |", "|---|---:|---:|---:|---:|---:|"]
    for s in ("ntc","ntc_frontend","ldr"):
      for b in (30,75):
        a,z=get(s,b,"direct_lut"),get(s,b,"monotone_lut")
        lines.append(f"| {s} | {b} | {a['mean_noisy_mae']:.6f} | {z['mean_noisy_mae']:.6f} | {a['worst_noisy_max_abs']:.6f} | {z['worst_noisy_max_abs']:.6f} |")
    lines += ["", "NTC errors are in °C; LDR errors in log10(lux). All monotone LUTs preserve the declared direction after float32/int16 encoding. Monotonicity does not bound the physical error, and it can worsen validation-selected mean error in other sizes, notably L64/B75. See the complete table below.","",
      "## What to carry forward", "",
      "- Monotonicity is useful as a shape safeguard at no additional inference payload. It substantially reduces severe sparse-fit outliers, but is not a universal MAE improvement.",
      "- At L32/B30, the paired 95% unit-bootstrap intervals for mean MAE differences include zero for both NTC scenarios (see paired.json). The reduction of the observed largest errors should not be presented as a statistically established mean-accuracy gain.",
      "- Nonuniform knots can improve approximation, especially in the earlier failed cases. They add 4L knot bytes and binary search at inference. Their accuracy cannot be compared as if payload and execution cost were unchanged.",
      "- Strong physics/polynomial/cubic comparators remain necessary. This follow-up does not establish a resource win: real-board timings are the next required observation.",
      "- The firmware fixture is predeclared (new seed, distorted NTC, unit0/repeat0, B75), not the best test case. Its C++ kernels were checked against exported Python references; cross-compilation is distinct from execution on a board.", "",
      "## Known failures from the original pilot (regression examples only)", "", "| Scenario | Method | L | Payload | MAE | Max |", "|---|---|---:|---:|---:|---:|"]
    for r in json.loads((d/"known_failure_regressions.json").read_text()):
      if r["encoding"]=="int16":lines.append(f"| {r['scenario']} | {r['method']} | {r['length']} | {r['payload_bytes']} | {r['noisy_mae']:.6f} | {r['noisy_max_abs']:.6f} |")
    lines += ["", "These two previously observed failures were kept separate from new confirmation data. They cannot be treated as independent evidence for a newly chosen method.", "", "## Complete summary", "",
      "| Scenario | B | Method | L | Encoding | Mean payload B | MAE | Worst max |", "|---|---:|---|---:|---|---:|---:|---:|"]
    for r in rows:lines.append(f"| {r['scenario']} | {r['budget']} | {r['method']} | {r['length']} | {r['encoding']} | {r['mean_payload_bytes']:.1f} | {r['mean_noisy_mae']:.6f} | {r['worst_noisy_max_abs']:.6f} |")
    (d/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf8")

if __name__=="__main__":main()
