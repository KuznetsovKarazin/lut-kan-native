"""Generate descriptive figures and a numerical report from retained results.

No model selection is performed here. L=32/B=75 is a labelled slice of the
predeclared sweep; all settings, including failures, remain in summary.csv.
"""
import argparse
import gzip
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument("results",type=Path)
    args=p.parse_args()
    out=args.results
    summary=json.loads((out/"summary.json").read_text())
    with gzip.open(out/"raw_results.jsonl.gz", "rt", encoding="utf-8") as f:
        raw=[json.loads(line) for line in f]
    info=json.loads((out/"run_info.json").read_text())
    pairs=json.loads((out/"paired_comparisons.json").read_text())
    diag=json.loads((out/"optimizer_diagnostics.json").read_text())
    def get(s,m,l=0,e="float32",b=75):
        return next(r for r in summary if (r["scenario"],r["budget"],r["method"],r["length"],r["encoding"])==(s,b,m,l,e))
    names={"ntc":"NTC: matched physics", "ntc_frontend":"NTC: injected frontend distortion", "ldr":"LDR: matched power law"}
    methods=[("physics",0,"float32","Physics"),("polynomial",0,"float32","Polynomial"),
             ("cubic_spline",32,"float32","Cubic spline"),("kan_edge",32,"float32","KAN edge"),
             ("direct_lut",32,"int16","Direct LUT"),("kan_to_lut",32,"int16","KAN → LUT"),
             ("physics_to_lut",32,"int16","Physics → LUT"),("physics_residual_lut",32,"float32","Physics + residual")]
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,3,figsize=(15,5.8),layout="constrained")
    for ax,s in zip(axes,names):
        values=[get(s,m,l,e) for m,l,e,_ in methods]
        y=np.array([r["mean_noisy_mae"] for r in values])
        err=np.array([[r["mean_noisy_mae"]-r["unit_bootstrap_mae_lo"] for r in values],
                      [r["unit_bootstrap_mae_hi"]-r["mean_noisy_mae"] for r in values]])
        ax.scatter(y,np.arange(len(values)),color=["#64748b"]*4+["#007f85","#c46b24","#8d729b","#64748b"],s=42,zorder=3)
        ax.errorbar(y,np.arange(len(values)),xerr=err,fmt="none",color="black",capsize=2,lw=.8)
        ax.set_yticks(np.arange(len(values)),[m[3] for m in methods]);ax.invert_yaxis()
        ax.set_xscale("log");ax.set_title(names[s],fontsize=11)
        ax.set_xlabel("MAE, log10(lux)" if s=="ldr" else "MAE, °C")
        ax.grid(axis="x",alpha=.15);ax.set_axisbelow(True)
    fig.suptitle("Synthetic pilot · 75 labelled references · 8 virtual units × 3 repeats\n95% unit-bootstrap intervals; L=32 for spline/LUT methods; different model payloads",fontsize=13)
    fig.savefig(out/"method_comparison.png",dpi=180)
    fig.savefig(out/"method_comparison.pdf")
    plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(11,4.6),layout="constrained")
    for ax,s in zip(axes,("ntc","ntc_frontend")):
        for fmt,color,marker in (("float32","#64748b","o"),("int16","#007f85","s"),("uint8","#c46b24","^")):
            v=[get(s,"direct_lut",l,fmt) for l in (8,16,32,64)]
            ax.plot([r["mean_payload_bytes"] for r in v],[r["mean_noisy_mae"] for r in v],marker=marker,color=color,label=f"Direct LUT {fmt}")
        for method,label,color in (("physics","Physics","#8d729b"),("polynomial","Polynomial","#b53552")):
            r=get(s,method)
            ax.scatter(r["mean_payload_bytes"],r["mean_noisy_mae"],marker="D",s=40,color=color,label=label,zorder=4)
        ax.set_title(names[s],fontsize=11);ax.set_xlabel("Model payload, bytes (transforms included)")
        ax.set_ylabel("Noisy-test MAE, °C");ax.set_yscale("log");ax.grid(alpha=.15)
        ax.legend(fontsize=8)
    fig.suptitle("Synthetic NTC pilot · 75 references · payload excludes firmware and training RAM",fontsize=12)
    fig.savefig(out/"precision_payload.png",dpi=180)
    fig.savefig(out/"precision_payload.pdf")
    plt.close(fig)

    lines=["# Sensor recovery pilot — numerical results", "",
           "All results below are synthetic simulations. No physical sensor or MCU was measured.","",
           f"Run: {info['cases']} calibration cases, {info['result_rows']} model evaluations, 8 virtual units, 3 repeats per unit, two reference budgets (30/75).",
           "The original protocol precedes these results; L=32/B=75 below is a descriptive slice of the full sweep, not a test-selected configuration.","",
           "## Comparable deployment formats", "",
           "All three tabulated methods below use L=32, int16: 64 table bytes + 8 quantizer bytes + 16 transform bytes = 88 bytes.","",
           "| Scenario | Direct LUT MAE | KAN-edge → LUT MAE | Physics → LUT MAE | Direct vs KAN → LUT reduction |",
           "|---|---:|---:|---:|---:|"]
    for s in names:
        d,k,f=[get(s,m,32,"int16")["mean_noisy_mae"] for m in ("direct_lut","kan_to_lut","physics_to_lut")]
        lines.append(f"| {names[s]} | {d:.6f} | {k:.6f} | {f:.6f} | {100*(1-d/k):.1f}% |")
    lines += ["", "NTC units are °C; LDR units are log10(lux). KAN means one fixed-grid cubic spline edge plus SiLU, fitted by augmented least squares. It does not mean a multilayer/adaptive-grid KAN.","",
              "## Stronger models before LUT compilation", "", "| Scenario | Physics MAE | Polynomial MAE | Cubic spline MAE | Original KAN edge MAE | Physics + LUT residual MAE |", "|---|---:|---:|---:|---:|---:|"]
    for s in names:
        vals=[get(s,m,l)["mean_noisy_mae"] for m,l in (("physics",0),("polynomial",0),("cubic_spline",32),("kan_edge",32),("physics_residual_lut",32))]
        lines.append("| "+names[s]+" | "+" | ".join(f"{v:.6f}" for v in vals)+" |")
    lines += ["", "Their payloads are 28–32 bytes (physics), 48–56 bytes (selected polynomial), 144 bytes (cubic), 148 bytes (KAN edge), and 156–160 bytes (physics + residual). The last model still evaluates logarithms at inference. Host arithmetic uses float32-rounded stored coefficients; it is not a full MCU arithmetic simulation.", "",
              "## Interpretation", "",
              "- Direct training improves over naive sampling of a fitted scalar KAN edge into the same uniform deployment table. The paired unit-bootstrap intervals for these L32/int16 differences exclude zero in all three scenarios at B75; these are exploratory, unadjusted comparisons among many settings.",
              "- It does not establish better accuracy than the uncompiled KAN/cubic model. For matched NTC and LDR physics, the physical inverse is both more accurate and smaller.",
              "- The intentionally distorted NTC exposes model mismatch: physics fails, but a selected polynomial/cubic spline also corrects it. A claim that LUT alone handles such mismatch would be unsupported.",
              "- int16 preserves float32-table accuracy closely; uint8 roughly doubles NTC error at L32. Precision must be assessed after encoding, not inferred from a float32 experiment.",
              "- Some sparse/noisy fits have large isolated errors. Do not select a model by mean error alone; monotonicity, conditioning and maximum-error safeguards are the next step.",
              "- Uniform sampling of a surrogate is only one compilation method. Optimized/nonuniform knots and deployment-aware distillation may remove some of the observed gap and must be included before claiming broad novelty.",
              "- The exact linear solve is an offline reference optimum, not a proposed low-RAM MCU trainer. A portable C inference kernel is supplied; on-device updates, timings and memory still require measurement.", "",
              "## Quantization (L32, B75)", "", "| Scenario | float32 MAE / 144 B | int16 MAE / 88 B | uint8 MAE / 56 B |", "|---|---:|---:|---:|"]
    for s in names:
        lines.append("| "+names[s]+" | "+" | ".join(f"{get(s,'direct_lut',32,fmt)['mean_noisy_mae']:.6f}" for fmt in ("float32","int16","uint8"))+" |")
    ratios=np.array([r["objective_adam"]/r["objective_ls"] for r in diag])
    lines += ["", "## Optimizer and tail diagnostics", "",
              f"For identical fixed-grid objectives, 1000-step Adam / exact-solve loss ratio: median {np.median(ratios):.6f}, maximum {ratios.max():.6f}. Thus there is no evidence of a universal optimization breakthrough in the scalar problem.","",
              "Worst direct-LUT float32 cases in each scenario (all predeclared budgets/sizes):", "",
              "| Scenario | Budget | L | Unit / repeat | Selected lambda | Max absolute error | MAE |", "|---|---:|---:|---|---:|---:|---:|"]
    for s in names:
        r=max((r for r in raw if r["scenario"]==s and r["method"]=="direct_lut" and r["encoding"]=="float32"),key=lambda r:r["noisy_max_abs"])
        lines.append(f"| {s} | {r['budget']} | {r['length']} | {r['unit']} / {r['repeat']} | {r['hyper']['lambda_2']} | {r['noisy_max_abs']:.6f} | {r['noisy_mae']:.6f} |")
    lines += ["", "## Next experiments, in priority order", "",
              "1. Freeze a follow-up protocol: monotone LUTs, boundary/occupancy guards, adaptive/nonuniform piecewise-linear baseline, maximum error as well as MAE. Use fresh virtual units for confirmation.",
              "2. Use Mega 2560 and ESP32-C3 to measure direct LUT, physical inverse, Chebyshev, cubic and KAN-edge inference; report flash, SRAM, latency and end-to-end physical error. Build a separate memory-bounded online trainer and compare against recursive spline/RLS and SGD baselines.",
              "3. Acquire/borrow NTC probes and a sufficiently accurate reference setup. Start with multiple units of one model, independent measurement sessions and temperature cycles; obtain raw ADC/reference records and an uncertainty budget before adding another sensor family.",
              "4. For IEEE Sensors Journal, frame the contribution around measured sensor-chain recalibration under memory/energy constraints, if the data support it. Without physical evidence or an MCU advantage over polynomial/physics, this pilot alone is insufficient for resubmission elsewhere.", "",
              "## Reproduction and provenance", "",
              "See [README](../../README.md) for commands. `run_info.json` records seed/configuration, source SHA256 hashes, versions and the base commit. The working tree was deliberately marked dirty because code/results were committed after execution. All selected hyperparameters are in `raw_results.jsonl.gz`; all generated references and holdouts are in `cases.jsonl.gz` (standard gzip-compressed JSON Lines). The synthesis code is the authoritative provenance, not a manufacturer's measurement dataset.", ""]
    (out/"RESULTS.md").write_text("\n".join(lines),encoding="utf-8")


if __name__=="__main__":
    main()
