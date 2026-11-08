"""Regenerate numerical summaries and figures solely from saved raw records."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matched_adaptation import paired_unit_difference


def read_lines(path):
    with gzip.open(path,"rt") as f:
        for line in f:yield json.loads(line)


def dump(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")


def aggregate(rows,keys,metrics):
    groups=defaultdict(list)
    for r in rows:groups[tuple(r[k] for k in keys)].append(r)
    out=[]
    for key,values in sorted(groups.items()):
        units=sorted({r["unit"] for r in values})
        item=dict(zip(keys,key));item.update(n_units=len(units),n_cases=len(values))
        for metric in metrics:
            perunit=[float(np.mean([r[metric] for r in values if r["unit"]==u])) for u in units]
            ix=np.random.default_rng(2026110804).integers(0,len(units),(2000,len(units)))
            draws=np.asarray(perunit)[ix].mean(axis=1)
            lo,hi=np.quantile(draws,[.025,.975])
            item[metric]=dict(mean=float(np.mean(perunit)),ci95_lo=float(lo),ci95_hi=float(hi),
                              per_unit=dict(zip(map(str,units),perunit)))
        item["worst_observed_error"]=max(r.get("noisy_max_abs",r.get("peak_grid_max_abs",0)) for r in values)
        if "persistent_state_bytes" in values[0]:
            item["persistent_state_bytes_range"]=[min(r["persistent_state_bytes"] for r in values),
                                                   max(r["persistent_state_bytes"] for r in values)]
        out.append(item)
    return out


def save_figure(fig,path):
    fig.savefig(path.with_suffix(".png"),dpi=180,bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"),bbox_inches="tight",metadata={"CreationDate":None,"ModDate":None})
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("results",type=Path)
    p.add_argument("--out",type=Path)
    a=p.parse_args();root=a.results;out=a.out or root
    out.mkdir(parents=True,exist_ok=True)
    infos={s:json.loads((root/s/"run_info.json").read_text()) for s in ("static","development","confirmation")}
    for s,info in infos.items():
        if not info["complete"]:raise ValueError("Incomplete stage: "+s)
        for name,expected in info["result_sha256"].items():
            if hashlib.sha256((root/s/name).read_bytes()).hexdigest()!=expected:
                raise ValueError("Result hash mismatch: "+s+"/"+name)
    selection=json.loads((root/"development/selected_parameters.json").read_text())
    static=list(read_lines(root/"static/raw_results.jsonl.gz"))
    trajectories=[];curves=defaultdict(list);failures=[];order_violations=0
    for r in read_lines(root/"confirmation/raw_trajectories.jsonl.gz"):
        if r["failure"]:failures.append({k:r[k] for k in ("case_id","method","failure")})
        for snap in r["snapshots"]:
            curves[(r["scenario"],r["budget"],r["drift"],r["method"],snap["t"])].append(
                dict(unit=r["unit"],repeat=r["repeat"],mae=snap["after"]["mae"],before_mae=snap["before"]["mae"]))
            if "projected" in r["method"]:order_violations+=snap.get("coefficient_order_violations",0)
        trajectories.append({k:v for k,v in r.items() if k not in ("snapshots","prequential")})
    if failures:
        dump(out/"failed_trajectories.json",failures)
        raise RuntimeError("Failures retained; do not aggregate incomplete trajectories as successes")
    ss=aggregate(static,["scenario","budget","cap_bytes","method"],["noisy_mae","noisy_rmse","ideal_mae"])
    ts=aggregate(trajectories,["scenario","budget","drift","method"],["final_mae","integrated_mae","final_rmse"])
    pairs=[]
    for method in ("lut_float32","lut_int16","monotone_lut_float32","monotone_lut_int16","cubic_compact","kan_compact"):
        pairs+=paired_unit_difference(static,method,"cubic_horner",["scenario","budget","cap_bytes"],"noisy_mae")
    apairs=[]
    for ma,mb in (("lut_projected_nlms","frozen_lut"),("cubic_projected_nlms","frozen_cubic"),
                  ("lut_projected_nlms","cubic_projected_nlms"),("lut_nlms","cubic_nlms"),
                  ("lut_projected_nlms","cubic_rls"),("lut_projected_nlms","poly_rls")):
        apairs+=paired_unit_difference(trajectories,ma,mb,["scenario","budget","drift"],"final_mae")
    dump(out/"static_summary.json",ss);dump(out/"adaptation_summary.json",ts)
    dump(out/"static_paired.json",pairs);dump(out/"adaptation_paired.json",apairs)
    cr=[]
    for key,values in sorted(curves.items()):
        units=sorted({v["unit"] for v in values})
        perunit=[np.mean([v["mae"] for v in values if v["unit"]==u]) for u in units]
        before=[np.mean([v["before_mae"] for v in values if v["unit"]==u]) for u in units]
        cr.append(dict(zip(("scenario","budget","drift","method","t"),key),
                       mean_mae=float(np.mean(perunit)),mean_before_mae=float(np.mean(before))))
    dump(out/"adaptation_curves.json",cr)

    methods=[("lut_float32","LUT f32"),("lut_int16","LUT int16"),
             ("monotone_lut_float32","Monotone LUT f32"),
             ("cubic_compact","Cubic compact"),("cubic_horner","Cubic Horner")]
    colors=["#146fa6","#4b9fc3","#eea135","#8271a7","#ca4e52"]
    titles={"ntc":"NTC","ntc_frontend":"Distorted NTC","ldr":"LDR"}
    fig,axes=plt.subplots(2,3,figsize=(11,6.3),constrained_layout=True)
    for row,budget in enumerate((30,75)):
        for col,scenario in enumerate(titles):
            ax=axes[row,col]
            for (method,label),color in zip(methods,colors):
                vals=sorted([v for v in ss if v["scenario"]==scenario and v["budget"]==budget and v["method"]==method],key=lambda v:v["cap_bytes"])
                ax.plot([v["cap_bytes"] for v in vals],[v["noisy_mae"]["mean"] for v in vals],"o-",label=label,color=color,lw=1.5,ms=4)
            ax.set_title(f"{titles[scenario]}, B={budget}",fontsize=11)
            ax.set_xticks([80,144,224]);ax.grid(alpha=.2)
            ax.set_ylabel("MAE, log10(lux)" if scenario=="ldr" else "MAE, °C")
            if row==1:ax.set_xlabel("Inference payload, bytes")
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",ncol=3,fontsize=9)
    save_figure(fig,out/"matched_payload")
    methods2=[("frozen_lut","Frozen LUT"),("lut_projected_nlms","Projected LUT NLMS"),
              ("cubic_projected_nlms","Projected cubic NLMS"),("cubic_rls","Cubic RLS"),("poly_rls","Polynomial RLS")]
    fig,axes=plt.subplots(2,3,figsize=(11,6.3),constrained_layout=True)
    for row,budget in enumerate((30,75)):
        for col,scenario in enumerate(titles):
            ax=axes[row,col]
            for (method,label),color in zip(methods2,colors):
                vals=[v for v in cr if v["scenario"]==scenario and v["budget"]==budget and v["drift"]=="step" and v["method"]==method]
                ax.plot([v["t"] for v in vals],[v["mean_mae"] for v in vals],label=label,color=color,lw=1.5)
            ax.axvline(128,color=".5",ls=":",lw=.8)
            ax.set_title(f"{titles[scenario]}, B={budget}",fontsize=11);ax.grid(alpha=.2)
            ax.set_ylabel("MAE, log10(lux)" if scenario=="ldr" else "MAE, °C")
            if row==1:ax.set_xlabel("Sequential reference samples")
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",ncol=3,fontsize=9)
    save_figure(fig,out/"step_adaptation")

    lookup={(v["scenario"],v["budget"],v["cap_bytes"],v["method"]):v for v in ss}
    alookup={(v["scenario"],v["budget"],v["drift"],v["method"]):v for v in ts}
    mean=lambda s,b,m:lookup[(s,b,144,m)]["noisy_mae"]["mean"]
    amean=lambda s,b,d,m:alookup[(s,b,d,m)]["final_mae"]["mean"]
    lines=["# Matched storage and sequential adaptation: new reconstruction run","",
      "All sensor observations are synthetic. New seeds and the predeclared protocol distinguish this run from the interrupted conversation. No new physical MCU measurements were made.","",
      f"Static: {len(static)} evaluations / {infos['static']['cases']} calibration cases. Development: {infos['development']['trajectories']} candidate trajectories / {infos['development']['streams']} streams. Confirmation: {len(trajectories)} trajectories / {infos['confirmation']['streams']} streams. All confirmation trajectories completed; projected coefficient-order violations: {order_violations}.","",
      "## Static error at exactly 144 bytes","",
      "| Scenario | B | LUT32 f32 | LUT60 int16 | Monotone LUT60 int16 | Compact cubic32 | Horner8 |",
      "|---|---:|---:|---:|---:|---:|---:|"]
    for scenario in titles:
        for budget in (30,75):
            vals=[mean(scenario,budget,m) for m in ("lut_float32","lut_int16","monotone_lut_int16","cubic_compact","cubic_horner")]
            lines.append(f"| {scenario} | {budget} | "+" | ".join(f"{v:.6f}" for v in vals)+" |")
    lines += ["","NTC units are °C; LDR is log10(lux). Means average repetitions within each virtual unit, then units. All seven equal-cap methods and both smaller physical/polynomial references are retained in static_summary.json; the figure includes all three predeclared caps.","",
      "The 144-byte Horner model has eight pieces fitted with eleven B-spline coefficients, then stored as 32 local polynomial coefficients. The compact cubic32 has a different function class and evaluation path. Equal payload does not imply equal degrees of freedom or execution time.","",
      "The old measured Horner model used thirteen pieces and 224 bytes. Its MCU timing cannot be assigned to this new eight-piece model. The new static sweep has no target timing measurements.","",
      "## Final error after 384 sequential references","",
      "| Scenario | B | Drift | Frozen LUT | Projected LUT NLMS | Projected cubic NLMS | Cubic RLS | Polynomial RLS |",
      "|---|---:|---|---:|---:|---:|---:|---:|"]
    for scenario in titles:
        for budget in (30,75):
            for drift in ("none","step","ramp"):
                vals=[amean(scenario,budget,drift,m) for m in ("frozen_lut","lut_projected_nlms","cubic_projected_nlms","cubic_rls","poly_rls")]
                lines.append(f"| {scenario} | {budget} | {drift} | "+" | ".join(f"{v:.6f}" for v in vals)+" |")
    lines += ["","The complete nine-method tables, paired unit-bootstrap intervals, and checkpoint curves are in the JSON summaries. Intervals are exploratory and unadjusted. Frozen models receive the same available stream but do not consume its labels for updates.","",
      "## Storage and interpretation","",
      "LUT32 and compact cubic32 use 144 bytes for inference. Their NLMS paths use 148 persistent bytes including eta. PAVA working storage and the Python reference implementation's temporary arrays are additional and unmeasured on MCU. Cubic32 RLS uses 4244 persistent bytes including its full covariance; polynomial RLS uses a degree-dependent larger state, up to 460 bytes. These RLS methods are stronger, larger-state references, not equal-memory algorithms.","",
      "Every method starts from an offline reference fit. The online coefficients/covariance are rounded to float32 after each sample, with float64 host intermediates. A future MCU implementation needs numerical validation and its own update-time/stack measurements. The earlier UPDATE_COST_ONLY timings are not timings of these algorithms.","",
      "The controlled stream supplies one labelled reference in every quantity bin per shuffled block. This is a recalibration experiment with reference access, not unsupervised self-calibration. Offset and nonlinear ADC drift are injected assumptions. Physical NTC accuracy, sensor aging, energy and peak stack remain unmeasured.","",
      "## Selected parameters and reproducibility","",
      "Development-only selection: `"+json.dumps(selection["parameters"],sort_keys=True)+"`.","",
      "The development seed is 2026110800; confirmation seed is 2026110801. Protocol and selected parameters were committed before confirmation. Every raw reference, static model, online prequential prediction, checkpoint model state and RLS covariance is retained. run_info.json contains commands, versions, source hashes and result hashes.","",
      "Regenerate: `python experiments/sensor_recovery/report_matched_adaptation.py experiments/sensor_recovery/results/matched_adaptation_20261108`.","",
      "![Matched payload](matched_payload.png)","","![Step adaptation](step_adaptation.png)",""]
    (out/"RESULTS.md").write_text("\n".join(lines))
    pairs144=[v for v in pairs if v["cap_bytes"]==144 and v["method_a"] in ("lut_float32","lut_int16")]
    boundary=[r["model"]["boundary_gaps"] for r in static if r["method"]=="cubic_horner"]
    diagnostics=dict(static_evaluations=len(static),confirmation_trajectories=len(trajectories),
        confirmation_failures=failures,projected_order_violations=order_violations,
        horner144_better_mean_count=sum(v["mean_a_minus_b"]>0 for v in pairs144),
        horner144_comparisons=len(pairs144),
        maximum_normalized_horner_value_gap=max(v["value"] for v in boundary),
        maximum_normalized_horner_derivative_gap=max(v["derivative"] for v in boundary),
        report_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    dump(out/"report_diagnostics.json",diagnostics)
    print(json.dumps(diagnostics,indent=2))


if __name__=="__main__":main()
