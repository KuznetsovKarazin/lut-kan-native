"""Build descriptive cross-board tables and figures from audited captures."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
METHODS = ("lut32_float32", "lut60_int16", "cubic32_unrolled", "horner8")
LABELS = ("LUT32 float32", "LUT60 int16", "Cubic32", "Horner8")
UPDATES = ("lut_nlms", "lut_projected_nlms", "cubic_nlms", "cubic_projected_nlms")


def summarize(values):
    return dict(session_medians_us=values, median_of_session_medians_us=statistics.median(values),
                min_session_median_us=min(values), max_session_median_us=max(values),
                relative_session_range_percent=100*(max(values)-min(values))/statistics.mean(values))


def compare():
    boards = {}
    firmware_sources = None
    for board, folder in [("mega2560", "mega_matched_20261108"), ("esp32c3", "c3_matched_20261108")]:
        path = HERE / "results" / folder
        audit = json.loads((path / "audit.json").read_text(encoding="utf8"))
        if audit["status"] != "validated" or audit["board"] != board or len(audit["sessions"]) != 2:
            raise ValueError("Expected two validated sessions for the named board")
        sources = {name: row["lf_sha256"] for name, row in audit["source_checks"].items()
                   if name.endswith((".h", ".cpp", "platformio.ini"))}
        if firmware_sources is not None and sources != firmware_sources:
            raise ValueError("Board captures do not use the same firmware sources and build configuration")
        firmware_sources = sources
        runs = [json.loads((path / f"run{i:02d}.json").read_bytes()) for i in (1, 2)]
        for i in (1, 2):
            name = f"run{i:02d}.json"
            if hashlib.sha256((path/name).read_bytes()).hexdigest() != audit["files"][name]["sha256"]:
                raise ValueError("Parsed result changed after archive audit")
        for field in ("accuracy", "update_accuracy", "payload_bytes", "update_storage"):
            if runs[0][field] != runs[1][field]:
                raise ValueError(f"Session {field} differ; single-value summary needs review")
        if any(runs[0]["payload_bytes"][m] != 144 for m in METHODS):
            raise ValueError("Expected the fixed 144-byte inference comparison")
        static = {m: dict(**summarize([r["inference_median_us"][m] for r in runs]),
                          metrics=[r["accuracy"][m] for r in runs], payload_bytes=144) for m in METHODS}
        online = {d: {m: dict(**summarize([next(x["gross_us_per_update"] for x in r["update_median_us"]
                                                if x["drift"]==d and x["method"]==m) for r in runs]),
                               metrics=[next(x for x in r["update_accuracy"] if x["drift"]==d and x["method"]==m) for r in runs],
                               **runs[0]["update_storage"][m]) for m in UPDATES} for d in ("none", "step", "ramp")}
        def central(row): return row["median_of_session_medians_us"]
        ratios = dict(horner_over_lut=central(static["horner8"])/central(static["lut32_float32"]),
                      lut_inference_time_reduction_percent=100*(1-central(static["lut32_float32"])/central(static["horner8"])),
                      step_plain_cubic_over_lut=central(online["step"]["cubic_nlms"])/central(online["step"]["lut_nlms"]),
                      step_projected_cubic_over_lut=central(online["step"]["cubic_projected_nlms"])/central(online["step"]["lut_projected_nlms"]),
                      step_projected_lut_time_reduction_percent=100*(1-central(online["step"]["lut_projected_nlms"])/central(online["step"]["cubic_projected_nlms"])))
        boards[board] = dict(source_commit=audit["source_commit"], archive_sha256=audit["archive_sha256"],
                             sessions=audit["sessions"], static=static, online=online, ratios=ratios)
    differences = []
    for m in METHODS:
        for metric in ("mae_C", "rmse_C", "max_C"):
            differences.append(abs(boards["mega2560"]["static"][m]["metrics"][0][metric] - boards["esp32c3"]["static"][m]["metrics"][0][metric]))
    for d in ("none", "step", "ramp"):
        for m in UPDATES:
            for metric in ("mae_C", "rmse_C", "max_C"):
                differences.append(abs(boards["mega2560"]["online"][d][m]["metrics"][0][metric] - boards["esp32c3"]["online"][d][m]["metrics"][0][metric]))
    return dict(boards=boards, maximum_reported_metric_difference_between_boards_C=max(differences),
                matched_firmware_source_hashes=firmware_sources,
                aggregation="Median of the two within-session medians; raw repeats are not pooled. Descriptive statistic, not independent-board inference.",
                scope="Synthetic labelled NTC fixture; one physical board per architecture, two sequential sessions each; gross timings with reads/checksums included. PAVA needs no block merges in this fixture.")


def figures(data, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10, "axes.spines.top":False,
                         "axes.spines.right":False, "svg.hashsalt":"matched-boards-20261108"})
    names = [("mega2560", "Mega2560 · 16 MHz"), ("esp32c3", "ESP32-C3 · 160 MHz")]
    colors = ["#0072B2", "#56B4E9", "#009E73", "#D55E00"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, (board, title) in zip(axes, names):
        values = data["boards"][board]["static"]
        for index, (method, label, color) in enumerate(zip(METHODS, LABELS, colors)):
            row=values[method];x=row["median_of_session_medians_us"];y=row["metrics"][0]["mae_C"]
            ax.scatter(x,y,s=85,c=color,edgecolors="white",linewidths=.8,zorder=3)
            dx,dy,ha = (-9,10,"right") if index==2 else (8,9,"left")
            if index==0:dy=-15
            if index==3:dy=-17
            ax.annotate(label,(x,y),xytext=(dx,dy),textcoords="offset points",ha=ha,fontsize=9)
        ax.set_xlim(0,max(v["median_of_session_medians_us"] for v in values.values())*1.17)
        ax.set_ylim(.040,.060);ax.grid(alpha=.18);ax.set_axisbelow(True)
        ax.set_title(title,loc="left",fontweight="bold");ax.set_xlabel("Inference time (µs/call)")
    axes[0].set_ylabel("Mean absolute error (°C)")
    fig.suptitle("Equal 144-byte payload: latency and error",x=.065,ha="left",fontsize=15,fontweight="bold")
    fig.subplots_adjust(left=.075,right=.98,top=.82,bottom=.23,wspace=.19)
    fig.text(.075,.06,"Lower is better on both axes. Synthetic NTC fixture; two sessions per board, one board per architecture.",fontsize=9,color="#444444")
    for ext in ("png","svg"):fig.savefig(out/("matched_latency_accuracy."+ext),dpi=180,metadata={"Date":None} if ext=="svg" else None)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    labels=["LUT NLMS","LUT NLMS + PAVA","Cubic NLMS","Cubic NLMS + PAVA"]
    for ax, (board,title) in zip(axes,names):
        values=[data["boards"][board]["online"]["step"][m]["median_of_session_medians_us"] for m in UPDATES]
        ax.barh(range(4),values,color=["#0072B2","#56B4E9","#D55E00","#E69F00"],height=.58)
        ax.set_yticks(range(4),labels,fontsize=9);ax.invert_yaxis()
        ax.set_xlim(0,max(values)*1.23);ax.set_xlabel("Complete update time (µs/reference)")
        ax.set_title(title,loc="left",fontweight="bold");ax.grid(axis="x",alpha=.18);ax.set_axisbelow(True)
        for i,v in enumerate(values):ax.text(v+max(values)*.018,i,f"{v:.2f}",va="center",fontsize=9)
    fig.suptitle("Complete adaptation under step drift",x=.04,ha="left",fontsize=15,fontweight="bold")
    fig.subplots_adjust(left=.15,right=.98,top=.82,bottom=.24,wspace=.55)
    fig.text(.04,.08,"148-byte persistent state; PAVA adds 160-byte workspace. Gross times; reset and validation excluded.",fontsize=9,color="#444444")
    fig.text(.04,.035,"Bars: median of two session medians. This fixture requires no PAVA block merges; not worst-case timing.",fontsize=9,color="#444444")
    for ext in ("png","svg"):fig.savefig(out/("matched_complete_update."+ext),dpi=180,metadata={"Date":None} if ext=="svg" else None)
    plt.close(fig)


def table(data):
    lines=[r"\begin{tabular}{lrrr}",r"\hline",r"Model (144 B) & Mega ($\mu$s) & C3 ($\mu$s) & MAE ($^\circ$C) \\",r"\hline"]
    for method,label in zip(METHODS,LABELS):
        mega=data["boards"]["mega2560"]["static"][method];c3=data["boards"]["esp32c3"]["static"][method]
        lines.append(f"{label} & {mega['median_of_session_medians_us']:.3f} & {c3['median_of_session_medians_us']:.3f} & {c3['metrics'][0]['mae_C']:.5f} "+r"\\")
    return "\n".join(lines+[r"\hline",r"\end{tabular}","% MAE shown at five decimals; platform differences are at most about 1e-7 deg C.",
                              "% Times: median of two acquisition medians, not a pooled sample or a confidence interval.",""])


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    data=compare();(args.out/"comparison.json").write_text(json.dumps(data,indent=2)+"\n",encoding="utf8")
    (args.out/"static_table.tex").write_text(table(data),encoding="utf8");figures(data,args.out)
    print(json.dumps({board:values["ratios"] for board,values in data["boards"].items()},indent=2))
