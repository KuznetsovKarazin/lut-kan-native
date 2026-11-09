"""Generate manuscript assets from the immutable audited result snapshot.

No calibration or MCU benchmark is rerun. All numerical inputs are versioned
results; data/source_hashes.json makes the figure inputs explicit.
"""
from pathlib import Path
import hashlib
import json
import os
import tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, ListedColormap
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = ROOT / "experiments/sensor_recovery/results"
FIG = HERE / "figures"
TABLES = HERE / "tables"
DATA = HERE / "data"
SNAPSHOT = "db39238c1467169b65ff1f3581945e57ee7ffd7a"
COL = {"lut_float32":"#0072B2", "lut_int16":"#56B4E9", "cubic_compact":"#009E73",
       "cubic_horner":"#D55E00", "frozen_lut":"#777777", "frozen_cubic":"#999999",
       "lut_projected_nlms":"#0072B2", "cubic_projected_nlms":"#D55E00",
       "cubic_rls":"#009E73", "poly_rls":"#B07AA1"}
plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":8,
    "axes.titlesize":9, "axes.labelsize":8, "xtick.labelsize":7,
    "ytick.labelsize":7, "legend.fontsize":7, "axes.spines.top":False,
    "axes.spines.right":False, "pdf.fonttype":42, "ps.fonttype":42,
    "svg.hashsalt":"sensors-tradeoff-20261108"})


def read(relative):
    path = RESULTS / relative
    if not path.exists():
        path = DATA / Path(relative).name
    b = path.read_bytes()
    key = 'experiments/sensor_recovery/results/' + relative
    digest = hashlib.sha256(b).hexdigest()
    manifest = DATA / 'source_hashes.json'
    if manifest.exists():
        expected = json.loads(manifest.read_text())
        assert expected['evidence_commit'] == SNAPSHOT
        assert expected['sha256'][key] == digest, f'Figure input changed: {key}'
    hashes[key] = digest
    inputs[Path(relative).name] = b
    return json.loads(b)


def save(fig, name):
    for ext in ("pdf", "svg", "png"):
        meta = {"CreationDate":None, "ModDate":None} if ext == "pdf" else {"Date":None} if ext == "svg" else None
        fd, temp = tempfile.mkstemp(prefix=f'.{name}_', suffix=f'.{ext}', dir=FIG)
        os.close(fd)
        try:
            fig.savefig(temp, format=ext, dpi=300, bbox_inches="tight", pad_inches=.035, metadata=meta)
            assert Path(temp).stat().st_size > 1000
            if ext == 'pdf':
                assert Path(temp).read_bytes().rstrip().endswith(b'%%EOF')
            os.replace(temp, FIG / f"{name}.{ext}")
        finally:
            if os.path.exists(temp): os.unlink(temp)
    plt.close(fig)


def mean(scenario, budget, cap, method):
    return next(r for r in static if (r["scenario"], r["budget"], r["cap_bytes"], r["method"]) ==
                (scenario, budget, cap, method))["noisy_mae"]["mean"]


def box(ax, x, y, w, h, title, body, color):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.012,rounding_size=.025",
                              facecolor=color,edgecolor="#CFD7DF",linewidth=.7))
    ax.text(x+w/2,y+h*.73,title,ha="center",va="center",fontweight="bold",fontsize=8)
    ax.text(x+w/2,y+h*.33,body,ha="center",va="center",fontsize=7.2,linespacing=1.35)


def protocol():
    fig,ax=plt.subplots(figsize=(7.05,1.65));ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis("off")
    box(ax,.012,.16,.22,.72,"SYNTHETIC REFERENCES","NTC / distorted NTC / LDR\n30 or 75 initial labels\n16-conversion averages", "#EAF3F9")
    box(ax,.27,.55,.28,.34,"STATIC COMPARISON","80 / 144 / 224 bytes; 3312 fits", "#EEF5ED")
    box(ax,.27,.10,.28,.34,"SEQUENTIAL ADAPTATION","384 labels; 3888 trajectories", "#FFF1E4")
    box(ax,.60,.16,.18,.72,"FROZEN CASE","Same data and kernels\nfloat32 validation\n0.002 °C gate", "#F1EFF6")
    box(ax,.83,.16,.155,.72,"PHYSICAL MCU","Mega + C3\n2 sessions each\nTime, not sensor error", "#EAF3F9")
    for a,b in [((.235,.55),(.267,.72)),((.235,.45),(.267,.27)),((.555,.72),(.597,.56)),((.555,.27),(.597,.43)),((.783,.52),(.827,.52))]:
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=8,color="#52616E",lw=.9))
    fig.subplots_adjust(left=0,right=1,top=1,bottom=0);save(fig,"protocol")


def payload():
    fig,axes=plt.subplots(2,3,figsize=(7.05,3.2))
    methods=[("lut_float32","LUT float32"),("lut_int16","LUT int16"),
             ("cubic_compact","Compact cubic"),("cubic_horner","Horner pieces")]
    names={"ntc":"NTC", "ntc_frontend":"Distorted NTC", "ldr":"LDR"}
    for row,B in enumerate((30,75)):
        for col,sc in enumerate(names):
            ax=axes[row,col]
            for m,label in methods:
                ax.plot([80,144,224],[mean(sc,B,c,m) for c in (80,144,224)],"o-",lw=1.3,ms=3,label=label,color=COL[m])
            ax.set_title(f"{names[sc]}, B = {B}",loc="left")
            ax.set_xticks([80,144,224]);ax.grid(alpha=.17);ax.set_axisbelow(True)
            ax.set_ylabel("MAE (log₁₀ lux)" if sc=="ldr" else "MAE (°C)")
            if row==1:ax.set_xlabel("Model payload (bytes)")
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc="upper center",ncol=4,bbox_to_anchor=(.5,1.035),frameon=False)
    fig.subplots_adjust(left=.07,right=.99,bottom=.13,top=.85,wspace=.43,hspace=.65)
    save(fig,"payload_accuracy")


def crossover():
    rows=[(s,b) for s in ("ntc","ntc_frontend","ldr") for b in (30,75)]
    ratios=np.array([[mean(s,b,c,"lut_float32")/mean(s,b,c,"cubic_horner") for c in (80,144,224)] for s,b in rows])
    fig,ax=plt.subplots(figsize=(3.36,2.25))
    ax.imshow(ratios,cmap="RdBu_r",norm=TwoSlopeNorm(vmin=.5,vcenter=1,vmax=1.5),aspect="auto")
    ax.set_xticks(range(3),["80 B","144 B","224 B"])
    ax.set_yticks(range(6),[f"{ {'ntc':'NTC','ntc_frontend':'Dist. NTC','ldr':'LDR'}[s]}, B={b}" for s,b in rows])
    for i in range(6):
        for j in range(3):ax.text(j,i,f"{ratios[i,j]:.3g}",ha="center",va="center",fontsize=8,color="white" if abs(ratios[i,j]-1)>.3 else "#16212B")
    ax.set_title("MAE(LUT float32) / MAE(Horner)",pad=10,fontsize=9)
    ax.tick_params(length=0);ax.set_xlabel("< 1: LUT lower error     > 1: Horner lower error",fontsize=7)
    save(fig,"memory_crossover")


def adaptation():
    fig,axes=plt.subplots(1,3,figsize=(7.05,2.35),sharey=True)
    methods=[("frozen_lut","Frozen LUT"),("lut_projected_nlms","LUT + PAVA"),
             ("cubic_projected_nlms","Cubic + PAVA"),("cubic_rls","Cubic RLS")]
    for ax,drift in zip(axes,("none","step","ramp")):
        for method,label in methods:
            vals=[v for v in curves if v["scenario"]=="ntc_frontend" and v["budget"]==75 and v["drift"]==drift and v["method"]==method]
            vals=sorted(vals,key=lambda v:v["t"])
            ax.plot([v["t"] for v in vals],[v["mean_mae"] for v in vals],label=label,color=COL[method],lw=1.3,linestyle="--" if method=="frozen_lut" else "-")
        ax.set_yscale("log");ax.set_ylim(.037,.8);ax.set_xticks([0,128,256,384])
        ax.set_title({"none":"(a) No drift", "step":"(b) Step drift", "ramp":"(c) Ramp drift"}[drift],loc="left")
        if drift!="none":ax.axvline(128,color="#888888",ls=":",lw=.8)
        ax.grid(alpha=.17,which="major");ax.set_xlabel("Sequential labelled references")
    axes[0].set_ylabel("Mean grid MAE (°C; log scale)")
    fig.legend(*axes[0].get_legend_handles_labels(),loc="upper center",ncol=4,bbox_to_anchor=(.5,1.04),frameon=False)
    fig.subplots_adjust(left=.07,right=.99,bottom=.21,top=.80,wspace=.16);save(fig,"adaptation")


STATIC_M=["lut32_float32","lut60_int16","cubic32_unrolled","horner8"]
STATIC_L=["LUT32","LUT60 int16","Cubic32","Horner8"]
STATIC_C=[COL[m] for m in ("lut_float32","lut_int16","cubic_compact","cubic_horner")]


def hardware_static():
    fig,axes=plt.subplots(1,2,figsize=(7.05,2.5),sharey=True)
    for ax,(board,title) in zip(axes,[("mega2560","(a) Mega2560, 16 MHz"),("esp32c3","(b) ESP32-C3, 160 MHz")]):
        values=hw["boards"][board]["static"]
        for i,(m,label,color) in enumerate(zip(STATIC_M,STATIC_L,STATIC_C)):
            x=values[m]["median_of_session_medians_us"];y=values[m]["metrics"][0]["mae_C"]
            ax.scatter(x,y,s=36,c=color,edgecolors="white",linewidths=.6,zorder=3)
            dx,dy,ha=(-6,8,"right") if i==2 else (5,8,"left")
            if i in (0,3):dy=-12
            ax.annotate(label,(x,y),xytext=(dx,dy),textcoords="offset points",ha=ha,fontsize=7)
        ax.set_xlim(0,max(r["median_of_session_medians_us"] for r in values.values())*1.12)
        ax.set_ylim(.0415,.059);ax.grid(alpha=.18);ax.set_title(title,loc="left")
        ax.set_xlabel("Gross inference time (µs/call)")
    axes[0].set_ylabel("MAE on fixed synthetic case (°C)")
    fig.subplots_adjust(left=.08,right=.99,bottom=.21,top=.86,wspace=.17);save(fig,"hardware_static")


def hardware_updates():
    fig,axes=plt.subplots(1,2,figsize=(7.05,2.25))
    keys=["lut_nlms","lut_projected_nlms","cubic_nlms","cubic_projected_nlms"]
    labels=["LUT", "LUT + PAVA", "Cubic", "Cubic + PAVA"]
    for ax,(board,title) in zip(axes,[("mega2560","(a) Mega2560"),("esp32c3","(b) ESP32-C3")]):
        vals=[hw["boards"][board]["online"]["step"][m]["median_of_session_medians_us"] for m in keys]
        ax.barh(range(4),vals,color=["#0072B2","#56B4E9","#D55E00","#E69F00"],height=.62)
        ax.set_yticks(range(4),labels);ax.invert_yaxis();ax.set_xlim(0,max(vals)*1.25)
        for i,v in enumerate(vals):ax.text(v+max(vals)*.02,i,f"{v:.3g}",va="center",fontsize=7)
        ax.set_xlabel("Gross time (µs/labelled update)");ax.set_title(title,loc="left");ax.grid(axis="x",alpha=.18);ax.set_axisbelow(True)
    fig.subplots_adjust(left=.13,right=.99,bottom=.22,top=.85,wspace=.58);save(fig,"hardware_updates")


def decision_regions():
    fig,axes=plt.subplots(1,2,figsize=(7.05,2.25),sharey=True)
    colors=["#F0F2F4", "#CEE5F3", "#F4D1BC", "#C6E7DC"]
    # Fastest feasible among the four measured candidates. The int16 LUT is
    # dominated by Horner here, so it has no selected region.
    for ax,(board,title,max_time) in zip(axes,[("mega2560","(a) Mega2560",430),("esp32c3","(b) ESP32-C3",33)]):
        xx=np.linspace(0,max_time,500);yy=np.linspace(.041,.061,400)
        z=np.zeros((len(yy),len(xx)),dtype=int)
        vals=hw["boards"][board]["static"]
        candidates=sorted(STATIC_M,key=lambda m:vals[m]["median_of_session_medians_us"],reverse=True)
        for m in candidates:
            idx={"lut32_float32":1,"horner8":2,"cubic32_unrolled":3,"lut60_int16":4}[m]
            t=vals[m]["median_of_session_medians_us"];e=vals[m]["metrics"][0]["mae_C"]
            z[(yy[:,None]>=e)&(xx[None,:]>=t)]=idx
        assert not (z==4).any(),"Unexpected int16 selection region; revise colour mapping"
        ax.pcolormesh(xx,yy,z,cmap=ListedColormap(colors),vmin=0,vmax=3,shading="nearest",rasterized=True)
        for m,label in [("lut32_float32","LUT32"),("horner8","Horner8"),("cubic32_unrolled","Cubic32")]:
            t=vals[m]["median_of_session_medians_us"];e=vals[m]["metrics"][0]["mae_C"]
            ax.plot(t,e,"o",color="#394852",ms=3)
            ax.annotate(label,(t,e),xytext=(4,5),textcoords="offset points",fontsize=7)
        ax.text(max_time*.08,.047,"No feasible\nmeasured candidate",fontsize=7,color="#56616C")
        ax.set_xlim(0,max_time);ax.set_ylim(yy[0],yy[-1]);ax.set_xlabel("Allowed inference time (µs)");ax.set_title(title,loc="left")
    axes[0].set_ylabel("Allowed mean absolute error (°C)")
    fig.subplots_adjust(left=.08,right=.99,bottom=.22,top=.84,wspace=.17);save(fig,"decision_regions")


def tex_tables():
    names={"ntc":"NTC", "ntc_frontend":"Dist. NTC", "ldr":"LDR"}
    static_rows=[]
    for sc in names:
        for B in (30,75):
            vals=[mean(sc,B,144,m) for m in ("lut_float32","lut_int16","cubic_compact","cubic_horner")]
            static_rows.append(f"{names[sc]} & {B} & "+" & ".join(f"{v:.3g}" for v in vals)+r" \\")
    (TABLES/"static144.tex").write_text("\n".join(static_rows)+"\n")
    rows=[]
    for m,label in zip(STATIC_M,STATIC_L):
        a=hw["boards"]["mega2560"]["static"][m];b=hw["boards"]["esp32c3"]["static"][m]
        rows.append(f"{label} & {a['median_of_session_medians_us']:.3g} & {b['median_of_session_medians_us']:.3g} & {b['metrics'][0]['mae_C']:.3g} & {b['metrics'][0]['max_C']:.3g}"+r" \\")
    (TABLES/"hardware.tex").write_text("\n".join(rows)+"\n")
    methods=["lut_float32","lut_int16","monotone_lut_float32","monotone_lut_int16","cubic_compact","cubic_horner","kan_compact"]
    rows=[]
    for sc in names:
        for B in (30,75):
            for c in (80,144,224):
                rows.append(f"{names[sc]} & {B} & {c} & "+" & ".join(f"{mean(sc,B,c,m):.3g}" for m in methods)+r" \\")
    (TABLES/"all_static.tex").write_text("\n".join(rows)+"\n")
    rows=[]
    for sc in names:
        for B in (30,75):
            for d in ("none","step","ramp"):
                vals=[next(x for x in online if (x['scenario'],x['budget'],x['drift'],x['method'])==(sc,B,d,m))['final_mae']['mean'] for m in ("frozen_lut","frozen_cubic","lut_projected_nlms","cubic_projected_nlms","cubic_rls","poly_rls")]
                rows.append(f"{names[sc]} & {B} & {d} & "+" & ".join(f"{v:.3g}" for v in vals)+r" \\")
    (TABLES/"all_online.tex").write_text("\n".join(rows)+"\n")
    rows=[]
    for sc in names:
        for B in (30,75):
            vals=[mean(sc,B,0,m) for m in ("physics","polynomial")]
            rows.append(f"{names[sc]} & {B} & "+" & ".join(f"{v:.3g}" for v in vals)+r" \\")
    (TABLES/"small_references.tex").write_text("\n".join(rows)+"\n")
    rows=[]
    for sc in names:
        for B in (30,75):
            for d in ("none","step","ramp"):
                vals=[next(x for x in online if (x['scenario'],x['budget'],x['drift'],x['method'])==(sc,B,d,m))['final_mae']['mean'] for m in ("frozen_poly","lut_nlms","cubic_nlms")]
                rows.append(f"{names[sc]} & {B} & {d} & "+" & ".join(f"{v:.3g}" for v in vals)+r" \\")
    (TABLES/"other_online.tex").write_text("\n".join(rows)+"\n")
    rows=[]
    for d in ("none","step","ramp"):
        for m,label in zip(("lut_nlms","lut_projected_nlms","cubic_nlms","cubic_projected_nlms"),("LUT NLMS","LUT + PAVA","Cubic NLMS","Cubic + PAVA")):
            a=hw['boards']['mega2560']['online'][d][m]
            b=hw['boards']['esp32c3']['online'][d][m]
            rows.append(f"{d} & {label} & {a['median_of_session_medians_us']:.3g} & {b['median_of_session_medians_us']:.3g} & {b['metrics'][0]['mae_C']:.3g}"+r" \\")
    (TABLES/"all_hardware_updates.tex").write_text("\n".join(rows)+"\n")
    rows=[]
    for sc in names:
        for B in (30,75):
            for cap in (80,144,224):
                r=next(x for x in paired if (x['scenario'],x['budget'],x['cap_bytes'],x['method_a'],x['method_b'])==(sc,B,cap,'lut_float32','cubic_horner'))
                rows.append(f"{names[sc]} & {B} & {cap} & {r['mean_a_minus_b']:.3g} & [{r['ci95_lo']:.3g}, {r['ci95_hi']:.3g}]"+r" \\")
    (TABLES/"paired.tex").write_text("\n".join(rows)+"\n")


if __name__ == "__main__":
    for p in (FIG,TABLES,DATA):p.mkdir(exist_ok=True)
    hashes={}; inputs={}
    static=read("matched_adaptation_20261108/static_summary.json")
    online=read("matched_adaptation_20261108/adaptation_summary.json")
    curves=read("matched_adaptation_20261108/adaptation_curves.json")
    paired=read("matched_adaptation_20261108/static_paired.json")
    hw=read("matched_boards_20261108/comparison.json")
    # Preserve an exact input snapshot so the manuscript ZIP is self-contained.
    for name, content in inputs.items():
        (DATA/name).write_bytes(content)
    (DATA/"source_hashes.json").write_text(json.dumps({"evidence_commit":SNAPSHOT,"sha256":hashes},indent=2)+"\n")
    for fn in (protocol,payload,crossover,adaptation,hardware_static,hardware_updates,decision_regions,tex_tables):fn()
    import make_revision_assets
    make_revision_assets.generate()
    print("Generated revision and historical scientific figures, separate graphical abstract and tables from hashed saved results.")
