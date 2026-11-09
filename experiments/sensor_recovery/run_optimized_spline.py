"""Frozen-function compilation pilot: host numbers are never MCU timings."""
import argparse
import csv
import ctypes
import hashlib
import io
import json
import platform
import re
import statistics
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.interpolate import BSpline

from export_optimized_spline import HERE,INCLUDE,read_array


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def scalar(name):
    text=(INCLUDE/"fixture.h").read_text()
    return np.float32(re.search(r"static const float "+name+r" = (.*?);",text)[1].removesuffix("f"))


def evaluate(lib,mid,adc):
    return np.array([lib.evaluate(mid,float(x)) for x in adc],dtype=np.float64)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--calls",type=int,default=262144);p.add_argument("--batches",type=int,default=15)
    args=p.parse_args()
    if args.out.exists():p.error("Choose a fresh output directory")
    if args.calls<2048 or args.batches<3:p.error("Use at least 2048 calls and three batches")
    args.out.mkdir(parents=True)
    manifest=json.loads((HERE/"mcu/optimized_manifest.json").read_text())
    reference=json.loads((HERE/"mcu/host_parity.json").read_text())
    models=manifest["models"]+manifest["diagnostic_only_models"]
    expected_names={m["id"]:m["name"] for m in manifest["models"] if m["id"]<=8}
    adc=read_array("fixture_adc");truth=read_array("fixture_truth").astype(float)
    edges=np.linspace(-1,1,14,dtype=np.float32)
    normalized=np.r_[np.linspace(-1.5,1.5,20001),edges,np.nextafter(edges,np.float32(-np.inf)),np.nextafter(edges,np.float32(np.inf))]
    dense_adc=np.asarray(scalar("x_mid")+scalar("x_half")*normalized,dtype=np.float32)
    dense_x=np.clip((dense_adc.astype(float)-float(scalar("x_mid")))/float(scalar("x_half")),-1,1)
    knots=np.r_[np.full(3,-1.),np.linspace(-1,1,14),np.full(3,1.)]
    frozen_refs={}
    for name in ("cubic_c","kan_c"):
        c=read_array(name).astype(float)
        y=BSpline(knots,c[:16],3)(dense_x)
        if name=="kan_c":y+=c[16]*dense_x/(1+np.exp(-dense_x))
        frozen_refs[name]=y*float(scalar("y_half"))+float(scalar("y_mid"))
    result=dict(evidence_kind="host_execution_and_numerical_pilot_not_MCU_timing",retrained=False,
                fixture_points=len(adc),legacy_gate_points=len(reference["parity_adc"]),dense_points=len(dense_adc),
                parity_gate_C=.002,host_system=platform.system(),host_machine=platform.machine(),
                compiler=subprocess.check_output(["g++","--version"],text=True).splitlines()[0],
                calls_per_batch=args.calls,batches=args.batches,profiles={})
    cpuinfo=Path("/proc/cpuinfo")
    if cpuinfo.exists():
        m=re.search(r"model name\s*:\s*(.*)",cpuinfo.read_text())
        if m:result["host_cpu"]=m[1]
    with tempfile.TemporaryDirectory() as t:
        folder=Path(t);wrapper=folder/"wrapper.cpp"
        wrapper.write_text('#include "models.h"\nextern "C" float evaluate(int id,float x){return entries[id].predict(x);}\n')
        for profile in ("O2","Os"):
            flags=["g++","-std=c++11","-"+profile,"-ffp-contract=off","-DSENSOR_OPTIMIZED_SUITE","-DSENSOR_OPTIMIZED_DIAGNOSTICS","-I",str(INCLUDE)]
            library=folder/(profile+".so");executable=folder/(profile+".bench")
            subprocess.run(flags+["-shared","-fPIC",str(wrapper),"-o",str(library)],check=True)
            lib=ctypes.CDLL(str(library));lib.evaluate.argtypes=[ctypes.c_int,ctypes.c_float];lib.evaluate.restype=ctypes.c_float
            metrics=[]
            for model in models:
                mid=model["id"];refid=model.get("reference_id",mid)
                prediction=evaluate(lib,mid,adc);error=prediction-truth
                gold=read_array("expected_"+str(refid)).astype(float)
                legacy=evaluate(lib,mid,reference["parity_adc"])
                gap=float(np.max(abs(legacy-np.array(reference["parity_expected"][expected_names[refid]]))))
                row=dict(model,mae_C=float(np.mean(abs(error))),rmse_C=float(np.sqrt(np.mean(error**2))),
                         max_C=float(np.max(abs(error))),fixture_max_host_gap_C=float(np.max(abs(prediction-gold))),
                         legacy_grid_max_host_gap_C=gap,legacy_gate_pass=gap<=.002,
                         hardware_suite_member=mid<=10)
                if refid in (3,4):
                    dense=evaluate(lib,mid,dense_adc);frozen=frozen_refs["cubic_c" if refid==3 else "kan_c"]
                    row["dense_frozen_model_max_gap_C"]=float(np.max(abs(dense-frozen)))
                metrics.append(row)
            for m in metrics:
                if m["id"] in (9,10) and not m["legacy_gate_pass"]:raise ValueError("New cubic model failed existing gate")
            subprocess.run(flags+[str(HERE/"optimized_host_bench.cpp"),"-o",str(executable)],check=True)
            raw=subprocess.check_output([str(executable),str(args.calls),str(args.batches)],text=True)
            (args.out/("host_"+profile+"_raw.txt")).write_text(raw)
            batches={}
            for row in csv.DictReader(io.StringIO(raw)):
                item=dict(batch=int(row["batch"]),calls=int(row["calls"]),total_ns=int(row["total_ns"]),checksum=float(row["checksum"]))
                if item["total_ns"]<=0 or not np.isfinite(item["checksum"]):raise ValueError("Invalid host timing record")
                batches.setdefault(row["method"],[]).append(item)
            timing={}
            for name,rows in batches.items():
                if len(rows)!=args.batches or {r["batch"] for r in rows}!=set(range(args.batches)):raise ValueError("Missing host batches")
                values=[r["total_ns"]/r["calls"] for r in rows]
                timing[name]=dict(median_ns_per_call=statistics.median(values),min_ns_per_call=min(values),max_ns_per_call=max(values),batches=rows)
            result["profiles"][profile]=dict(flags=flags,models=metrics,timing=timing,executable_sha256=digest(executable))
            print(profile,{name:round(timing[name]["median_ns_per_call"],3) for name in ("cubic16","cubic16_unrolled","cubic16_horner","uniform32_float32")},flush=True)
    source_paths=[Path(__file__),HERE/"export_optimized_spline.py",HERE/"optimized_host_bench.cpp",*sorted(INCLUDE.glob("*.h")),HERE/"mcu/host_parity.json",HERE/"mcu/optimized_manifest.json"]
    result["source_sha256"]={str(p.relative_to(HERE)):digest(p) for p in source_paths}
    (args.out/"analysis.json").write_text(json.dumps(result,indent=2)+"\n")
    lines=["# Optimized cubic deployment pilot", "", "Host measurements only: no MCU timings are estimated from these numbers. Frozen float32 model coefficients and the same 401-point fixture are used; no training, model selection or test-point selection was performed.","",
           "Two admitted cubic implementations: six unrolled de Boor interpolations (80-byte payload) and 13 precompiled local cubic polynomials evaluated by Horner (224 bytes). Float32 LUT32 uses 144 bytes. KAN variants are retained as host-only numerical diagnostics and are excluded from the ten-model firmware suite.",""]
    for profile,pdata in result["profiles"].items():
        lines += ["## Host -"+profile,"",f"{args.batches} rotated-order batches × {args.calls} calls. These include fixture reads, function dispatch and checksum accumulation; they are not isolated arithmetic costs.","",
                  "| Method | Payload B | Host ns/call | MAE C | Existing 1005-point max gap C | Gate |",
                  "|---|---:|---:|---:|---:|---|"]
        for m in pdata["models"]:
            lines.append(f"| {m['name']} | {m['payload_bytes']} | {pdata['timing'][m['name']]['median_ns_per_call']:.3f} | {m['mae_C']:.7f} | {m['legacy_grid_max_host_gap_C']:.7f} | {'PASS' if m['legacy_gate_pass'] else 'FAIL'} |")
        lines.append("")
    lines += ["## Interpretation", "",
              "- Use actual MCU captures to establish speed on Mega or C3; the host has a different instruction set, floating-point implementation and compiler environment.",
              "- The Horner representation stores 52 float coefficients instead of 16. Its memory increase must accompany any speed claim; the unrolled variant retains the original payload.",
              "- KAN's fitted spline and SiLU terms have large opposing magnitudes. The optimized float32 rearrangements fail the existing 0.002 C host parity gate and are not admitted to hardware timing. The old implementation passes the existing grid but also has larger errors on the expanded frozen-model diagnostic grid; a passing 401-point device check is not an all-input numerical guarantee.",
              "- All candidates, including failed KAN candidates, are retained. No tolerance was relaxed and no coefficient refit was used to admit a method.",
              "- Numerical accuracy is measured on a synthetic sensor fixture, not a physical NTC. Compiler size reports and MCU measurements are separate evidence.", ""]
    (args.out/"RESULTS.md").write_text("\n".join(lines))


if __name__=="__main__":main()
