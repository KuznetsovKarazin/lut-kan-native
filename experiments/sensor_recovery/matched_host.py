"""Build and inspect the actual portable float32 matched-MCU kernels."""
import ctypes
import json
import subprocess
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
PTR=ctypes.POINTER(ctypes.c_float)


def compile_library(out,opt="-O2"):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    library=out/("matched_"+opt[1:]+".so")
    command=["g++",opt,"-std=c++11","-Wall","-Wextra","-Werror","-fno-fast-math",
        "-ffp-contract=off","-fPIC","-shared","-I",str(HERE/"mcu/include"),
        str(HERE/"mcu/matched_host_bridge.cpp"),"-o",str(library)]
    run=subprocess.run(command,capture_output=True,text=True)
    if run.returncode:raise RuntimeError(run.stdout+run.stderr)
    lib=ctypes.CDLL(str(library))
    for name,args,ret in [
        ("matched_predict",[ctypes.c_int,ctypes.c_float],ctypes.c_float),
        ("matched_reset",[ctypes.c_int],None),
        ("matched_step",[ctypes.c_int,ctypes.c_float,ctypes.c_float],ctypes.c_float),
        ("matched_state",[PTR],None),("matched_online_predict",[ctypes.c_int,ctypes.c_float],ctypes.c_float),
        ("matched_project",[PTR],None),("matched_basis",[ctypes.c_float,PTR],None),
        ("matched_cubic",[ctypes.c_float,PTR],ctypes.c_float)]:
        f=getattr(lib,name);f.argtypes=args;f.restype=ret
    return lib,command


def state(lib):
    c=np.empty(32,np.float32);lib.matched_state(c.ctypes.data_as(PTR));return c


def validate_fixture(lib):
    d=json.loads((HERE/"mcu/matched_host_reference.json").read_text())
    half=d["transform"]["y_half"];static=[];online=[]
    for method,name in enumerate(d["static_names"]):
        pred=np.array([lib.matched_predict(method,x) for x in d["adc"]])
        gap=float(np.max(abs(pred-np.asarray(d["static_expected"][method]))))
        if not np.isfinite(pred).all() or gap>.002:raise AssertionError((name,gap))
        static.append(dict(method=name,max_gap_C=gap))
    for stream in d["streams"]:
        for method,name in enumerate(d["online_names"]):
            lib.matched_reset(method);maxgap=0.;check=0;minstep=0
            for t in range(385):
                if t:
                    p=lib.matched_step(method,stream["adc"][t-1],stream["reference"][t-1])
                    if not np.isfinite(p):raise AssertionError("Nonfinite prequential prediction")
                c=state(lib)
                if not np.isfinite(c).all():raise AssertionError("Nonfinite online state")
                if method%2 and np.any(np.diff(c)>0):raise AssertionError("Projection order violation")
                if t in d["checkpoints"]:
                    gap=float(np.max(abs(c-np.array(stream["reference_states"][method][check])))*half)
                    maxgap=max(maxgap,gap);check+=1
            pred=np.array([lib.matched_online_predict(method,x) for x in stream["final_adc"]])
            error=pred-np.array(d["truth"])
            metrics=[float(np.mean(abs(error))),float(np.sqrt(np.mean(error**2))),float(np.max(abs(error)))]
            metricgap=float(np.max(abs(np.array(metrics)-stream["reference_final_metrics"][method])))
            if maxgap>.002 or metricgap>.002:raise AssertionError((name,stream["mode"],maxgap,metricgap))
            online.append(dict(method=name,drift=stream["mode"],max_state_gap_C=maxgap,
                               final_metrics=metrics,max_final_metric_gap_C=metricgap))
    return dict(static=static,online=online,parity_gate_C=.002,status="pass")
