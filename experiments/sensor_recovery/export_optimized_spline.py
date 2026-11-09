"""Compile the already frozen float32 splines, without training or selection."""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import scipy
from scipy.interpolate import BSpline,PPoly

HERE=Path(__file__).resolve().parent
INCLUDE=HERE/"mcu/include"


def read_array(name):
    text=(INCLUDE/"fixture.h").read_text()
    match=re.search(r"static const float "+re.escape(name)+r"\[\d+\](?: FIXTURE_FLASH)? = \{(.*?)\};",text,re.S)
    if not match:raise ValueError("Missing frozen array: "+name)
    return np.array([float(x.strip().removesuffix("f")) for x in match[1].split(",") if x.strip()],dtype=np.float32)


def compile_horner(coefficients):
    coefficients=np.asarray(coefficients,dtype=np.float32)
    if coefficients.shape!=(16,):raise ValueError("Exactly 16 cubic coefficients required")
    knots=np.r_[np.full(3,-1.),np.linspace(-1,1,14),np.full(3,1.)]
    pp=PPoly.from_spline(BSpline(knots,coefficients.astype(np.float64),3))
    selected=np.flatnonzero(np.diff(pp.x)>0)
    assert len(selected)==13
    widths=np.diff(pp.x)[selected]
    local=pp.c[:,selected]*widths[None,:]**np.arange(3,-1,-1)[:,None]
    return np.ascontiguousarray(local.T,dtype=np.float32)


def main():
    arrays={"cubic_horner_c":compile_horner(read_array("cubic_c")),
            "kan_horner_c":compile_horner(read_array("kan_c")[:16])}
    lines=["// Generated from frozen fixture.h, not re-fitted. 13 local cubic pieces.","#pragma once"]
    for name,values in arrays.items():
        if name=="kan_horner_c":lines.append("#ifdef SENSOR_OPTIMIZED_DIAGNOSTICS")
        lines.append(f"static const float {name}[52] = {{")
        for row in values:
            values_text=[]
            for x in row:
                s=format(float(x),".9g")
                if "." not in s and "e" not in s:s+=".0"
                values_text.append(s+"f")
            lines.append("  "+", ".join(values_text)+",")
        lines.append("};")
        if name=="kan_horner_c":lines.append("#endif")
    header="\n".join(lines)+"\n"
    (INCLUDE/"optimized_coefficients.h").write_text(header)
    original=json.loads((HERE/"mcu/fixture_manifest.json").read_text())
    additions=[dict(id=9,name="cubic16_unrolled",payload_bytes=80,reference_id=3),
               dict(id=10,name="cubic16_horner",payload_bytes=224,reference_id=3)]
    diagnostics=[dict(id=11,name="kan16_unrolled",payload_bytes=84,reference_id=4),
                 dict(id=12,name="kan16_horner",payload_bytes=228,reference_id=4)]
    manifest=dict(original,suite="spline_optimized_v1",models=original["models"]+additions,diagnostic_only_models=diagnostics,
                  source_fixture_sha256=hashlib.sha256((INCLUDE/"fixture.h").read_bytes()).hexdigest(),
                  compiled_header_sha256=hashlib.sha256(header.encode()).hexdigest(),
                  compiler="SciPy PPoly.from_spline; float64 conversion then float32 storage; local interval coordinate",
                  scipy_version=scipy.__version__,numpy_version=np.__version__,retrained=False)
    (HERE/"mcu/optimized_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print("Compiled two frozen splines into 13 local cubic pieces each; no training or selection.")


if __name__=="__main__":main()
