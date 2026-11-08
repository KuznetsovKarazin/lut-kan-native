"""Numerical equivalence, interval boundaries and frozen-fixture release gate."""
import ctypes
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy.interpolate import BSpline

from export_optimized_spline import compile_horner,read_array
from mcu.capture import parse_lines
from test_capture import protocol_fixture

HERE=Path(__file__).resolve().parent


@pytest.fixture(scope="module")
def optimized_lib(tmp_path_factory):
    p=tmp_path_factory.mktemp("optimized_cpp")
    source=p/"test.cpp"
    source.write_text('''#include "models.h"
extern "C" float evaluate(int id,float x){return entries[id].predict(x);}
extern "C" float raw(int mode,float x,const float *c){return mode==0?cubic_eval(x,c,16):mode==1?cubic16_unrolled_eval(x,c):cubic16_horner_eval(x,c);}
''')
    library=p/"models.so"
    subprocess.run(["g++","-O2","-std=c++11","-ffp-contract=off","-DSENSOR_OPTIMIZED_SUITE",
                    "-fPIC","-shared","-I",str(HERE/"mcu/include"),str(source),"-o",str(library)],check=True)
    lib=ctypes.CDLL(str(library));lib.evaluate.argtypes=[ctypes.c_int,ctypes.c_float];lib.evaluate.restype=ctypes.c_float
    lib.raw.argtypes=[ctypes.c_int,ctypes.c_float,ctypes.POINTER(ctypes.c_float)];lib.raw.restype=ctypes.c_float
    return lib


def test_boundaries_clamps_and_arbitrary_control_vectors(optimized_lib):
    knots=np.r_[np.full(3,-1.),np.linspace(-1,1,14),np.full(3,1.)]
    edges=np.linspace(-1,1,14,dtype=np.float32)
    rng=np.random.default_rng(20261108)
    x=np.r_[[-2.,2.],edges,np.nextafter(edges,np.float32(-np.inf)),
            np.nextafter(edges,np.float32(np.inf)),rng.uniform(-1,1,300)].astype(np.float32)
    vectors=np.r_[np.eye(16),rng.uniform(-2,2,(12,16))].astype(np.float32)
    for c in vectors:
        expected=BSpline(knots,c.astype(float),3)(np.clip(x.astype(float),-1,1))
        for mode,coefficients in [(1,c),(2,compile_horner(c).reshape(-1))]:
            ptr=coefficients.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            actual=np.array([optimized_lib.raw(mode,v,ptr) for v in x])
            # Include float32 interval-coordinate and power-basis rounding.
            # This bound remains below the 0.002 C deployment gate at y_half=50.
            bound=64*np.finfo(np.float32).eps*max(1,float(np.max(abs(c))))
            np.testing.assert_allclose(actual,expected,atol=bound,rtol=0)


def test_two_admitted_models_pass_existing_host_gate(optimized_lib):
    data=json.loads((HERE/"mcu/host_parity.json").read_text())
    for mid in (9,10):
        prediction=np.array([optimized_lib.evaluate(mid,x) for x in data["parity_adc"]])
        np.testing.assert_allclose(prediction,data["parity_expected"]["cubic16"],atol=.002,rtol=0)


def test_serialized_horner_coefficients_and_payload():
    compiled=compile_horner(read_array("cubic_c"))
    assert compiled.shape==(13,4) and compiled.nbytes==208
    assert compiled.nbytes+16==224
    header=(HERE/"mcu/include/optimized_coefficients.h").read_text()
    body=re.search(r"cubic_horner_c\[52\] = \{(.*?)\};",header,re.S)[1]
    stored=np.array([float(v.strip().removesuffix("f")) for v in body.split(",") if v.strip()],dtype=np.float32)
    np.testing.assert_array_equal(compiled.reshape(-1),stored)


def test_suite_requires_both_additional_models():
    lines=protocol_fixture();lines.insert(1,"SUITE,spline_optimized_v1")
    with pytest.raises(ValueError,match="Incomplete benchmark"):parse_lines(lines)
    extra=[]
    for method,payload in [("cubic16_unrolled",80),("cubic16_horner",224)]:
        extra.extend([f"PAYLOAD,{method},{payload}",f"ACCURACY,{method},0.1,0.2,0.3,0.00001,PASS"])
        extra.extend(f"TIME,{method},{r},2048,4096,12.0" for r in range(10))
    lines[-1:-1]=extra
    result=parse_lines(lines)
    assert len(result["accuracy"])==10
    lines[1]="SUITE,unknown"
    with pytest.raises(ValueError,match="Unsupported benchmark suite"):parse_lines(lines)
