"""Independent basis/projection checks and complete float32 deployment trajectories."""
import json
from pathlib import Path
import numpy as np
import pytest
from scipy.interpolate import BSpline
from scipy.optimize import isotonic_regression
from calibration import spline_knots
from matched_host import compile_library,validate_fixture,PTR
import export_matched_mcu

HERE=Path(__file__).resolve().parent


@pytest.fixture(scope="module",params=["-O2","-Os"])
def kernel(request,tmp_path_factory):
    return compile_library(tmp_path_factory.mktemp("matched"),request.param)[0]


def test_basis_boundaries_and_arbitrary_vectors(kernel):
    edges=np.linspace(-1,1,30,dtype=np.float32)
    rng=np.random.default_rng(2026110811)
    x=np.r_[[-2.,2.],edges,np.nextafter(edges,np.float32(-np.inf)),
            np.nextafter(edges,np.float32(np.inf)),rng.uniform(-1,1,200)].astype(np.float32)
    expected=BSpline(spline_knots(32),np.eye(32),3)(np.clip(x,-1,1).astype(float))
    actual=np.zeros((len(x),32),np.float32)
    for row,v in zip(actual,x):kernel.matched_basis(v,row.ctypes.data_as(PTR))
    np.testing.assert_allclose(actual,expected,atol=3e-6,rtol=0)
    np.testing.assert_allclose(actual.sum(axis=1),1,atol=3e-7,rtol=0)
    assert actual.min()>=0
    for c in np.r_[np.eye(32),rng.uniform(-2,2,(8,32))].astype(np.float32):
        prediction=np.array([kernel.matched_cubic(v,c.ctypes.data_as(PTR)) for v in x])
        np.testing.assert_allclose(prediction,expected@c,atol=8e-6,rtol=0)


def test_complete_pava_against_independent_scipy(kernel):
    rng=np.random.default_rng(2026110812)
    values=np.r_[rng.normal(size=(100,32)),np.ones((1,32)),np.arange(32)[None,:],
                 -np.arange(32)[None,:],np.tile([1.,1.+1e-6],(1,16))].astype(np.float32)
    for row in values:
        expected=isotonic_regression(row.astype(float),increasing=False).x
        c=row.copy();kernel.matched_project(c.ctypes.data_as(PTR))
        np.testing.assert_allclose(c,expected,atol=3e-6,rtol=0)
        assert np.all(np.diff(c)<=0)
        old=c.copy();kernel.matched_project(c.ctypes.data_as(PTR));np.testing.assert_array_equal(c,old)


def test_full_fixture_and_all_float32_trajectories(kernel):
    result=validate_fixture(kernel)
    assert len(result["static"])==4 and len(result["online"])==12


def test_export_is_byte_reproducible(tmp_path,monkeypatch):
    (tmp_path/"include").mkdir()
    monkeypatch.setattr(export_matched_mcu,"OUT",tmp_path)
    export_matched_mcu.export()
    for name in ["include/matched_fixture.h","matched_manifest.json","matched_host_reference.json"]:
        assert (tmp_path/name).read_bytes()==(HERE/"mcu"/name).read_bytes()
