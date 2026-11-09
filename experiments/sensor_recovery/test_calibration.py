"""Scientific invariants and deployment regression tests (stdlib unittest)."""
import ctypes
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from scipy.interpolate import BSpline
from calibration import *


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.tr = Transform(0.5,0.5,50,50)

    def test_linear_identity_and_endpoint_partition(self):
        for length in (4,8,32):
            x=np.r_[-1,np.linspace(-.99,.99,57),1]
            a=linear_basis(x,length)
            knots=np.r_[-1,np.linspace(-1,1,length),1]
            b=BSpline(knots,np.eye(length),1)(x)
            np.testing.assert_allclose(a,b,atol=3e-15)
            np.testing.assert_allclose(a.sum(axis=1),1,atol=1e-15)
            np.testing.assert_allclose(a@np.linspace(-1,1,length),x,atol=1e-15)

    def test_exact_fit_beats_adam_same_loss(self):
        x=np.linspace(-1,1,75);y=np.tanh(3*x)
        a=linear_basis(x,32)
        ls=regularized_fit(a,y,1)
        ad=adam_fit(a,y,1,epochs=200)
        self.assertLessEqual(objective(a,y,ls,1),objective(a,y,ad,1)+1e-12)
        # Finite-difference check of the stationarity of the exact solution.
        d=np.diff(np.eye(32),n=2,axis=0)
        g=2*a.T@(a@ls-y)/len(y)+2*d.T@(d@ls)/len(d)
        self.assertLess(np.max(abs(g)),1e-12)

    def test_quantization_roundtrip_error_bound_and_actual_bytes(self):
        m=Model("lut",np.sin(np.linspace(-1,1,32)).astype(np.float32),self.tr,"ntc")
        for fmt,bytes_ in (("float32",144),("uint8",56),("int16",88)):
            enc=EncodedLUT.from_model(m,fmt)
            self.assertEqual(enc.payload_bytes(),bytes_)
            if fmt != "float32":
                self.assertLessEqual(np.max(abs(enc.decoded()-m.coefficients)),float(enc.scale)/2+2e-7)

    def test_constant_quantization_and_clipping(self):
        m=Model("lut",np.full(8,.123,np.float32),self.tr,"ntc")
        for fmt in ("float32","uint8","int16"):
            enc=EncodedLUT.from_model(m,fmt)
            np.testing.assert_allclose(enc.predict([-3,0,3]),.123,atol=1e-8)

    def test_scalar_kan_contains_linear_function(self):
        x=np.linspace(-1,1,60); y=.2+.3*x
        a=cubic_basis(x,8,True)
        c=regularized_fit(a,y,0,8)
        np.testing.assert_allclose(a@c,y,atol=1e-12)

    def test_budget_splits_and_reproducibility(self):
        s=make_sensor("ntc",2)
        a=calibration_case(s,75,1);b=calibration_case(s,75,1)
        self.assertEqual(a["reference_sha256"],b["reference_sha256"])
        self.assertEqual(len(a["train_idx"])+len(a["val_idx"]),75)
        self.assertFalse(set(a["train_idx"]) & set(a["val_idx"]))
        self.assertEqual(len(a["y_test"]),401)
        self.assertNotEqual(a["reference_sha256"],calibration_case(s,75,2)["reference_sha256"])

    def test_no_test_data_in_model_selection(self):
        c=calibration_case(make_sensor("ldr",1),30,0)
        t,v=c["train_idx"],c["val_idx"]
        args=(c["x"][t],c["y"][t],c["x"][v],c["y"][v],c["transform"],"ldr")
        a=fit_selected("lut",*args,n_coeff=16)
        c["y_test"][:]=999
        b=fit_selected("lut",*args,n_coeff=16)
        np.testing.assert_array_equal(a.coefficients,b.coefficients)

    def test_physics_exact_beta_noiseless(self):
        s=make_sensor("ntc",3)
        q=np.linspace(0,100,75); adc=s.adc(q)
        tr=Transform.from_references(adc,s.q_range)
        m=fit_physics(tr.nx(adc),tr.ny(q),tr,"ntc")
        self.assertLess(np.max(abs(tr.dy(m.predict(tr.nx(adc)))-q)),.001)

    def test_compilation_uses_identical_grid(self):
        m=Model("poly",np.array([.2,.5],np.float32),self.tr,"ntc")
        lut=compile_lut(m,16)
        x=np.linspace(-1,1,203)
        np.testing.assert_allclose(lut.predict(x),m.predict(x),atol=5e-8)

    def test_legacy_onepoint_ignores_all_other_references(self):
        path=Path(__file__).resolve().parents[2]/"src/lut_native/sensors.py"
        spec=importlib.util.spec_from_file_location("sensor_legacy_under_test",path)
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module
        spec.loader.exec_module(module)
        x=np.linspace(-1,1,10);y=x+0.7
        kw=dict(factory_predict=lambda z:np.asarray(z),point_index=4)
        a=module.fit_onepoint_offset_baseline(x,y,x,y,**kw)
        changed=y.copy();changed[np.arange(10)!=4]=100
        b=module.fit_onepoint_offset_baseline(x,changed,x,y,**kw)
        np.testing.assert_array_equal(a["y_pred"],b["y_pred"])
        self.assertAlmostEqual(a["delta"],.7,places=6)
        self.assertEqual(a["n_reference_used"],1)
        with self.assertRaises(ValueError):
            module.fit_onepoint_offset_baseline(x,y,x,y)

    def test_c_float32_integer_kernels(self):
        source=Path(__file__).resolve().parent/"lut_kernel.c"
        with tempfile.TemporaryDirectory() as tmp:
            so=Path(tmp)/"kernel.so"
            subprocess.run(["cc","-O2","-shared","-fPIC",str(source),"-lm","-o",str(so)],check=True)
            lib=ctypes.CDLL(str(so))
            m=Model("lut",np.sin(np.linspace(-1,1,32)).astype(np.float32),self.tr,"ntc")
            x=np.r_[-3,-1,np.linspace(-.999,.999,129),1,3].astype(np.float32)
            for fmt,ctype in (("float32",ctypes.c_float),("uint8",ctypes.c_uint8),("int16",ctypes.c_int16)):
                enc=EncodedLUT.from_model(m,fmt)
                fn=getattr(lib,"lut_"+fmt)
                fn.restype=ctypes.c_float
                fn.argtypes=[ctypes.c_float,ctypes.POINTER(ctype),ctypes.c_uint16,ctypes.c_float,ctypes.c_float]
                ptr=enc.values.ctypes.data_as(ctypes.POINTER(ctype))
                pred=np.array([fn(float(z),ptr,len(enc.values),float(enc.scale),float(enc.offset)) for z in x])
                np.testing.assert_allclose(pred,enc.predict(x),atol=3e-7)


if __name__=="__main__":
    unittest.main(verbosity=2)
