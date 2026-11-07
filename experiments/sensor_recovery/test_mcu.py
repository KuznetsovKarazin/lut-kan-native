"""Host checks for the exact kernels that are cross-compiled for both boards."""
import ctypes,json,subprocess,tempfile,unittest
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
class FirmwareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();folder=Path(cls.tmp.name)
        cpp=folder/"test.cpp"
        cpp.write_text('#include "models.h"\nextern "C" float evaluate(int id,float x){return entries[id].predict(x);}\nextern "C" void update(float *c,int n,float x,float y,float rate){monotone_step(c,n,x,y,rate);}\n')
        so=folder/"test.so"
        subprocess.run(["g++","-std=c++11","-O2","-ffp-contract=off","-shared","-fPIC","-I",str(HERE/"mcu/include"),str(cpp),"-o",str(so)],check=True)
        cls.lib=ctypes.CDLL(str(so));cls.lib.evaluate.argtypes=[ctypes.c_int,ctypes.c_float];cls.lib.evaluate.restype=ctypes.c_float
        cls.lib.update.argtypes=[ctypes.POINTER(ctypes.c_float),ctypes.c_int,ctypes.c_float,ctypes.c_float,ctypes.c_float]
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def test_all_firmware_models_against_host_reference(self):
        data=json.loads((HERE/"mcu/host_parity.json").read_text())
        for m in data["models"]:
            with self.subTest(model=m["name"]):
                pred=np.array([self.lib.evaluate(m["id"],x) for x in data["parity_adc"]])
                expected=np.array(data["parity_expected"][m["name"]])
                np.testing.assert_allclose(pred,expected,rtol=0,atol=.002)
    def test_local_update_preserves_global_monotonicity(self):
        rng=np.random.default_rng(411);c=np.linspace(1,-1,32,dtype=np.float32)
        ptr=c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        for i in range(2000):
            self.lib.update(ptr,32,rng.uniform(-2,2),rng.uniform(-3,3),rng.uniform(0,1))
            self.assertTrue(np.all(np.diff(c)<=0))
            self.assertTrue(np.all(np.isfinite(c)))
    def test_local_projection_pools_before_boundary_clipping(self):
        # Nodes 1,2 become (-10,4) before projection. Their constrained
        # least-distance pair is (0,0), not the (2,2) from clip-then-pool.
        c=np.array([10,5,4,0,-1],dtype=np.float32)
        self.lib.update(c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),5,-.5,0,1.5)
        np.testing.assert_allclose(c,[10,0,0,0,-1],atol=1e-6)

if __name__=="__main__":unittest.main()
