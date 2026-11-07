import unittest
import numpy as np
from calibration import Transform,make_sensor,calibration_case,linear_basis,regularized_fit
from robust import irregular_basis,slope_penalty,solve_shape,fit_shape,encode_shape

class ShapeTests(unittest.TestCase):
    def test_nonuniform_affine_exact_and_partition(self):
        knots=np.array([-1,-.8,-.2,.1,1]);x=np.linspace(-1,1,103)
        a=irregular_basis(x,knots)
        np.testing.assert_allclose(a.sum(1),1)
        np.testing.assert_allclose(a@(.3-.7*knots),.3-.7*x,atol=1e-14)
        np.testing.assert_allclose(slope_penalty(knots)@knots,0,atol=1e-14)
    def test_uniform_matches_original_optimizer(self):
        x=np.linspace(-1,1,51);y=np.tanh(-3*x);a=linear_basis(x,16);d=np.diff(np.eye(16),n=2,axis=0)
        np.testing.assert_allclose(solve_shape(a,y,d,.01,False),regularized_fit(a,y,.01),atol=1e-12)
    def test_monotone_fit_and_encoding_and_grid_bytes(self):
        c=calibration_case(make_sensor("ntc",5),30,0);t,v=c["train_idx"],c["val_idx"]
        for irregular in (False,True):
            m,k=fit_shape(c["x"][t],c["y"][t],c["x"][v],c["y"][v],c["transform"],"ntc",32,True,irregular)
            for fmt in ("float32","uint8","int16"):
                e=encode_shape(m,k,fmt,irregular)
                self.assertLessEqual(np.max(np.diff(e.predict(np.linspace(-1,1,10001)))),1e-10)
                self.assertEqual(e.payload_bytes(),e.table.payload_bytes()+(128 if irregular else 0))
    def test_constraint_preserves_feasible_exact_line(self):
        x=np.linspace(-1,1,100);a=linear_basis(x,16);d=np.diff(np.eye(16),n=2,axis=0)
        c=solve_shape(a,.2-.8*x,d,1,True)
        np.testing.assert_allclose(a@c,.2-.8*x,atol=1e-12)

if __name__=="__main__":unittest.main()
