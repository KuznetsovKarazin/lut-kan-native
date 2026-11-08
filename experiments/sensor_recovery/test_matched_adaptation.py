import copy
import unittest
import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import minimize
from calibration import Model,Transform,spline_knots,make_sensor,calibration_case
from matched_adaptation import *


class MatchedAdaptationTests(unittest.TestCase):
    def setUp(self):
        self.tr=Transform(.5,.5,50.,50.)

    def test_horner_independent_spline_parity_and_knots(self):
        rng=np.random.default_rng(91)
        for pieces in (4,8,13):
            c=rng.normal(size=pieces+3).astype(np.float32)
            m=HornerModel.from_coefficients(c,self.tr)
            knots=np.linspace(-1,1,pieces+1)
            x=np.r_[-3,3,np.linspace(-1,1,2001),knots,
                     np.nextafter(knots,-np.inf),np.nextafter(knots,np.inf)]
            ref=BSpline(spline_knots(len(c)),c.astype(float),3)(np.clip(x,-1,1))
            np.testing.assert_allclose(m.predict(x),ref,atol=2e-6,rtol=2e-6)
            self.assertLess(m.boundary_gaps()["value"],2e-6)
            self.assertLess(m.boundary_gaps()["derivative"],3e-5)

    def test_entire_static_inventory_and_exact_serialized_budgets(self):
        case=calibration_case(make_sensor("ntc",0,123),30,0,123)
        found=list(static_models(case))
        self.assertEqual(len(found),23)
        for cap,name,m in found:
            raw=serialized_payload(m)
            self.assertEqual(len(raw),m.payload_bytes(),name)
            if cap:self.assertEqual(len(raw),cap,name)
            else:self.assertLessEqual(len(raw),80)
            self.assertTrue(np.isfinite(m.predict(case["x_noisy"])).all())

    def test_pava_against_independent_constrained_minimizer(self):
        c=np.array([1.2,1.3,-.4,.2,-.9,.1])
        out=project_decreasing(c)
        ref=minimize(lambda z:np.sum((z-c)**2),np.sort(c)[::-1],
            jac=lambda z:2*(z-c),constraints={"type":"ineq","fun":lambda z:-np.diff(z)},
            method="SLSQP",options={"ftol":1e-12,"maxiter":200})
        self.assertTrue(ref.success)
        np.testing.assert_allclose(out,ref.x,atol=2e-7)
        self.assertTrue(np.all(np.diff(out)<=0))
        np.testing.assert_array_equal(project_decreasing(out),out)

    def test_nlms_update_matches_formula_and_dtype(self):
        m=Model("lut",np.array([.1,.2,.3,.4],np.float32),self.tr,"ntc")
        s=OnlineState.initialize(m,"lut_nlms",.2)
        b=np.array([0,.2,.8,0]);label=.9;c=s.coefficients.astype(float)
        expected=(c+float(np.float32(.2))*(label-b@c)*b/(b@b+1e-6)).astype(np.float32)
        before=s.step(b,label)
        self.assertAlmostEqual(before,float(b@c))
        np.testing.assert_array_equal(s.coefficients,expected)
        self.assertEqual(s.coefficients.dtype,np.float32)

    def test_rls_equals_regularized_batch_with_prior(self):
        rng=np.random.default_rng(92);a=rng.normal(size=(70,6));y=rng.normal(size=70)
        prior=rng.normal(size=6).astype(np.float32)
        m=Model("poly",prior,self.tr,"ntc")
        s=OnlineState.initialize(m,"poly_rls",1.)
        for b,label in zip(a,y):s.step(b,label)
        gram=a.T@a+.01*np.eye(6)
        ref=np.linalg.solve(gram,a.T@y+.01*prior)
        np.testing.assert_allclose(s.coefficients,ref,atol=2e-6,rtol=2e-5)
        np.testing.assert_allclose(s.covariance,np.linalg.inv(gram),atol=2e-7,rtol=2e-5)

    def test_projected_cubic_sufficient_shape_and_memory(self):
        rng=np.random.default_rng(93)
        m=Model("cubic",rng.normal(size=32).astype(np.float32),self.tr,"ntc",{"n_coeff":32})
        s=OnlineState.initialize(m,"cubic_projected_nlms",.5)
        for x in rng.uniform(-1,1,60):s.step(cubic_basis([x],32)[0],rng.normal())
        der=BSpline(spline_knots(32),s.coefficients.astype(float),3).derivative()(np.linspace(-1,1,2001))
        self.assertLessEqual(float(der.max()),1e-9)
        self.assertEqual(s.persistent_bytes(),148)
        self.assertEqual(OnlineState.initialize(m,"cubic_rls",.995).persistent_bytes(),4244)
        self.assertEqual(OnlineState.initialize(m,"frozen_cubic").persistent_bytes(),144)

    def test_stream_prefix_and_drift_boundaries(self):
        case=calibration_case(make_sensor("ntc",1,94),30,1,94)
        short=generate_stream(case,1,30,"step",94,steps=80)
        long=generate_stream(case,1,30,"step",94,steps=160)
        np.testing.assert_array_equal(short["x"],long["x"][:80])
        np.testing.assert_array_equal(short["y"],long["y"][:80])
        np.testing.assert_array_equal(drift_level("step",[127,128,129]),[0,1,1])
        np.testing.assert_array_equal(drift_level("ramp",[127,128,192,256,300]),[0,0,.5,1,1])
        self.assertNotEqual(long["x"].tobytes(),generate_stream(case,1,30,"step",95,steps=160)["x"].tobytes())

    def test_causality_future_labels_cannot_change_prefix(self):
        case=calibration_case(make_sensor("ntc",2,96),30,0,96)
        initial=initial_models(case)
        stream=generate_stream(case,0,30,"step",96,steps=160)
        changed=copy.deepcopy(stream);changed["y"][80:]+=10
        for method,p in (("lut_nlms",.5),("cubic_projected_nlms",.2),("poly_rls",.995)):
            a=trajectory(initial,stream,method,p)
            b=trajectory(initial,changed,method,p)
            self.assertEqual(a["prequential"][:80],b["prequential"][:80])
            self.assertEqual([x for x in a["snapshots"] if x["t"]<=80],
                             [x for x in b["snapshots"] if x["t"]<=80])

    def test_evaluation_truth_never_changes_learned_state(self):
        case=calibration_case(make_sensor("ldr",2,97),30,0,97)
        initial=initial_models(case)
        stream=generate_stream(case,0,30,"ramp",97,steps=144)
        changed=copy.deepcopy(stream);changed["q_truth"][:]=999
        for v in changed["tests"].values():v["truth"][:]=999
        a=trajectory(initial,stream,"cubic_projected_nlms",.2)
        b=trajectory(initial,changed,"cubic_projected_nlms",.2)
        self.assertEqual(a["prequential"],b["prequential"])
        self.assertEqual([x["state"] for x in a["snapshots"]],[x["state"] for x in b["snapshots"]])
        self.assertNotEqual(a["final_mae"],b["final_mae"])

    def test_test_arrays_cannot_change_initial_model_selection(self):
        case=calibration_case(make_sensor("ldr",0,98),30,0,98)
        a=initial_models(case)
        case["y_test"][:]=999;case["x_noisy"][:]=999
        b=initial_models(case)
        for name in a:
            np.testing.assert_array_equal(a[name].coefficients,b[name].coefficients)
            self.assertEqual(a[name].hyper,b[name].hyper)

    def test_replay_is_exact_and_frozen_state_unchanged(self):
        case=calibration_case(make_sensor("ntc_frontend",1,99),30,0,99)
        initial=initial_models(case);s=generate_stream(case,0,30,"step",99,steps=144)
        a=trajectory(initial,s,"lut_projected_nlms",.5)
        self.assertEqual(a,trajectory(initial,s,"lut_projected_nlms",.5))
        f=trajectory(initial,s,"frozen_lut")
        self.assertEqual(f["consumed_update_labels"],0)
        self.assertEqual(f["completed_steps"],144)
        self.assertTrue(all(x["state"]==f["snapshots"][0]["state"] for x in f["snapshots"]))


if __name__=="__main__":unittest.main()
