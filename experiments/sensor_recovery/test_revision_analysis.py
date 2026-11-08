"""Causality and numerical contracts for the post-review analyses."""
import copy
import unittest
import numpy as np
import revision_analysis as r


class RevisionTests(unittest.TestCase):
    def test_nominal_parity_and_inverse(self):
        for kind in r.SCENARIOS:
            s=r.sensor_for(kind,0,r.SEED,'nominal');q=np.linspace(*s.q_range,101)
            np.testing.assert_allclose(s.adc(q),r.make_sensor(kind,0,r.SEED).adc(q),rtol=0,atol=1e-14)
            np.testing.assert_allclose(r.oracle_inverse(s,s.adc(q)),q,rtol=0,atol=1e-10)
        for setting in r.SETTINGS:
            s=r.sensor_for('ntc_frontend',0,r.SEED,setting);q=np.linspace(0,100,1001)
            self.assertTrue(np.all(np.diff(s.adc(q))<0))
            np.testing.assert_allclose(r.oracle_inverse(s,s.adc(q)),q,atol=1e-10)

    def test_cv_never_uses_test_labels(self):
        case=r.calibration_case(r.make_sensor('ntc',0,r.SEED),30,0,r.SEED)
        altered=copy.deepcopy(case);altered['y_test'][:]=123
        for family in ('lut','cubic','poly'):
            a=r.selected_cv(case,family);b=r.selected_cv(altered,family)
            np.testing.assert_array_equal(a.coefficients,b.coefficients)
            self.assertEqual(a.hyper,b.hyper)

    def test_all_new_updates_are_prefix_causal(self):
        case=r.calibration_case(r.make_sensor('ntc_frontend',0,r.SEED),30,0,r.SEED)
        initial=r.initial_models(case);stream=r.generate_stream(case,0,30,'step',r.SEED)
        altered=copy.deepcopy(stream);altered['x'][128:]*=-1;altered['y'][128:]+=1
        for family in ('lut','cubic'):
            for suffix,param in (('decayed',128),('two_point',None),('batch_all',None),('batch_window',None)):
                method=family+'_'+suffix
                a=r.custom_trajectory(case,initial,stream,method,param)
                b=r.custom_trajectory(case,initial,altered,method,param)
                for sa,sb in zip(a['snapshots'],b['snapshots']):
                    if sa['t']<=128:self.assertEqual(sa,sb,method)
                self.assertEqual(a['completed_steps'],384)

    def test_affine_uses_completed_block_endpoints(self):
        case=r.calibration_case(r.make_sensor('ntc',0,r.SEED),30,0,r.SEED)
        initial=r.initial_models(case);stream=r.generate_stream(case,0,30,'none',r.SEED)
        model=initial['lut'];p=r.design('lut',stream['x'][:32],32)@model.coefficients
        i,j=np.argmin(p),np.argmax(p)
        gain=np.float32((stream['y'][j]-stream['y'][i])/(p[j]-p[i]))
        offset=np.float32(stream['y'][i]-float(gain)*p[i])
        out=r.custom_trajectory(case,initial,stream,'lut_two_point')
        self.assertEqual(out['snapshots'][1]['gain'],float(gain))
        self.assertEqual(out['snapshots'][1]['offset'],float(offset))
        self.assertEqual(out['snapshots'][0]['before'],out['snapshots'][0]['after'])


if __name__=='__main__':unittest.main()
