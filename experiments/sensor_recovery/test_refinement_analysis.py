"""Independent least-squares and causal checks of the 32-pair correction."""
import copy,unittest
import numpy as np
import refinement_analysis as f

class AffineCorrection(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  r=f.r
  cls.case=r.calibration_case(r.make_sensor('ntc_frontend',0,r.SEED),75,0,r.SEED)
  cls.initial=r.initial_models(cls.case)
  cls.stream=r.generate_stream(cls.case,0,75,'ramp',r.SEED)

 def test_block_solution_matches_independent_lstsq(self):
  for fam in ('lut','cubic'):
   c=self.initial[fam].coefficients
   pred=f.r.design(fam,self.stream['x'],len(c))@c
   out=f.affine_trajectory(self.case,self.initial,self.stream,fam,'float64')
   for snap in out['snapshots']:
    t=snap['t'];end=t//32*32
    expected=np.array([1,0]) if not end else np.linalg.lstsq(np.column_stack([pred[end-32:end],np.ones(32)]),self.stream['y'][end-32:end],rcond=None)[0].astype(np.float32)
    np.testing.assert_allclose([snap['gain'],snap['offset']],expected,rtol=1e-6,atol=1e-7)

 def test_future_labels_do_not_change_past_and_partial_blocks_hold(self):
  changed=copy.deepcopy(self.stream);changed['y'][161:]+=10
  a=f.affine_trajectory(self.case,self.initial,self.stream,'cubic')
  b=f.affine_trajectory(self.case,self.initial,changed,'cubic')
  self.assertEqual([s for s in a['snapshots'] if s['t']<=160],[s for s in b['snapshots'] if s['t']<=160])
  ss={s['t']:s for s in a['snapshots']}
  for t in (129,132,144):
   self.assertEqual((ss[t]['gain'],ss[t]['offset']),(ss[128]['gain'],ss[128]['offset']))

 def test_degenerate_blocks_hold_identity(self):
  stream=copy.deepcopy(self.stream);stream['x'][:]=stream['x'][0]
  out=f.affine_trajectory(self.case,self.initial,stream,'cubic','float64')
  self.assertEqual(out['degenerate_blocks'],12)
  self.assertTrue(all((s['gain'],s['offset'])==(1,0) for s in out['snapshots']))

if __name__=='__main__':unittest.main()
