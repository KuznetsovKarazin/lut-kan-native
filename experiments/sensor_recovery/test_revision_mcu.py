import ctypes,json,subprocess,tempfile,unittest
from pathlib import Path
import numpy as np
from mcu.capture_references import parse_lines
HERE=Path(__file__).resolve().parent
class RevisionKernels(unittest.TestCase):
 def test_capture_rejects_incomplete_or_tampered_records(self):
  # Artificial protocol records for parser tests only: never hardware evidence.
  manifest=json.loads((HERE/'mcu/revision_manifest.json').read_text());token='a'*32
  lines=['READY,SENSOR_REFERENCES_V1,esp32s3','ECHO,'+token,'ACCEPTED,'+token,
   'SENSOR_BENCH_V1,synthetic_ntc_frontend,seed2026110801,unit0,B75',
   'SUITE,matched_references_v1','SESSION,'+token,'BOARD,esp32s3','CPU_HZ,240000000',
   'FIXTURE_SHA256,'+manifest['fixture_sha256'],'PRECISION,float32']
  for name,size in manifest['payload_bytes'].items():
   lines += [f'PAYLOAD,{name},{size}',f'ACCURACY,{name},0.1,0.2,0.3,0,PASS']
  for name in ['overhead',*manifest['payload_bytes']]:
   lines += [f'TIME,{name},{i},2048,1000,10' for i in range(10)]
  lines+=['END_SENSOR_BENCH'];self.assertEqual(len(parse_lines(lines)['accuracy']),6)
  variants=[lines[:-1],lines[:15]+lines[16:],lines+[lines[-1]],lines[:4]+['CPU_HZ,1']+lines[4:]]
  variants += [[line.replace('CPU_HZ,240000000','CPU_HZ,160000000') for line in lines],
               [line.replace(',0,PASS',',0.003,PASS') for line in lines],
               [line.replace('ECHO,'+token,'ECHO,'+'b'*32) for line in lines]]
  for altered in variants:
   with self.assertRaises(ValueError):parse_lines(altered)
 def test_float32_parity_and_guard(self):
  with tempfile.TemporaryDirectory() as out:
   libfile=Path(out)/'revision.so'
   subprocess.run(['g++','-std=c++11','-O2','-Wall','-Wextra','-Werror','-fno-fast-math','-ffp-contract=off','-shared','-fPIC','-I',str(HERE/'mcu/include'),str(HERE/'mcu/revision_host_bridge.cpp'),'-o',str(libfile)],check=True)
   lib=ctypes.CDLL(str(libfile));lib.revision_predict.argtypes=[ctypes.c_int,ctypes.c_float];lib.revision_predict.restype=ctypes.c_float
   d=json.loads((HERE/'mcu/revision_reference.json').read_text())
   for method in range(2):
    pred=np.array([lib.revision_predict(method,a) for a in d['adc']]);gap=np.max(abs(pred-d['expected'][method]))
    self.assertTrue(np.isfinite(pred).all());self.assertLess(gap,.002);print('reference',method,'max_gap_C',gap)
   lib.revision_project.argtypes=[ctypes.POINTER(ctypes.c_float)];lib.revision_project.restype=ctypes.c_int
   c=np.linspace(1,-1,32,dtype=np.float32);before=c.copy();ptr=c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
   self.assertEqual(lib.revision_project(ptr),0);np.testing.assert_array_equal(c,before)
   c[5]=c[4]+1;self.assertEqual(lib.revision_project(ptr),1);self.assertTrue(np.all(np.diff(c)<=0))
if __name__=='__main__':unittest.main()
