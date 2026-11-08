"""Reparse previously acquired optimized-suite MCU logs; acquire no new timings."""
from pathlib import Path
import hashlib,json,statistics
from mcu.capture import parse_lines
HERE=Path(__file__).resolve().parent

def main():
 out=HERE/'results/revision_20261108';boards={};hashes={}
 for board,prefix in [('mega2560','mega'),('esp32c3','c3')]:
  folder=HERE/'results'/f'{prefix}_optimized_20261108';runs=[]
  for i in [1,2]:
   p=folder/'captures'/f'{board}_capture{i:02d}.json';raw=p.with_suffix('.log');side=p.with_suffix('.session.json')
   r=parse_lines(raw.read_text(encoding='utf-8-sig').splitlines());stored=json.loads(p.read_text());s=json.loads(side.read_text())
   if r!=stored:raise ValueError('Raw/JSON mismatch '+str(p))
   if s['raw_sha256']!=hashlib.sha256(raw.read_bytes()).hexdigest() or s['result_sha256']!=hashlib.sha256(p.read_bytes()).hexdigest():raise ValueError('Sidecar hash mismatch')
   if s['status']!='validated' or r['metadata']['BOARD']!=board:raise ValueError('Capture not validated')
   runs.append(r)
   for f in [p,raw,side]:hashes[str(f.relative_to(HERE))]=hashlib.sha256(f.read_bytes()).hexdigest()
  methods={}
  for name in runs[0]['summary']:
   times=[x['summary'][name]['gross_us_per_call'] for x in runs]
   methods[name]=dict(payload_bytes=runs[0]['payload_bytes'][name],median_of_session_medians_us=statistics.median(times),session_medians_us=times,metrics=[r['accuracy'][name] for r in runs])
  boards[board]=dict(methods=methods,sessions=2,batches_per_session=10,calls_per_batch=2048)
 result=dict(boards=boards,sha256=hashes,seed=20261108,scenario='ntc_frontend',unit=0,repeat=0,budget=75,
  scope='Reanalysis of existing physical-MCU captures; synthetic accuracy fixture. Distinct from matched seed 2026110801. No new device measurement.',
  limit='Legacy archives record firmware hashes but do not include firmware images; no device attestation. Source/serial/sidecar audits are preserved.')
 (out/'recovered_hardware.json').write_text(json.dumps(result,indent=2)+'\n')
 for b,d in boards.items():
  print(b,[(m,round(x['median_of_session_medians_us'],4),x['metrics'][0]['mae_C']) for m,x in d['methods'].items() if m in ['physics','polynomial','uniform32_float32','cubic16_unrolled','cubic16_horner']])
if __name__=='__main__':main()
