"""Revision figures/tables from preserved numerical records, never inferred timings."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import FancyBboxPatch
from PIL import Image
from make_assets import save,FIG,TABLES,DATA
HERE=Path(__file__).resolve().parent
NAMES={'ntc':'NTC','ntc_frontend':'Distorted NTC','ldr':'LDR'}
COLORS={'lut_cv':'#0072B2','cubic_cv':'#009E73','poly_cv':'#AA4499','physics':'#A45E1A','lut_nlms':'#0072B2','cubic_nlms':'#009E73','cubic_decayed':'#AA4499','cubic_batch_all':'#777777','cubic_batch_window':'#D55E00','cubic_two_point':'#C28E13','frozen_cubic':'#444444'}
LABELS={'lut_cv':'LUT f32','cubic_cv':'Compact cubic','poly_cv':'Polynomial','physics':'Physical model','lut_nlms':'LUT NLMS','cubic_nlms':'Cubic NLMS','cubic_decayed':'Cubic decaying NLMS','cubic_batch_all':'Cubic batch: all','cubic_batch_window':'Cubic batch: recent 64','cubic_two_point':'Cubic two-point','frozen_cubic':'Frozen cubic'}

def generate():
 files=['revision_summary.json','revision_curves.json','recovered_hardware.json','revision_audit.json','revision_selection.json']
 manifest={name:hashlib.sha256((DATA/name).read_bytes()).hexdigest() for name in files}
 p=DATA/'revision_source_hashes.json'
 if p.exists():assert json.loads(p.read_text())==manifest,'Revision input changed'
 p.write_text(json.dumps(manifest,indent=2)+'\n')
 summary=json.loads((DATA/'revision_summary.json').read_text());curves=json.loads((DATA/'revision_curves.json').read_text());pilot=json.loads((DATA/'recovered_hardware.json').read_text());hw=json.loads((DATA/'comparison.json').read_text())
 def get(stage,sc,B,m,cap=0,setting='nominal',drift='none'):
  return next(r for r in summary if (r['stage'],r['scenario'],r['budget'],r['method'],r['cap_bytes'],r['setting'],r['drift'])==(stage,sc,B,m,cap,setting,drift))
 fig,ax=plt.subplots(figsize=(7.05,1.3));ax.axis('off')
 cards=[('CONTROLLED NUMERICS','NTC / distorted NTC / LDR\nOriginal: 3312 static fits\nRevision: 3488 static fits','#EAF3F9'),('REFERENCE-ASSISTED UPDATES','Original: 3888 trajectories\nRevision: 5472 trajectories\nSeparate development seeds','#EEF5ED'),('REAL MCU EXECUTION','Mega2560 + ESP32-C3\nTwo numerical fixtures\nTwo sessions / board / fixture','#FFF1E4')]
 for j,(title,body,color) in enumerate(cards):
  x=.01+j*.335;ax.add_patch(FancyBboxPatch((x,.08),.31,.85,boxstyle='round,pad=.005',fc=color,ec='#CDD5DC',lw=.7));ax.text(x+.155,.77,title,ha='center',fontsize=8,weight='bold');ax.text(x+.155,.41,body,ha='center',va='center',fontsize=8,linespacing=1.5)
 fig.subplots_adjust(0,0,1,1);save(fig,'protocol')
 # B=30 exposes selection variance without mixing scenario units.
 fig,axes=plt.subplots(1,3,figsize=(7.05,2.25))
 for ax,sc in zip(axes,NAMES):
  for fam in ['lut','cubic']:
   for sel,style in [('holdout','--'),('cv','-')]:
    vals=[get('static',sc,30,fam+'_'+sel,c)['noisy_mae']['mean'] for c in [80,144,224]]
    ax.plot([80,144,224],vals,style,marker='o',ms=3,lw=1.2,color=COLORS[fam+'_cv'],label=LABELS[fam+'_cv']+(' / CV' if sel=='cv' else ' / holdout'))
  ax.set_title(NAMES[sc]+', B = 30',loc='left');ax.set_xticks([80,144,224]);ax.set_xlabel('Payload (bytes)');ax.set_ylabel('MAE (log₁₀ lux)' if sc=='ldr' else 'MAE (°C)');ax.grid(alpha=.18)
 fig.legend(*axes[0].get_legend_handles_labels(),ncol=4,loc='upper center',bbox_to_anchor=(.5,1.035),frameon=False)
 fig.subplots_adjust(left=.07,right=.99,bottom=.22,top=.76,wspace=.42);save(fig,'cv_capacity')
 settings=['noise_low','nominal','noise_high','tolerance_half','tolerance_double','amplitude_low','amplitude_high','frequency_low','frequency_high'];labels=['Noise 0.25 LSB','Nominal','Noise 4 LSB','Tolerances ×0.5','Tolerances ×2','Distortion 1 LSB','Distortion 12 LSB','Frequency 1 cycle','Frequency 9 cycles'];methods=['lut_cv','cubic_cv','poly_cv','physics']
 a=np.array([[get('static','ntc_frontend',75,m,144 if m in methods[:2] else 0,s)['noisy_mae']['mean'] for m in methods] for s in settings])
 fig,ax=plt.subplots(figsize=(3.4,2.65));im=ax.imshow(a,aspect='auto',cmap='YlOrRd',norm=LogNorm(.02,1));ax.set_xticks(range(4),['LUT','Cubic','Poly.','Physics']);ax.set_yticks(range(9),labels);ax.tick_params(length=0);ax.set_title('Distorted NTC, B = 75: MAE (°C)',fontsize=9,pad=8)
 for i in range(9):
  for j in range(4):ax.text(j,i,f'{a[i,j]:.3g}',ha='center',va='center',fontsize=7,color='white' if a[i,j]>.25 else '#17222B')
 fig.subplots_adjust(left=.34,right=.97,top=.88,bottom=.1);save(fig,'sensitivity')
 fig,axes=plt.subplots(1,3,figsize=(7.05,2.7),sharey=True)
 methods=['frozen_cubic','cubic_nlms','cubic_decayed','cubic_two_point','cubic_batch_all','cubic_batch_window']
 for ax,drift in zip(axes,['none','step','ramp']):
  for m in methods:
   rr=sorted([r for r in curves if r['scenario']=='ntc_frontend' and r['budget']==75 and r['drift']==drift and r['method']==m],key=lambda r:r['t'])
   ax.plot([r['t'] for r in rr],[r['mean_mae'] for r in rr],label=LABELS[m],color=COLORS[m],lw=1.2,ls='--' if m=='frozen_cubic' else '-')
  ax.set_yscale('log');ax.set_ylim(.035,.8);ax.set_xticks([0,128,256,384]);ax.set_xlabel('Labelled references');ax.set_title(drift.capitalize(),loc='left');ax.grid(alpha=.18)
  if drift!='none':ax.axvline(128,ls=':',color='#999',lw=.8)
 axes[0].set_ylabel('Mean grid MAE (°C; log scale)');fig.legend(*axes[0].get_legend_handles_labels(),ncol=3,loc='upper center',bbox_to_anchor=(.5,1.02),frameon=False)
 fig.subplots_adjust(left=.07,right=.99,top=.72,bottom=.18,wspace=.15);save(fig,'revision_adaptation')
 # Fully separate pilot and matched rows: times/errors never mixed across fixtures.
 fig,axes=plt.subplots(2,2,figsize=(7.05,4.6))
 pms=['physics','polynomial','uniform32_float32','uniform32_int16','uniform32_uint8','nonuniform16_int16','cubic16_unrolled','cubic16_horner']
 pls=['Physics 32 B','Poly. 56 B','LUT f32 144 B','LUT i16 88 B','LUT u8 56 B','Nonuniform 120 B','Cubic 80 B','Horner 224 B']
 colors=['#A45E1A','#AA4499','#0072B2','#56B4E9','#777777','#B6A246','#009E73','#D55E00']
 offsets=[(4,4),(2,-17),(-5,10),(4,12),(4,3),(4,1),(-4,-4),(-5,-17)]
 for j,board in enumerate(['mega2560','esp32c3']):
  ax=axes[0,j]
  for i,(m,l,c) in enumerate(zip(pms,pls,colors)):
   r=pilot['boards'][board]['methods'][m];x=r['median_of_session_medians_us'];y=r['metrics'][0]['mae_C'];ax.scatter(x,y,s=30,color=c,zorder=4,label=l,marker=['o','s','D','^','v','P','X','*'][i])

  ax.set_yscale('log');ax.set_ylim(.032,.4);ax.set_xlim(0,410 if j==0 else 41);ax.set_title(('(a) Mega2560' if j==0 else '(b) ESP32-C3')+' / pilot',loc='left');ax.grid(alpha=.15);ax.set_xlabel('Gross inference time (µs/call)')
  ax=axes[1,j]
  for m,l,c in [('lut32_float32','LUT32','#0072B2'),('lut60_int16','LUT60 i16','#56B4E9'),('cubic32_unrolled','Cubic32','#009E73'),('horner8','Horner8','#D55E00')]:
   r=hw['boards'][board]['static'][m];x=r['median_of_session_medians_us'];y=r['metrics'][0]['mae_C'];ax.scatter(x,y,s=25,color=c,zorder=4);ax.annotate(l,(x,y),xytext=(-5,8) if m=='cubic32_unrolled' else (4,7),textcoords='offset points',fontsize=7,ha='right' if m=='cubic32_unrolled' else 'left')
  ax.set_xlim(0,430 if j==0 else 34);ax.set_ylim(.041,.061);ax.set_title(('(c) Mega2560' if j==0 else '(d) ESP32-C3')+' / matched 144 B',loc='left');ax.grid(alpha=.15);ax.set_xlabel('Gross inference time (µs/call)')
 for ax in axes[:,0]:ax.set_ylabel('Fixed-case MAE (°C)')
 fig.legend(*axes[0,0].get_legend_handles_labels(),ncol=4,loc='upper center',bbox_to_anchor=(.5,1.01),frameon=False,fontsize=7)
 fig.subplots_adjust(left=.09,right=.99,bottom=.10,top=.83,hspace=.60,wspace=.24);save(fig,'hardware_complete')
 # Exact mean/max metrics; three significant figures, raw precision retained in JSON.
 rows=[]
 for sc in NAMES:
  for m in ['lut_cv','cubic_cv','poly_cv','physics']:
   r=get('static',sc,75,m,144 if m.endswith('_cv') and m!='poly_cv' else 0)
   rows.append(f"{NAMES[sc]} & {LABELS[m]} & {r['ideal_mae']['mean']:.3g} & {r['noisy_mae']['mean']:.3g} & {r['max_error']['maximum_observed']:.3g}"+r' \\')
 (TABLES/'revision_static.tex').write_text('\n'.join(rows)+'\n')
 rows=[]
 for sc in NAMES:
  for B in [30,75]:
   for cap in [80,144,224]:
    vals=[get('static',sc,B,m,cap)['noisy_mae']['mean'] for m in ['lut_holdout','lut_cv','cubic_holdout','cubic_cv']]
    rows.append(f'{NAMES[sc]} & {B} & {cap} & '+' & '.join(f'{v:.3g}' for v in vals)+r' \\')
 (TABLES/'cv_all.tex').write_text('\n'.join(rows)+'\n')
 rows=[]
 for sc in NAMES:
  for B in [30,75]:
   for drift in ['none','step','ramp']:
    vals=[get('confirmation',sc,B,m,drift=drift)['final_mae']['mean'] for m in ['frozen_cubic','cubic_nlms','cubic_decayed','cubic_two_point','cubic_batch_all','cubic_batch_window','cubic_rls_extended','poly_rls_extended']]
    rows.append(f'{NAMES[sc]} & {B} & {drift} & '+' & '.join(f'{v:.3g}' for v in vals)+r' \\')
 (TABLES/'revision_online.tex').write_text('\n'.join(rows)+'\n')
 rows=[]
 for m,l in zip(pms,pls):
  a=pilot['boards']['mega2560']['methods'][m];b=pilot['boards']['esp32c3']['methods'][m]
  rows.append(f"{l} & {a['median_of_session_medians_us']:.3g} & {b['median_of_session_medians_us']:.3g} & {b['metrics'][0]['mae_C']:.3g} & {b['metrics'][0]['max_C']:.3g}"+r' \\')
 (TABLES/'recovered_hardware.tex').write_text('\n'.join(rows)+'\n')
 # Separate IEEE graphical abstract; exact pixels, >=16px labels, 300dpi metadata.
 fig=plt.figure(figsize=(6.6,2.95),dpi=100);ax=fig.add_axes([0,0,1,1]);ax.axis('off')
 ax.text(.04,.9,'Sensor calibration: three resource budgets',fontsize=16,weight='bold',color='#17334A')
 for x,title,body,col in [(.04,'ACCURACY','Compact cubic\n& fitted physics','#E8F4EF'),(.365,'STORAGE','Polynomial\n24–56 bytes','#F1EAF6'),(.69,'UPDATE TIME','LUT NLMS\n49–53% lower*','#EAF3F9')]:
  ax.add_patch(FancyBboxPatch((x,.27),.27,.5,boxstyle='round,pad=.01',fc=col,ec='#CBD5DF'));ax.text(x+.135,.65,title,ha='center',fontsize=12,weight='bold');ax.text(x+.135,.43,body,ha='center',va='center',fontsize=12,linespacing=1.4)
 ax.text(.04,.14,'Synthetic accuracy  |  Real Mega2560 / ESP32-C3 timing',fontsize=12)
 ax.text(.04,.04,'*Versus cubic NLMS; higher residual error',fontsize=12)
 fig.savefig(FIG/'gagraphic.png',dpi=100);fig.savefig(FIG/'gagraphic.pdf',metadata={'CreationDate':None,'ModDate':None});fig.savefig(FIG/'gagraphic.svg',metadata={'Date':None});plt.close(fig)
 p=FIG/'gagraphic.png'
 with Image.open(p) as im:im.save(p,dpi=(300,300),optimize=True)
 print('Revision assets regenerated; numerical figures use saved results only.')
if __name__=='__main__':generate()
