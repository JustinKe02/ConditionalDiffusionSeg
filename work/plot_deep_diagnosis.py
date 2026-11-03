from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
out=Path('outputs/launches/p1_cosine_20261017/analysis');r=json.loads((out/'deep_diffusion_diagnosis.json').read_text())
plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(1,2,figsize=(12.5,4.6),layout='constrained')
steps=np.arange(1,21)
ax[0].plot(steps,[s['teacher_forced']['miou_4class'] for s in r['trajectory']],color='#176B98',marker='o',markersize=3,label='True-mask-derived state (diagnostic)')
ax[0].plot(steps,[s['rollout']['miou_4class'] for s in r['trajectory']],color='#B0485F',marker='o',markersize=3,label='Model-generated state (inference)')
ax[0].set(xlabel='Denoising iteration (t=999 to t=0)',ylabel='Four-class mIoU',title='Same model, images and initial Gaussian noise',ylim=(.65,1.03),xticks=[1,5,10,15,20]);ax[0].yaxis.set_major_formatter(PercentFormatter(1));ax[0].legend(fontsize=8,loc='lower right');ax[0].grid(alpha=.2)
colors=['#176B98','#D39345','#B0485F'];modes=['matched_rgb','zero_rgb','permuted_rgb'];labels=['Correct RGB','RGB replaced by zeros','RGB from another patch']
for i,(mode,label,color) in enumerate(zip(modes,labels,colors)):
 vals=[next(s['metrics']['miou_4class'] for s in r['rgb_ablations'] if s['t']==t and s['condition']==mode) for t in [0,500,950,999]]
 ax[1].bar(np.arange(4)+(i-1)*.25,vals,width=.24,color=color,label=label)
ax[1].set(xticks=np.arange(4),xticklabels=['t=0','t=500','t=950','t=999'],ylabel='Four-class mIoU',title='Sensitivity to RGB when the mask is GT-derived',ylim=(0,1.08));ax[1].yaxis.set_major_formatter(PercentFormatter(1));ax[1].legend(fontsize=8,loc='lower left');ax[1].grid(axis='y',alpha=.2)
fig.suptitle('P1 diffusion diagnosis | fixed 12 training crops at 512 x 512 | no weight updates',fontsize=12)
fig.savefig(out/'diffusion_failure_diagnosis.png',dpi=180,facecolor='white');fig.savefig(out/'diffusion_failure_diagnosis.pdf',facecolor='white')
print('Saved diagnostic figure.')
