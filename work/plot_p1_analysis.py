from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
ROOT=Path.cwd();batch='p1_cosine_20261017';out=ROOT/'outputs/launches'/batch/'analysis';out.mkdir(exist_ok=True)
summary=json.loads((ROOT/'outputs/launches'/batch/'pilot_summary.json').read_text())
colors=['#176B98','#DE812E','#A8436A'];names=['Direct','Iterative (5 passes)','Diffusion (20 DDIM steps)']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.facecolor':'white'})
fig,axes=plt.subplots(1,3,figsize=(15,4.4),layout='constrained')
classes=['Background','Monolayer','Few-layer','Multilayer'];keys=['background','monolayer','fewlayer','multilayer']
train_times=[];val_times=[]
for i,item in enumerate(summary['results']):
 run=Path(item['run']);rows=list(csv.DictReader((run/'metrics_val.csv').open()))
 vals=[r for r in rows if r['tag']=='val/miou_4class']
 axes[0].plot([int(r['epoch']) for r in vals],[float(r['value']) for r in vals],color=colors[i],marker='o',linewidth=2,label=names[i])
 m=item['best_val']['metrics'];axes[1].bar(np.arange(4)+(i-1)*.24,[m['iou_'+k] for k in keys],width=.23,color=colors[i],label=names[i])
 erows=list(csv.DictReader((run/'metrics_epochs.csv').open()))
 train_times.append(sum(float(r['value']) for r in erows if r['tag']=='resources/train_seconds')/60)
 val_times.append(sum(float(r['value']) for r in rows if r['tag']=='val/seconds')/60)
axes[0].set(xlabel='Epoch',ylabel='Four-class mIoU',title='Full validation curves (25 images)',xticks=[10,20,30,40],ylim=(0.45,.9));axes[0].yaxis.set_major_formatter(PercentFormatter(1));axes[0].grid(alpha=.2);axes[0].legend(fontsize=8,loc='lower right')
axes[1].set(xticks=np.arange(4),xticklabels=classes,ylabel='IoU',ylim=(0,1.08),title='Per-class IoU at selected best epoch');axes[1].tick_params(axis='x',labelrotation=15);axes[1].yaxis.set_major_formatter(PercentFormatter(1));axes[1].grid(axis='y',alpha=.2)
x=np.arange(3);axes[2].bar(x,train_times,color='#467B8E',label='Training');axes[2].bar(x,val_times,bottom=train_times,color='#D9B67D',label='Validation');axes[2].set(xticks=x,xticklabels=['Direct','Iterative','Diffusion'],ylabel='Minutes',title='Measured time across 40 epochs');axes[2].legend(fontsize=8);axes[2].grid(axis='y',alpha=.2)
for j,(a,b) in enumerate(zip(train_times,val_times)):axes[2].text(j,a+b+.8,f'{a+b:.1f}',ha='center')
fig.suptitle('P1 pilot | seed 42 | 512 x 512 | batch 2 | train / validation only',fontsize=13)
fig.savefig(out/'p1_overview.png',dpi=180);fig.savefig(out/'p1_overview.pdf')
print('PLOT_SAVED',out,flush=True)
