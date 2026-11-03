"""Export descriptive quality/cost plots from the saved audit (no inference)."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

out=Path(__file__).resolve().parents[1]/'outputs/analysis/story_audit_20261025'
d=json.loads((out/'audit.json').read_text())
metrics=d['metrics']['test']
bench={r['name']:r for r in d['benchmark_current']['objects']}
names=['E0','E1','E3','E3R','segformer_b1','mask2former','ddp','repela']
labels={'E0':'E0','E1':'E1','E3':'E3','E3R':'E3R','segformer_b1':'SegFormer-B1',
        'mask2former':'Mask2Former','ddp':'DDP','repela':'RepELA-Small'}
colors={'E0':'#556b82','E1':'#6d92a0','E3':'#aaa9ab','E3R':'#c54a32','segformer_b1':'#286b97',
        'mask2former':'#74894a','ddp':'#8c649e','repela':'#ba9336'}
fig,axes=plt.subplots(1,2,figsize=(12.3,4.8),layout='constrained')
for n in names:
    m=metrics[n]
    axes[0].scatter(m['miou'],m['bf'],s=75,c=colors[n],zorder=3)
    offset={'E0':(5,-13),'E1':(5,2),'E3R':(-3,8),'segformer_b1':(5,-12),
            'mask2former':(-68,-12),'ddp':(-27,-16)}.get(n,(6,5))
    axes[0].annotate(labels[n],(m['miou'],m['bf']),xytext=offset,textcoords='offset points',fontsize=9)
    if n=='E3':continue
    b=bench['repela_deploy' if n=='repela' else n]
    axes[1].scatter(b['image_predict_s_median'],m['bf'],s=50+b['params']/1e6*2,c=colors[n],zorder=3)
    offset_right=(5,-15) if n=='mask2former' else (5,5)
    axes[1].annotate(labels[n],(b['image_predict_s_median'],m['bf']),xytext=offset_right,textcoords='offset points',fontsize=9)
axes[0].set(xlabel='Test global mIoU (%)',ylabel='Test foreground boundary F1 (%)',
            title='Region accuracy and boundary quality',xlim=(89.7,95.5),ylim=(45,78))
axes[1].set(xlabel='Full-image prediction median (s)',ylabel='Test foreground boundary F1 (%)',
            title='Cost: latest unified re-timing',xlim=(.9,5.9),ylim=(45,78))
for ax in axes:
    ax.grid(alpha=.2)
    ax.spines[['top','right']].set_visible(False)
fig.suptitle('Boundary quality and inference cost under the fixed split and seed 42',fontsize=12)
fig.savefig(out/'quality_cost_audit.png',dpi=180)
fig.savefig(out/'quality_cost_audit.pdf')
plt.close(fig)

fig,ax=plt.subplots(figsize=(9,4.2),layout='constrained')
comparators=['segformer_b1','E0','E1','E3']
for j,n in enumerate(comparators):
    r=d['paired']['test'][n];v=np.array([x['bf_delta'] for x in r['images']])
    ax.scatter(v,np.full(len(v),j)+np.linspace(-.18,.18,len(v)),s=25,color=colors[n],alpha=.75)
    ax.scatter(np.median(v),j,s=110,marker='|',c='black',linewidths=2,zorder=4)
    ax.text(46,j,f"{r['bf']['wins']}/27 improve",va='center',fontsize=9)
ax.axvline(0,color='gray',lw=1);ax.set(yticks=range(4),yticklabels=[labels[n] for n in comparators],
    xlabel='Per-image foreground boundary F1: E3R minus comparator (pp)',
    title='Saved test predictions: paired boundary differences',xlim=(-13,59))
ax.grid(axis='x',alpha=.2);ax.spines[['top','right']].set_visible(False)
fig.savefig(out/'boundary_paired_audit.png',dpi=180)
fig.savefig(out/'boundary_paired_audit.pdf')
