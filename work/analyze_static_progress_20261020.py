"""Read-only progress snapshot and Chinese scientific figure, no model execution."""
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'docs/thesis/chapter3_20261019/scripts'))
from font_config import configure_fonts, apply_complete_frame
FONT = configure_fonts()
BATCH = ROOT/'outputs/launches/p1_static_semantic_20261020'
NOW = datetime.now(timezone.utc)
OUT = ROOT/'outputs/analysis'/('static_semantic_progress_'+NOW.strftime('%Y%m%dT%H%M%SZ'))
OUT.mkdir(parents=True,exist_ok=False)
INPUTS = {}


def read_bytes(path):
    payload = Path(path).read_bytes()
    INPUTS[str(Path(path).relative_to(ROOT))] = hashlib.sha256(payload).hexdigest()
    return payload


def read(path):
    return json.loads(read_bytes(path))


def rows(path):
    return list(csv.DictReader(read_bytes(path).decode().splitlines()))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    plan = read(BATCH/'plan.json')
    legacy = read(BATCH/'source_before.json')['legacy_categories']
    legacy_count = 0
    for values in legacy.values():
        for path,digest in values.items():
            assert sha(ROOT/path) == digest, path
            legacy_count += 1
    for kind in ['source_sha256','workflow_sha256']:
        for path,digest in plan[kind].items():
            assert sha(ROOT/path) == digest, path
    report = dict(captured_at_utc=NOW.isoformat(),batch_status=read(BATCH/'status.json'),
                  purpose='Partial progress snapshot; no test, model execution, optimization or frozen artifact writes.',
                  legacy_hash_entries_verified=legacy_count, modes={})
    table = []
    for mode in ['S0','S1','S2']:
        run = Path(plan['runs'][mode]); status = read(run/'status.json')
        best = read(run/'best_val_metrics.json')
        epochs, validation = {}, {}
        for category, mapping in [('epochs',epochs),('val',validation)]:
            for row in rows(run/f'metrics_{category}.csv'):
                if row.get('value') in [None,'']:
                    continue
                mapping.setdefault(int(row['epoch']),{})[row['tag']]=float(row['value'])
        validation = {e:d for e,d in validation.items() if d.get('val/image_count') == 25 and 'val/seconds' in d}
        step_rows = rows(run/'metrics_steps.csv')
        grad = [float(x['value']) for x in step_rows if x['tag']=='train_step/grad_norm_pre_clip' and x.get('value')]
        assert all(math.isfinite(float(x['value'])) for x in step_rows if x.get('value'))
        with np.load(run/'best_val_confusion.npz') as data:
            matrix = data['global_matrix'].copy(); boundaries = data['boundary_counts'].copy()
            assert len(data['ids']) == 25 and np.array_equal(data['per_image'].sum(0),matrix)
        # Recompute all summary metrics using the existing CPU metric definitions.
        sys.path.insert(0,str(ROOT/'src'))
        from conditional_diffusion.metrics_impl import summarize,summarize_boundaries
        recomputed = dict(**summarize(matrix),**summarize_boundaries(boundaries))
        assert all(abs(v-best['metrics'][k]) < 1e-12 for k,v in recomputed.items())
        normalized = np.divide(matrix,matrix.sum(1,keepdims=True),out=np.zeros_like(matrix,dtype=float),where=matrix.sum(1,keepdims=True)>0)
        statistics = dict(last_logged_training_epoch=max(epochs),last_logged_training_step=int(step_rows[-1]['global_step']),
                          train_seconds_total=sum(d['resources/train_seconds'] for d in epochs.values()),
                          train_seconds_median=float(np.median([d['resources/train_seconds'] for d in epochs.values()])),
                          logged_grad_max=max(grad),logged_grad_median=float(np.median(grad)),
                          logged_grad_clipped_fraction=float(np.mean(np.array(grad)>1)),
                          confusion_audit_passed=True,monolayer_recall=float(normalized[1,1]),
                          monolayer_to_background=float(normalized[1,0]),fewlayer_to_monolayer=float(normalized[2,1]))
        report['modes'][mode]=dict(run=str(run),status=status,best=best,epochs=epochs,validation=validation,statistics=statistics)
        m=best['metrics'];table.append(dict(mode=mode,best_epoch=best['epoch'],miou_percent=100*m['miou_4class'],
            mono_iou_percent=100*m['iou_monolayer'],few_iou_percent=100*m['iou_fewlayer'],
            foreground_boundary_f1_percent=100*m['boundary_f1_foreground'],validation_seconds=m['seconds']))
    for mode in ['S1','S2']:
        m=report['modes'][mode]['best']['metrics'];base=report['modes']['S0']['best']['metrics']
        report['modes'][mode]['delta_vs_S0_pp']={k:100*(m[k]-base[k]) for k in
            ['miou_4class','iou_monolayer','iou_fewlayer','boundary_f1_foreground']}
    report['screening']=plan['screening']
    report['font_manifest']=FONT
    with (OUT/'best_snapshot.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=table[0].keys());writer.writeheader();writer.writerows(table)
    plt.rcParams.update({'font.size':10,'axes.spines.top':True,'axes.spines.right':True})
    colors={'S0':'#536d9c','S1':'#b87c52','S2':'#568778'}
    labels={'S0':'直接分割（S0）','S1':'确定性五步（S1）','S2':'条件扩散五步（S2）'}
    fig,axes=plt.subplots(1,3,figsize=(13.2,3.65))
    specifications=[('val/miou_4class','四类平均交并比（%）'),
                    ('thin','单层与少层平均交并比（%）'),
                    ('val/boundary_f1_foreground','前景边界F1（%）')]
    for ax,(key,ylabel) in zip(axes,specifications):
        for mode in ['S0','S1','S2']:
            values=report['modes'][mode]['validation'];xs=sorted(values)
            ys=[100*((values[x]['val/iou_monolayer']+values[x]['val/iou_fewlayer'])/2 if key=='thin' else values[x][key]) for x in xs]
            ax.plot(xs,ys,'o-',label=labels[mode],color=colors[mode],lw=1.7,ms=4)
        ax.set(xlabel='训练轮次',ylabel=ylabel,xticks=[10,20,30,40]);ax.grid(alpha=.18)
    axes[0].legend(loc='lower right',fontsize=8.5)
    pending=report['modes']['S2']['status']['status']!='completed'
    fig.suptitle('静态语义条件平台验证曲线'+('（阶段快照）' if pending else '（主采样结果）'),fontsize=13)
    fig.tight_layout()
    apply_complete_frame(fig)
    for ext in ['png','pdf','svg']:
        fig.savefig(OUT/('validation_progress.'+ext),dpi=220,bbox_inches='tight')
    plt.close(fig)
    report['inputs_sha256']=INPUTS
    (OUT/'snapshot.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    lines=['# 训练进度只读分析快照','',f'UTC：{NOW.isoformat()}。S2是否正式完成：{not pending}。',
           '','| 模式 | best轮 | mIoU % | 单层IoU % | 少层IoU % | 前景边界F1 % |',
           '|---|---:|---:|---:|---:|---:|']
    for row in table:
        lines.append(f"| {row['mode']} | {row['best_epoch']} | {row['miou_percent']:.4f} | {row['mono_iou_percent']:.4f} | {row['few_iou_percent']:.4f} | {row['foreground_boundary_f1_percent']:.4f} |")
    lines += ['','冻结哈希和best混淆矩阵/边界指标复算通过，未修改训练。',
              '主采样或3408复核未结束时不宣布全批最终结果；检查批次status与最终comparison_summary。',
              '不同模式总loss目标不同，不能直接排名；40轮探索结果不证明充分收敛或普遍无效。',
              '中文/西文数字字体沿用用户允许的NotoSerifSC/Tinos临时替代。',
              '![验证曲线](validation_progress.png)']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    print(OUT)
    print(json.dumps({'table':table,'state':report['batch_status'],
                     'statistics':{m:v['statistics'] for m,v in report['modes'].items()},
                     'deltas':{m:report['modes'][m]['delta_vs_S0_pp'] for m in ['S1','S2']}},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
