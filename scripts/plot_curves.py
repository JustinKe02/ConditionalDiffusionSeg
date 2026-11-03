"""Export unsmoothed training and validation curves from long-format CSV logs."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', nargs='+', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    args.output.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for run in args.runs:
        c = json.loads((run/'config.json').read_text())
        label = f"{c['method']} C{int(c['class_balance'])} B{int(c['boundary'])} seed={c['seed']} {run.name}"
        for ax, filename, tag in [(axes[0], 'metrics_epochs.csv', 'train_epoch/loss_total'),
                                  (axes[1], 'metrics_val.csv', 'val/miou_4class')]:
            with (run/filename).open() as f:
                rows = [r for r in csv.DictReader(f) if r['tag'] == tag and r['value'] != '']
            if rows:
                ax.plot([int(r['epoch']) for r in rows], [float(r['value']) for r in rows], marker='.', label=label)
    for ax, title, ylabel in [(axes[0], 'Training objective (definitions differ by method)', 'Loss'),
                              (axes[1], 'Validation, actual observations only', 'Four-class mIoU (0–1)')]:
        ax.set(title=title, xlabel='Epoch', ylabel=ylabel)
        ax.grid(alpha=.25)
        ax.legend(fontsize=6)
    fig.tight_layout()
    for ext in ['pdf', 'svg', 'png']:
        fig.savefig(args.output/f'curves.{ext}', dpi=300)
    plt.close(fig)
    (args.output/'sources.json').write_text(json.dumps({'runs': [str(x.resolve()) for x in args.runs], 'smoothing': None}, indent=2)+'\n')
    print(args.output)


if __name__ == '__main__':
    main()
