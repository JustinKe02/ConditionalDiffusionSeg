"""Comparison panels for the GLP-Flow test27 export (outputs/inference/glpflow_test_20261022).

Reads the saved masks only (no model inference). Each saved mask is checked against the stored confusion matrix.
Layout per image (full resolution panels): RGB | GT ; E0 | E1 ; E2 | E3 ; E3R | E0-vs-E3R error map.
"""
import sys, csv, json, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager
import glpflow_core as G
from conditional_diffusion.data import PALETTE, read_manifest
from conditional_diffusion.metrics_impl import confusion

OUT = G.ROOT/'outputs/inference/glpflow_test_20261022'
ARMS = ['E0', 'E1', 'E2', 'E3', 'E3R']
ONLY_E0, ONLY_E3R, BOTH = (0, 200, 255), (255, 0, 200), (255, 255, 255)


def image_miou(m):
    i = np.diag(m).astype(float); u = m.sum(0)+m.sum(1)-np.diag(m); v = u > 0
    return float((i[v]/u[v]).mean())


def main():
    man = read_manifest(G.config('E0')['manifest']); root = Path(man['dataset_root'])
    ids = Path(man['splits']['test']['snapshot_path']).read_text().split(); assert len(ids) == 27
    conf = {a: np.load(OUT/f'{a}_confusion.npz') for a in ARMS}
    for a in ARMS: assert list(conf[a]['ids']) == ids
    results = json.loads((OUT/'test_metrics.json').read_text())
    for folder in ('comparisons_native', 'comparisons_preview'): (OUT/folder).mkdir()
    fp = font_manager.findfont('DejaVu Sans')
    title_font, legend_font = ImageFont.truetype(fp, 54), ImageFont.truetype(fp, 40)
    rows = []
    for n, ident in enumerate(ids):
        with Image.open(root/man['image_pattern'].format(id=ident)) as im: rgb = np.array(im.convert('RGB'))
        with Image.open(root/man['mask_pattern'].format(id=ident)) as im: gt = np.array(im)
        preds = {}
        for a in ARMS:
            with Image.open(OUT/a/'masks'/f'{ident}.png') as im: preds[a] = np.array(im)
            m = confusion(gt, preds[a])
            assert np.array_equal(m, conf[a]['per_image'][n]), f'{a} {ident} mask/confusion mismatch'
        score = {a: image_miou(conf[a]['per_image'][n]) for a in ARMS}
        rows.append(dict(image_id=ident, **{f'{a}_miou_present': round(100*score[a], 4) for a in ARMS}))
        w0, w3 = preds['E0'] != gt, preds['E3R'] != gt
        err = (rgb.mean(2, keepdims=True)*.35).astype(np.uint8).repeat(3, 2)
        err[w0 & ~w3], err[~w0 & w3], err[w0 & w3] = ONLY_E0, ONLY_E3R, BOTH
        panels = [rgb, PALETTE[gt]]+[PALETTE[preds[a]] for a in ARMS]+[err]
        titles = [f'{ident} | RGB', 'Ground truth']+[f"{a} (ep{results[a]['best_epoch']}) | mIoU {100*score[a]:.2f}%" for a in ARMS]
        titles.append(f'Errors: E0 only {int((w0 & ~w3).sum())} / E3R only {int((~w0 & w3).sum())} px')
        h, w = gt.shape; margin, header, footer = 24, 100, 190
        canvas = Image.new('RGB', (2*w+3*margin, 4*(h+header)+5*margin+footer), 'white'); draw = ImageDraw.Draw(canvas)
        for i, (panel, title) in enumerate(zip(panels, titles)):
            x, y = margin+(i % 2)*(w+margin), margin+(i//2)*(h+header+margin)
            draw.text((x+16, y+15), title, font=title_font, fill='black'); canvas.paste(Image.fromarray(panel), (x, y+header))
        legend = [(tuple(PALETTE[k]), name) for k, name in enumerate(('Background', 'Monolayer', 'Few-layer', 'Multilayer'))]
        legend2 = [(ONLY_E0, 'Wrong in E0 only'), (ONLY_E3R, 'Wrong in E3R only'), (BOTH, 'Wrong in both')]
        for row, items in enumerate((legend, legend2)):
            ly = canvas.height-footer+15+row*80
            for i, (color, name) in enumerate(items):
                x = 50+i*(canvas.width//4)
                draw.rectangle((x, ly, x+55, ly+45), fill=color, outline='black', width=2)
                draw.text((x+75, ly-1), name, font=legend_font, fill='black')
        canvas.save(OUT/'comparisons_native'/f'{ident}.png', compress_level=3)
        canvas.thumbnail((1600, 2600), Image.Resampling.LANCZOS); canvas.save(OUT/'comparisons_preview'/f'{ident}.png')
        print('COMPARISON', n+1, '/27', ident, flush=True)
    with (OUT/'per_image_metrics.csv').open('w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    lines = (OUT/'README.md').read_text().rstrip('\n').split('\n')
    lines += ['', '## Files', '',
              '- `<arm>/masks/`: raw class PNGs (0 background, 1 monolayer, 2 few-layer, 3 multilayer); `<arm>/color/`: palette images.',
              '- `comparisons_native/`: full-resolution 8-panel comparison per image (RGB/GT, E0/E1, E2/E3, E3R/error map); `comparisons_preview/`: downscaled copies.',
              '- Error map: dimmed RGB background; cyan = wrong only in E0, magenta = wrong only in E3R, white = wrong in both.',
              '- `per_image_metrics.csv`: per-image mIoU over classes present in GT or prediction. All saved masks were re-checked against the stored confusion matrices.',
              '', '| Image | ' + ' | '.join(ARMS) + ' | Preview | Native |', '|---|' + '---:|'*len(ARMS) + '---|---|']
    for r in rows:
        i = r['image_id']
        lines.append(f'| {i} | ' + ' | '.join(f"{r[a+'_miou_present']:.2f}" for a in ARMS)
                     + f' | [view](comparisons_preview/{i}.png) | [view](comparisons_native/{i}.png) |')
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
