"""Export all frozen test label predictions as exact RGB colors and native comparisons."""
import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / 'outputs/launches/p1_test_20261018'
OUT = ROOT / 'outputs/analysis/p1_test_predictions_color_20261018'
ARCHIVE = OUT.with_suffix('.zip')
PALETTE = np.array([[0, 0, 0], [0, 170, 255], [255, 190, 0], [220, 50, 90]], dtype=np.uint8)
NAMES = ['Background', 'Monolayer', 'Few-layer', 'Multilayer']
METHODS = ['direct', 'iterative', 'diffusion']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_rgb(path):
    with Image.open(path) as im:
        return im.convert('RGB')


def main():
    assert not OUT.exists() and not ARCHIVE.exists(), 'Refusing to overwrite an earlier export'
    assert shutil.disk_usage(ROOT).free > 10 * 1024 ** 3
    plan = json.loads((BATCH / 'plan.json').read_text())
    summary = json.loads((BATCH / 'comparison_summary.json').read_text())
    assert json.loads((BATCH / 'status.json').read_text())['status'] == 'completed'
    manifest = json.loads((ROOT / 'data_manifests/source_dataset.json').read_text())
    ids = plan['test_ids']
    assert len(ids) == 27
    results = {item['method']: item for item in summary['results']}
    input_paths = [BATCH / 'plan.json', BATCH / 'comparison_summary.json']
    for ident in ids:
        for pattern in ['image_pattern', 'mask_pattern']:
            path = Path(manifest['dataset_root']) / manifest[pattern].format(id=ident)
            assert sha(path) == plan['dataset_files_sha256'][str(path)]
            input_paths.append(path)
        for method in METHODS:
            input_paths.append(Path(results[method]['evaluation_directory']) / (ident + '.png'))
    inputs = {str(path): sha(path) for path in input_paths}
    OUT.mkdir(parents=True, exist_ok=False)
    for name in METHODS + ['comparison']:
        (OUT / name).mkdir()
    font_path = font_manager.findfont('DejaVu Sans')
    heading = ImageFont.truetype(font_path, 44)
    legend_font = ImageFont.truetype(font_path, 34)
    legend = Image.new('RGB', (2200, 94), 'white')
    draw = ImageDraw.Draw(legend)
    for c, name in enumerate(NAMES):
        left = 30 + c * 540
        draw.rectangle((left, 27, left + 48, 67), fill=tuple(PALETTE[c]))
        draw.text((left + 64, 25), name, font=legend_font, fill='black')
    legend.save(OUT / 'legend.png')
    outputs = []
    for index, ident in enumerate(ids, 1):
        rgb_path = Path(manifest['dataset_root']) / manifest['image_pattern'].format(id=ident)
        mask_path = Path(manifest['dataset_root']) / manifest['mask_pattern'].format(id=ident)
        rgb = load_rgb(rgb_path)
        with Image.open(mask_path) as im:
            target = np.asarray(im).copy()
        w, h = rgb.size
        assert target.shape == (h, w)
        panels = [rgb, Image.fromarray(PALETTE[target])]
        for method in METHODS:
            source = Path(results[method]['evaluation_directory']) / (ident + '.png')
            with Image.open(source) as im:
                labels = np.asarray(im).copy()
            assert labels.shape == (h, w) and labels.min() >= 0 and labels.max() <= 3
            colored = PALETTE[labels]
            path = OUT / method / (ident + '.png')
            Image.fromarray(colored).save(path)
            # Every output color must represent exactly its original label, at native size.
            with Image.open(path) as im:
                assert im.mode == 'RGB' and im.size == (w, h)
                assert np.array_equal(np.asarray(im), colored)
            panels.append(Image.fromarray(colored))
            outputs.append(dict(path=str(path.relative_to(OUT)), sha256=sha(path),
                                size_wh=[w, h], mode='RGB', source_labels=str(source)))
        margin, header, footer = 16, 100, 116
        comparison = Image.new('RGB', (5 * w + 6 * margin, h + header + footer), 'white')
        draw = ImageDraw.Draw(comparison)
        titles = [f'{ident} | RGB', 'Ground truth', 'Direct', 'Iterative (5 iterations)', 'Diffusion (20 steps)']
        for col, (panel, title) in enumerate(zip(panels, titles)):
            x = margin + col * (w + margin)
            draw.text((x + 12, 24), title, font=heading, fill='black')
            comparison.paste(panel, (x, header))
        comparison.paste(legend, ((comparison.width - legend.width) // 2, h + header + 10))
        path = OUT / 'comparison' / (ident + '.png')
        comparison.save(path)
        with Image.open(path) as im:
            assert im.mode == 'RGB' and im.size == comparison.size
        outputs.append(dict(path=str(path.relative_to(OUT)), sha256=sha(path),
                            size_wh=list(comparison.size), native_panel_size_wh=[w, h], mode='RGB'))
        print(f'EXPORTED {index}/27 {ident}', flush=True)
    # All source label PNGs, original images/masks and batch records remain byte-identical.
    assert all(sha(path) == digest for path, digest in inputs.items())
    assert all(len(list((OUT / method).glob('*.png'))) == 27 for method in METHODS)
    assert len(list((OUT / 'comparison').glob('*.png'))) == 27
    info = dict(status='completed', generated_at_utc=datetime.now(timezone.utc).isoformat(),
                scope='Visualization only from saved predictions; no inference/training, no metric recalculation or result selection.',
                ids=ids, image_count=27, color_prediction_count=81, comparison_count=27,
                palette=PALETTE.tolist(), class_names=NAMES,
                comparisons='Columns: RGB, ground truth, Direct, Iterative5, best single-head Diffusion20. All five panels preserve native2560x1922 pixels; no resizing.',
                inputs_sha256=inputs, outputs=outputs, source_files_unchanged=True,
                exact_label_to_rgb_mapping_verified=True)
    (OUT / 'provenance.json').write_text(json.dumps(info, ensure_ascii=False, indent=2) + '\n')
    lines = ['# 完整测试集预测彩色图', '',
             '共27张测试图，三组各27张原尺寸2560×1922的RGB彩色预测，总计81张；另有27张完整对照图。', '',
             '- direct/：Direct彩色预测。', '- iterative/：5次迭代模型彩色预测。',
             '- diffusion/：最佳单头、高噪声＋回灌、20步DDIM扩散彩色预测。',
             '- comparison/：原图、真实标签、Direct、Iterative、Diffusion从左到右；每个面板保持原尺寸，放大查看细节。', '',
             '颜色：黑色=背景，蓝色=单层，黄色=少层，红色=多层。legend.png为图例。', '',
             '直接从已有冻结测试预测导出，没有重新推理或训练；不修改原始0—3类别标签PNG。所有81张RGB图已逐像素核对颜色映射，全部输入文件SHA256保持不变。provenance.json记录来源、输入/输出哈希和尺寸。', '',
             '| 测试图 | Direct | Iterative | Diffusion | 完整对照 |',
             '|---|---|---|---|---|']
    for ident in ids:
        lines.append(f'| {ident} | [彩色图](direct/{ident}.png) | [彩色图](iterative/{ident}.png) | [彩色图](diffusion/{ident}.png) | [对照图](comparison/{ident}.png) |')
    (OUT / 'README.md').write_text('\n'.join(lines) + '\n')
    with zipfile.ZipFile(ARCHIVE, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for path in sorted(OUT.rglob('*')):
            if path.is_file():
                archive.write(path, arcname=str(path.relative_to(OUT.parent)))
    with zipfile.ZipFile(ARCHIVE) as archive:
        assert archive.testzip() is None
        assert len([p for p in archive.namelist() if p.endswith('.png')]) == 109
    record = dict(status='completed', export_directory=str(OUT), archive=str(ARCHIVE),
                  archive_bytes=ARCHIVE.stat().st_size, archive_sha256=sha(ARCHIVE),
                  image_count=27, color_prediction_count=81, comparison_count=27,
                  archive_integrity_verified=True, source_files_unchanged=True)
    (OUT.parent / (OUT.name + '_export_summary.json')).write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
