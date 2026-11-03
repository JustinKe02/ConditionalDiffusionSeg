"""Stage 4 of the Chapter-4 extension: unified inference-cost benchmark (no training, no model selection, 2026-10-24).

Objects: the 9 comparison baselines, GLP-Flow E0/E1/E3R (K=4; K=1/2/8 window-level only), RepELA-Net-Small in
training form and deploy (reparameterized) form, each from its validation-best EMA checkpoint, each measured in a
fresh subprocess (`--one NAME`).

Settings: RTX 3090, FP32 with TF32 disabled, deterministic algorithms off and cudnn.benchmark on (inference only).
The deterministic training substitutes (det_interpolate / det_adaptive_avg_pool2d / det_grid_sample in the baselines,
routed operators in RepELA) are replaced by the native PyTorch operators; prediction agreement with the as-evaluated
code is recorded on the first val image.

Measurements per object:
  parameters and FP32 weight size; FLOPs of one 512x512 window (all sampling steps) and of the global branch;
  batch-1 window latency (50 warm-up, 200 timed, CUDA-synchronized; median / P95) and batch-8 window throughput;
  whole-image time on val25 (2560x1922, existing predict with 35 windows and CPU fusion; first image excluded) and the
  network-only time of the same windows; peak allocated / reserved memory during whole-image prediction.
Outputs: outputs/analysis/ch4_benchmark_20261024/{objects/<name>.json, benchmark.json, benchmark.csv, README.md}.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys
import csv
import json
import time
import socket
import subprocess
import contextlib
from pathlib import Path

WORK_DIR = Path(__file__).resolve().parent
if str(WORK_DIR) not in sys.path:
    sys.path.insert(0, str(WORK_DIR))

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.flop_counter import FlopCounterMode

import glpflow_core as G
from repela_20261024 import core as R
from repela_20261024.model import native_ops

B = R.B
# CH4_BENCH_SMOKE=1 (with CH4_BENCH_OUT=/tmp/...) runs a short functional check: 2/5 timing iterations, 3 images
SMOKE = os.environ.get('CH4_BENCH_SMOKE') == '1'
OUT = Path(os.environ.get('CH4_BENCH_OUT', G.ROOT / 'outputs/analysis/ch4_benchmark_20261024'))
BASE_PLAN = G.ROOT / 'outputs/launches/p1_baselines_20261022/plan.json'
GLP_RUNS = {
    'E0': G.ROOT / 'outputs/P1/glpflow/seed_42/p1_glpflow_20261021_e0',
    'E1': G.ROOT / 'outputs/P1/glpflow/seed_42/p1_glpflow_20261021_e1',
    'E3R': G.ROOT / 'outputs/P1/glpflow/seed_42/p1_glpflow_rollout_20261022_e3r',
}
BASELINES = list(B.ORDER)
OBJECTS = (BASELINES + ['E0', 'E1', 'E3R', 'E3R_K1', 'E3R_K2', 'E3R_K8', 'repela_train', 'repela_deploy'])
WINDOW_ONLY = {'E3R_K1': 1, 'E3R_K2': 2, 'E3R_K8': 8}
WARMUP, ITERS = (2, 5) if SMOKE else (50, 200)
N_IMAGES = 3 if SMOKE else None
WIN = 512


def checkpoint_of(name):
    if name in BASELINES:
        return Path(json.loads(BASE_PLAN.read_text())['runs'][name]) / 'checkpoints/best.pt'
    if name.startswith('E'):
        return GLP_RUNS[name.split('_')[0]] / 'checkpoints/best.pt'
    return R.RUN / 'checkpoints/best.pt'


# ----------------------------------------------------------------------------- native operators

def _nat_interp(x, size=None, scale_factor=None):
    h, w = x.shape[-2:]
    if size is not None:
        ho, wo = (size, size) if isinstance(size, int) else (int(size[0]), int(size[1]))
    else:
        sf = scale_factor if isinstance(scale_factor, (tuple, list)) else (scale_factor, scale_factor)
        ho, wo = int(round(h * sf[0])), int(round(w * sf[1]))
    if (ho, wo) == (h, w):
        return x
    return F.interpolate(x, size=(ho, wo), mode='bilinear', align_corners=False)


def native_baseline_ops():
    import baselines_20261022.models as M
    import baselines_20261022.diffusion as Dm
    M.det_interpolate = _nat_interp
    Dm.det_interpolate = _nat_interp
    M.det_adaptive_avg_pool2d = lambda x, o: F.adaptive_avg_pool2d(x, o)
    M.det_grid_sample = lambda v, g: F.grid_sample(v, g, mode='bilinear', padding_mode='zeros', align_corners=False)


def bench_mode():
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


# ----------------------------------------------------------------------------- object construction

def load(name):
    """Returns model, config, window_fn(img batch), global_fn(thumb) or None, predict_fn(d), meta."""
    state = torch.load(checkpoint_of(name), map_location='cpu', weights_only=False)
    c = state['config']
    meta = dict(name=name, checkpoint=str(checkpoint_of(name)), checkpoint_sha256=G.digest(checkpoint_of(name)), best_epoch=state['epoch'])
    if name in BASELINES or name.startswith('repela'):
        model = B.build_model(c).cuda()
        model.load_state_dict(state['model'], strict=True)
        if name == 'repela_deploy':
            model.switch_to_deploy()
        model.eval()
        m = c['method']
        if m == 'ddp':
            eps = torch.randn((8, 256, WIN // 4, WIN // 4), device='cuda')
            window = lambda img: model.infer_windows(img, eps=eps[:len(img)], steps=c['sampling_steps'])[0]
            meta['steps'] = c['sampling_steps']
        elif m == 'ccdm':
            u = torch.rand((8, c['sampling_steps'], WIN, WIN), device='cuda')
            window = lambda img: model.infer_windows(img, u_field=u[:len(img)], steps=c['sampling_steps'])[0]
            meta['steps'] = c['sampling_steps']
        else:
            window = lambda img: model(img)
            meta['steps'] = 1
        predict = lambda d: B.predict(model, c, d['image'], d['id'])[0]
        return model, c, window, None, predict, meta

    model = G.GLPFlowNet(c).cuda()
    model.load_state_dict(state['model'], strict=True)
    model.eval()
    steps = WINDOW_ONLY.get(name, c['sampling_steps'] if c['flow'] else 1)
    meta['steps'] = steps
    ctx_holder = {}
    box = torch.tensor([[0.0, 0.0, WIN / 1920, WIN / 2560]])
    eps = torch.randn((8, 4, WIN, WIN), device='cuda')

    def global_fn(thumb):
        ctx_holder['ctx'] = model.global_context([thumb.cuda()])
        return ctx_holder['ctx']

    def window(img):
        b = len(img)
        ctx = ctx_holder.get('ctx')
        return G.infer_windows(model, c, img, box.expand(b, 4), None if ctx is None else ctx * b,
                               eps[:b] if c['flow'] else None, steps)[0]

    predict = lambda d: G.predict(model, c, d['image'], d['thumb'], d['id'], steps)[0]
    return model, c, window, (global_fn if c['global_stream'] else None), predict, meta


# ----------------------------------------------------------------------------- measurements

def sync_time(fn, n, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        out.append(time.perf_counter() - t0)
    return np.array(out)


def flops_of(fn):
    with FlopCounterMode(display=False) as fc:
        fn()
    return int(fc.get_total_flops())


def windows_of(image, c):
    sz, st = c['crop_size'], c['stride']
    h, w = image.shape[-2:]
    return [(y, x) for y in G.positions(h, sz, st) for x in G.positions(w, sz, st)]


@torch.no_grad()
def one(name):
    G.configure(42)
    ds = G.GLPDataset(G.config('E0'), 'val', train=False)
    model, c, window, global_fn, predict, meta = load(name)
    ctx_mgr = native_ops() if name.startswith('repela') else contextlib.nullcontext()
    d0 = ds[0]
    res = dict(meta, params=sum(p.numel() for p in model.parameters()),
               state_dict_mib=sum(t.numel() * t.element_size() for t in model.state_dict().values()) / 2**20)
    res['params_fp32_mib'] = res['params'] * 4 / 2**20
    if name not in WINDOW_ONLY:
        if global_fn is not None:
            global_fn(d0['thumb'])
        ref_pred = predict(d0)  # as-evaluated operators, deterministic mode
    bench_mode()
    if name in BASELINES:
        native_baseline_ops()
    with ctx_mgr:
        if name not in WINDOW_ONLY:
            pred = predict(d0)
            res['agreement_with_as_evaluated'] = float((pred == ref_pred).mean())
            if res['agreement_with_as_evaluated'] < 0.9999:
                G.atomic_json(OUT / 'failures' / f'{name}.json', res)
                raise AssertionError(f'{name}: native-operator agreement below 99.99%: {res["agreement_with_as_evaluated"]}')
        x1 = torch.stack([d0['image'][:, :WIN, :WIN]]).cuda()
        x8 = torch.stack([d0['image'][:, y:y + WIN, x:x + WIN] for y, x in windows_of(d0['image'], c)[:8]]).cuda()
        if global_fn is not None:
            global_fn(d0['thumb'])
            res['global_flops'] = flops_of(lambda: global_fn(d0['thumb']))
            g = sync_time(lambda: global_fn(d0['thumb']), 50, 10)
            res['global_latency_ms_median'] = float(np.median(g) * 1e3)
        else:
            res['global_flops'] = 0
            res['global_latency_ms_median'] = 0.0
        res['window_flops'] = flops_of(lambda: window(x1))
        res['window_gmacs'] = res['window_flops'] / 2 / 1e9
        n_win = len(windows_of(d0['image'], c))
        res['windows_per_image'] = n_win
        res['image_gmacs'] = (n_win * res['window_flops'] + res['global_flops']) / 2 / 1e9
        lat = sync_time(lambda: window(x1), ITERS, WARMUP)
        res['window_latency_ms_median'] = float(np.median(lat) * 1e3)
        res['window_latency_ms_p95'] = float(np.percentile(lat, 95) * 1e3)
        thr = sync_time(lambda: window(x8), 50, 10)
        res['window_throughput_per_s_b8'] = float(8 / np.median(thr))

        if name not in WINDOW_ONLY:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            pred_t, net_t = [], []
            for i in range(len(ds) if N_IMAGES is None else N_IMAGES):
                d = ds[i]
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                predict(d)
                torch.cuda.synchronize()
                pred_t.append(time.perf_counter() - t0)
                coords = windows_of(d['image'], c)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                if global_fn is not None:
                    global_fn(d['thumb'])
                for s in range(0, len(coords), 8):
                    part = coords[s:s + 8]
                    img = torch.stack([d['image'][:, y:y + WIN, x:x + WIN] for y, x in part]).cuda()
                    window(img)
                torch.cuda.synchronize()
                net_t.append(time.perf_counter() - t0)
            res['image_predict_s_median'] = float(np.median(pred_t[1:]))
            res['image_network_s_median'] = float(np.median(net_t[1:]))
            res['image_fusion_and_transfer_s_median'] = res['image_predict_s_median'] - res['image_network_s_median']
            res['images_timed'] = len(pred_t) - 1
            res['infer_peak_allocated_mib'] = torch.cuda.max_memory_allocated() / 2**20
            res['infer_peak_reserved_mib'] = torch.cuda.max_memory_reserved() / 2**20
    G.atomic_json(OUT / 'objects' / f'{name}.json', res)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items() if k != 'checkpoint'}), flush=True)


def environment():
    smi = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.used', '--format=csv,noheader'],
                         capture_output=True, text=True).stdout.strip()
    return dict(host=socket.gethostname(), gpu=torch.cuda.get_device_name(), nvidia_smi=smi, torch=torch.__version__,
                cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), cpu_threads=torch.get_num_threads(),
                precision='FP32, TF32 disabled, deterministic off, cudnn.benchmark on, native operators')


COLUMNS = ['name', 'best_epoch', 'steps', 'params', 'params_fp32_mib', 'window_gmacs', 'image_gmacs', 'windows_per_image',
           'window_latency_ms_median', 'window_latency_ms_p95', 'window_throughput_per_s_b8', 'global_latency_ms_median',
           'image_predict_s_median', 'image_network_s_median', 'image_fusion_and_transfer_s_median',
           'infer_peak_allocated_mib', 'infer_peak_reserved_mib', 'agreement_with_as_evaluated']


def main(only=None):
    (OUT / 'objects').mkdir(parents=True, exist_ok=True)
    G.configure(42)
    env = environment()
    assert 'MiB' in env['nvidia_smi']
    for name in OBJECTS if only is None else only:
        if (OUT / 'objects' / f'{name}.json').exists():
            continue
        apps = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                              capture_output=True, text=True, check=True).stdout.split()
        assert not apps, f'GPU is busy before benchmark {name}: {apps}'
        with open(OUT / f'log_{name}.txt', 'a') as log:
            rc = subprocess.call([sys.executable, '-B', '-W', 'ignore', __file__, '--one', name], stdout=log, stderr=subprocess.STDOUT)
        if rc != 0:
            raise SystemExit(f'benchmark failed for {name} (rc={rc}); see log_{name}.txt')
    rows = [json.loads((OUT / 'objects' / f'{n}.json').read_text()) for n in OBJECTS]
    bad = [r['name'] for r in rows if 'agreement_with_as_evaluated' in r and r['agreement_with_as_evaluated'] < 0.9999]
    assert not bad, f'Native-operator agreement below 99.99%: {bad}'
    G.atomic_json(OUT / 'benchmark.json', dict(environment=env, warmup=WARMUP, iterations=ITERS, objects=rows,
                                               agreement_below_99_99=bad))
    with open(OUT / 'benchmark.csv', 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction='ignore')
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    (OUT / 'README.md').write_text(
        '# 第四章统一推理成本测速（2026-10-24）\n\n'
        f"- 环境：{env['gpu']}，torch {env['torch']}，CUDA {env['cuda']}，cuDNN {env['cudnn']}，CPU 线程 {env['cpu_threads']}；{env['precision']}。\n"
        '- 每个对象独立进程，val-best EMA 权重；对比方法的确定性训练算子与 RepELA 的路由算子换为原生实现，'
        '`agreement_with_as_evaluated` 为第一张 val 图上与原评估代码的像素一致率。\n'
        '- FLOPs：`torch.utils.flop_counter.FlopCounterMode`，GMACs = FLOPs/2；只统计矩阵乘与卷积类算子，逐元素运算、插值、'
        'grid_sample 采样不计。扩散方法含全部采样步；GLP 全图分支每图一次，`image_gmacs` = 35 窗口 + 全图分支。\n'
        f'- 窗口延迟：batch 1，预热 {WARMUP}、计时 {ITERS} 次，每次 CUDA 同步，中位数/P95；吞吐为 batch 8。\n'
        '- 整图：val25 原尺寸，现有 predict（35 窗口、384 步长、batch 8、CPU 概率融合），不含读图与指标，第一张不计；'
        '网络时间为同一 35 窗口只跑网络，差值记为融合与数据搬运。峰值显存取整图预测期间。\n'
        '- E3R_K1/K2/K8 只测窗口级指标（K≠4 的步长标记未训练，仅作成本参考）。\n'
        '- 只计时，不选模，不产生精度结论。\n')
    print('BENCHMARK DONE', 'agreement<99.99%:', bad, flush=True)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--one':
        one(sys.argv[2])
    elif len(sys.argv) >= 3 and sys.argv[1] == '--only':
        main(sys.argv[2].split(','))
    else:
        main()
