"""GLP-Flow: global-prior-bridged hybrid-hourglass flow matching for 4-class layer segmentation.

One registered study (E0 direct, E1 direct+global, E2 flow from noise, E3 GLP-Flow).
Historical src/scripts/configs are imported read-only and never modified.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys, json, math, time, hashlib, copy
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
BATCH = ROOT/'outputs/launches/p1_glpflow_20261021'
RUN_ROOT = ROOT/'outputs/P1/glpflow/seed_42'

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint
from PIL import Image
from conditional_diffusion.data import MEAN, STD, read_manifest
from conditional_diffusion.inference import positions, preview
from conditional_diffusion.metrics_impl import confusion, summarize, boundary_counts, summarize_boundaries
from conditional_diffusion.runtime import (seed_all, rng_state, restore_rng, StorageGuard,
                                           atomic_checkpoint, ExperimentLogger)

ARMS = {
    'E0': dict(global_stream=False, flow=False, source='none'),
    'E1': dict(global_stream=True, flow=False, source='prior_deterministic'),
    'E2': dict(global_stream=True, flow=True, source='gaussian_noise'),
    'E3': dict(global_stream=True, flow=True, source='prior_uncertainty'),
}
THUMB = 4          # full image (cropped to a multiple of 64) -> thumbnail factor
PRIOR = 16         # prior grid factor relative to the full image


def config(arm):
    assert arm in ARMS
    return dict(arm=arm, **ARMS[arm], schema='glpflow_v1', seed=42, sampling_seed=3407,
                manifest=str(ROOT/'data_manifests/source_dataset.json'), device='cuda',
                crop_size=512, stride=384, crops_per_image=2, thin_crop_fraction=.5,
                batch_size=8, workers=12, epochs=100, warmup_epochs=5, lr=2.8e-4, min_lr=1e-6,
                weight_decay=1e-4, grad_clip=1., val_every=10, window_batch_size=8,
                sampling_steps=4, self_condition_prob=.5, shortcut_fraction=.25,
                prior_loss_weight=.5, source_noise_scale=1., t_logit_mean=-.5, t_logit_std=1.,
                ema_sigma_rel=.10, photometric=dict(gain=.10, gamma=.10, vignette=.15),
                widths=dict(local=[32, 64, 128, 192], token=256, global_=[32, 64, 128, 256, 256]),
                heads=4, na_kernel=7, boundary_tolerance=2, preview_count=4, log_every=20,
                precision='FP32, TF32 disabled', data_scope='original train120/val25 (m109/m31 kept); test untouched',
                storage=dict(min_free_gib=8, warning_free_gib=12, project_cap_gib=26, run_cap_gib=1,
                             tensorboard_cap_mib=256))


def configure(seed=42):
    seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, obj):
    def clean(x):
        if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)): return [clean(v) for v in x]
        if isinstance(x, np.generic): return clean(x.item())
        if isinstance(x, float) and not math.isfinite(x): return None
        return x
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    with tmp.open('w') as f:
        json.dump(clean(obj), f, ensure_ascii=False, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


# ----------------------------------------------------------------------------- data

def normalize(rgb):
    return (torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float()/255-MEAN)/STD


def global_inputs(rgb, mask=None):
    """Thumbnail of the (augmented) full image and soft prior targets. Crops to a multiple of 64."""
    h, w = (rgb.shape[0]//64)*64, (rgb.shape[1]//64)*64
    img = Image.fromarray(np.ascontiguousarray(rgb[:h, :w]))
    thumb = normalize(np.asarray(img.resize((w//THUMB, h//THUMB), Image.Resampling.BOX)))
    soft = None
    if mask is not None:
        m = mask[:h, :w].reshape(h//PRIOR, PRIOR, w//PRIOR, PRIOR)
        soft = torch.from_numpy(np.stack([(m == k).mean((1, 3)) for k in range(4)]).astype(np.float32))
    return thumb, soft, (h, w)


def photometric(rgb, g, p):
    x = rgb.astype(np.float32)/255
    x = x**float(np.exp(g.uniform(-p['gamma'], p['gamma'])))
    x = x*(1+g.uniform(-p['gain'], p['gain'], size=3)).astype(np.float32)
    h, w = x.shape[:2]
    cy, cx = g.uniform(.3, .7)*h, g.uniform(.3, .7)*w
    yy = ((np.arange(h, dtype=np.float32)-cy)/(h/2))[:, None]
    xx = ((np.arange(w, dtype=np.float32)-cx)/(w/2))[None, :]
    x = x*(1-g.uniform(0, p['vignette'])*(yy**2+xx**2)/2)[:, :, None]
    return (np.clip(x, 0, 1)*255+.5).astype(np.uint8)


class GLPDataset(Dataset):
    """Training: crops_per_image crops per image per epoch; val/test: full images."""
    def __init__(self, c, split='train', train=True, fixed_crops=None):
        self.c, self.train = c, train
        self.manifest = read_manifest(c['manifest']); self.root = Path(self.manifest['dataset_root'])
        self.ids = Path(self.manifest['splits'][split]['snapshot_path']).read_text().split()
        self.fixed = fixed_crops; self.epoch = 0

    def __len__(self):
        if self.fixed is not None: return len(self.fixed)
        return len(self.ids)*self.c['crops_per_image'] if self.train else len(self.ids)

    def raw(self, ident):
        with Image.open(self.root/self.manifest['image_pattern'].format(id=ident)) as f: rgb = np.array(f.convert('RGB'))
        with Image.open(self.root/self.manifest['mask_pattern'].format(id=ident)) as f: mask = np.array(f)
        assert rgb.shape[:2] == mask.shape and mask.ndim == 2 and mask.max() < 4
        return rgb, mask

    def __getitem__(self, i):
        c = self.c
        if self.fixed is not None:
            spec = self.fixed[i]; rgb, mask = self.raw(spec['id']); s = spec['size']
            y, x = spec['top'], spec['left']
        elif not self.train:
            ident = self.ids[i]; rgb, mask = self.raw(ident)
            thumb, soft, hw = global_inputs(rgb, mask)
            return dict(image=normalize(rgb), mask=torch.from_numpy(mask.astype(np.int64)), thumb=thumb, soft=soft, hw=hw, id=ident)
        else:
            ident = self.ids[i//c['crops_per_image']]
            g = np.random.default_rng(np.random.SeedSequence([c['seed'], self.epoch, i]))
            rgb, mask = self.raw(ident)
            rgb = photometric(rgb, g, c['photometric'])
            k = int(g.integers(4)); rgb, mask = np.rot90(rgb, k), np.rot90(mask, k)
            if g.random() < .5: rgb, mask = rgb[:, ::-1], mask[:, ::-1]
            rgb, mask = np.ascontiguousarray(rgb), np.ascontiguousarray(mask)
            s = c['crop_size']; h, w = mask.shape
            thin = np.flatnonzero(((mask == 1) | (mask == 2)).ravel()[::7]) if g.random() < c['thin_crop_fraction'] else np.empty(0)
            if len(thin):
                py, px = divmod(int(thin[g.integers(len(thin))])*7, w)
                y = int(np.clip(py-s//2+g.integers(-s//4, s//4+1), 0, h-s)); x = int(np.clip(px-s//2+g.integers(-s//4, s//4+1), 0, w-s))
            else:
                y, x = int(g.integers(h-s+1)), int(g.integers(w-s+1))
        thumb, soft, hw = global_inputs(rgb, mask)
        box = torch.tensor([y/hw[0], x/hw[1], (y+s)/hw[0], (x+s)/hw[1]], dtype=torch.float32)
        return dict(image=normalize(rgb[y:y+s, x:x+s]), mask=torch.from_numpy(mask[y:y+s, x:x+s].astype(np.int64)),
                    thumb=thumb, soft=soft, box=box)


def collate(items):
    out = dict(image=torch.stack([d['image'] for d in items]), mask=torch.stack([d['mask'] for d in items]),
               box=torch.stack([d['box'] for d in items]))
    out['thumb'] = [d['thumb'] for d in items]; out['soft'] = [d['soft'] for d in items]
    return out


# ----------------------------------------------------------------------------- modules

def mp_sum(a, b, t=.5):
    """EDM2 magnitude-preserving sum."""
    return a.lerp(b, t)/math.sqrt((1-t)**2+t**2)


def sincos_2d(h, w, dim, box=None, device='cpu'):
    """Absolute sin-cos embedding of normalized full-image coordinates; box=(y0,x0,y1,x1)."""
    y0, x0, y1, x1 = (0., 0., 1., 1.) if box is None else [float(v) for v in box]
    ys = y0+(torch.arange(h, device=device, dtype=torch.float32)+.5)/h*(y1-y0)
    xs = x0+(torch.arange(w, device=device, dtype=torch.float32)+.5)/w*(x1-x0)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    freq = 2**torch.arange(dim//4, device=device, dtype=torch.float32)*math.pi
    parts = [f(v.reshape(-1, 1)*freq) for v in (yy, xx) for f in (torch.sin, torch.cos)]
    return torch.cat(parts, 1)  # (h*w, dim)


def attention(q, k, v):
    """Explicit softmax attention (deterministic under torch.use_deterministic_algorithms)."""
    return (q*q.shape[-1]**-.5 @ k.transpose(-2, -1)).softmax(-1) @ v


def timestep_embedding(t, dim=128):
    freq = torch.exp(-math.log(10000)*torch.arange(dim//2, device=t.device)/(dim//2))
    a = (t.float()*1000)[:, None]*freq[None]
    return torch.cat([a.sin(), a.cos()], 1)


class CCA(nn.Module):
    """DiCo compact channel attention."""
    def __init__(self, c):
        super().__init__(); self.fc = nn.Conv2d(c, c, 1)
    def forward(self, x):
        return x*torch.sigmoid(self.fc(x.mean((2, 3), keepdim=True)))


class Modulation(nn.Module):
    """adaLN-Zero from a per-sample vector plus a spatial map (uncertainty)."""
    def __init__(self, c, cond, n=6, spatial=True):
        super().__init__(); self.n = n
        self.vec = nn.Linear(cond, n*c); self.map = nn.Conv2d(1, n*c, 1) if spatial else None
        for m in (self.vec, self.map):
            if m is not None: nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, cond, umap, size):
        m = self.vec(F.silu(cond))[:, :, None, None]
        if umap is not None and self.map is not None:
            m = m+self.map(F.adaptive_avg_pool2d(umap, size))
        return m.chunk(self.n, 1)


class DiCoBlock(nn.Module):
    """DiCo conv module (1x1, dw3x3, GELU, CCA, 1x1) + Mix-FFN, optional adaLN-Zero."""
    def __init__(self, c, cond=0, spatial=True):
        super().__init__(); self.cond = cond
        self.n1, self.n2 = nn.GroupNorm(8, c, affine=not cond), nn.GroupNorm(8, c, affine=not cond)
        self.conv = nn.Sequential(nn.Conv2d(c, c, 1), nn.Conv2d(c, c, 3, padding=1, groups=c), nn.GELU(), CCA(c), nn.Conv2d(c, c, 1))
        self.ffn = nn.Sequential(nn.Conv2d(c, 2*c, 1), nn.Conv2d(2*c, 2*c, 3, padding=1, groups=2*c), nn.GELU(), nn.Conv2d(2*c, c, 1))
        if cond: self.mod = Modulation(c, cond, spatial=spatial)
    def forward(self, x, cond=None, umap=None):
        if not self.cond:
            x = x+self.conv(self.n1(x)); return x+self.ffn(self.n2(x))
        s1, b1, g1, s2, b2, g2 = self.mod(cond, umap, x.shape[-2:])
        x = x+g1*self.conv(self.n1(x)*(1+s1)+b1)
        return x+g2*self.ffn(self.n2(x)*(1+s2)+b2)


class NABlock(nn.Module):
    """Neighborhood attention (HDiT) in pure PyTorch via unfold; adaLN-Zero; Mix-FFN."""
    def __init__(self, c, cond, heads=4, kernel=7):
        super().__init__(); self.h, self.k = heads, kernel
        self.n1, self.n2 = nn.GroupNorm(8, c, affine=False), nn.GroupNorm(8, c, affine=False)
        self.qkv, self.proj = nn.Conv2d(c, 3*c, 1), nn.Conv2d(c, c, 1)
        self.ffn = nn.Sequential(nn.Conv2d(c, 2*c, 1), nn.Conv2d(2*c, 2*c, 3, padding=1, groups=2*c), nn.GELU(), nn.Conv2d(2*c, c, 1))
        self.rel = nn.Parameter(torch.zeros(heads, kernel*kernel, 1))
        self.mod = Modulation(c, cond)
    def attend(self, x):
        b, c, h, w = x.shape; d = c//self.h; k = self.k
        q, kk, v = self.qkv(x).chunk(3, 1)
        q = q.reshape(b, self.h, d, 1, h*w)*d**-.5
        kk = F.unfold(kk, k, padding=k//2).reshape(b, self.h, d, k*k, h*w)
        v = F.unfold(v, k, padding=k//2).reshape(b, self.h, d, k*k, h*w)
        valid = F.unfold(torch.ones(1, 1, h, w, device=x.device), k, padding=k//2).reshape(1, 1, k*k, h*w) > 0
        logits = (q*kk).sum(2)+self.rel
        attn = logits.masked_fill(~valid, -1e4).softmax(2)
        return self.proj((attn[:, :, None]*v).sum(3).reshape(b, c, h, w))
    def forward(self, x, cond, umap=None):
        s1, b1, g1, s2, b2, g2 = self.mod(cond, umap, x.shape[-2:])
        x = x+g1*self.attend(self.n1(x)*(1+s1)+b1)
        return x+g2*self.ffn(self.n2(x)*(1+s2)+b2)


class TransformerBlock(nn.Module):
    """Plain pre-norm transformer block for the global stream."""
    def __init__(self, c, heads):
        super().__init__(); self.h = heads
        self.n1, self.n2 = nn.LayerNorm(c), nn.LayerNorm(c)
        self.qkv, self.proj = nn.Linear(c, 3*c), nn.Linear(c, c)
        self.mlp = nn.Sequential(nn.Linear(c, 4*c), nn.GELU(), nn.Linear(4*c, c))
    def forward(self, x):
        b, n, c = x.shape
        q, k, v = self.qkv(self.n1(x)).reshape(b, n, 3, self.h, c//self.h).permute(2, 0, 3, 1, 4)
        x = x+self.proj(attention(q, k, v).transpose(1, 2).reshape(b, n, c))
        return x+self.mlp(self.n2(x))


class JointBlock(nn.Module):
    """MM-DiT joint attention over streams with stream-specific weights and adaLN-Zero."""
    def __init__(self, c, cond, heads, streams):
        super().__init__(); self.h, self.streams = heads, streams
        mk = lambda: nn.ModuleDict(dict(n1=nn.LayerNorm(c, elementwise_affine=False), n2=nn.LayerNorm(c, elementwise_affine=False),
                                        qkv=nn.Linear(c, 3*c), proj=nn.Linear(c, c),
                                        mlp=nn.Sequential(nn.Linear(c, 4*c), nn.GELU(), nn.Linear(4*c, c)), mod=nn.Linear(cond, 6*c)))
        self.s = nn.ModuleList([mk() for _ in range(streams)])
        for s in self.s: nn.init.zeros_(s['mod'].weight); nn.init.zeros_(s['mod'].bias)
    def forward(self, tokens, cond):
        mods, qkv, lens = [], [], []
        for x, s in zip(tokens, self.s):
            m = s['mod'](F.silu(cond))[:, None].chunk(6, 2); mods.append(m)
            qkv.append(s['qkv'](s['n1'](x)*(1+m[0])+m[1])); lens.append(x.shape[1])
        b, c = tokens[0].shape[0], tokens[0].shape[2]
        q, k, v = torch.cat(qkv, 1).reshape(b, sum(lens), 3, self.h, c//self.h).permute(2, 0, 3, 1, 4)
        out = attention(q, k, v).transpose(1, 2).reshape(b, sum(lens), c).split(lens, 1)
        result = []
        for x, o, m, s in zip(tokens, out, mods, self.s):
            x = x+m[2]*s['proj'](o)
            result.append(x+m[5]*s['mlp'](s['n2'](x)*(1+m[3])+m[4]))
        return result


class GlobalStream(nn.Module):
    """Whole-image thumbnail -> global tokens (1/32 thumb) + coarse full-image prior (1/4 thumb)."""
    def __init__(self, widths, token, heads):
        super().__init__(); w = widths
        self.stem = nn.Sequential(nn.Conv2d(3, w[0], 3, 2, 1), DiCoBlock(w[0]))
        self.l1 = nn.Sequential(nn.Conv2d(w[0], w[1], 3, 2, 1), DiCoBlock(w[1]), DiCoBlock(w[1]))
        self.l2 = nn.Sequential(nn.Conv2d(w[1], w[2], 3, 2, 1), DiCoBlock(w[2]), DiCoBlock(w[2]))
        self.d3, self.t3 = nn.Conv2d(w[2], w[3], 3, 2, 1), nn.ModuleList([TransformerBlock(w[3], heads) for _ in range(2)])
        self.d4, self.t4 = nn.Conv2d(w[3], w[4], 3, 2, 1), nn.ModuleList([TransformerBlock(w[4], heads) for _ in range(2)])
        self.lat = nn.ModuleList([nn.Conv2d(ch, 128, 1) for ch in w[1:]])
        self.head = nn.Sequential(DiCoBlock(128), nn.GroupNorm(8, 128), nn.GELU(), nn.Conv2d(128, 4, 1))
        self.token = nn.Linear(w[4], token)
    def seq(self, x, blocks):
        b, c, h, w = x.shape
        t = x.flatten(2).transpose(1, 2)+sincos_2d(h, w, c, device=x.device)[None]
        for blk in blocks: t = blk(t)
        return t, t.transpose(1, 2).reshape(b, c, h, w)
    def forward(self, thumb):
        f1 = self.l1(self.stem(thumb)); f2 = self.l2(f1)
        _, f3 = self.seq(self.d3(f2), self.t3)
        t4, f4 = self.seq(self.d4(f3), self.t4)
        top = None
        for f, lat in zip([f4, f3, f2, f1], reversed(self.lat)):
            z = lat(f); top = z if top is None else z+F.interpolate(top, size=z.shape[-2:], mode='nearest')
        h, w = f4.shape[-2:]
        tokens = self.token(t4)+0  # absolute positions are carried by the sincos added in seq()
        return tokens, self.head(top), (h, w)


class LocalEncoder(nn.Module):
    """Window RGB encoder, cached across flow steps."""
    def __init__(self, w):
        super().__init__()
        self.l0 = nn.Sequential(nn.Conv2d(3, w[0], 3, padding=1), DiCoBlock(w[0]))
        self.l1 = nn.Sequential(nn.Conv2d(w[0], w[1], 3, 2, 1), DiCoBlock(w[1]))
        self.l2 = nn.Sequential(nn.Conv2d(w[1], w[2], 3, 2, 1), DiCoBlock(w[2]), DiCoBlock(w[2]))
        self.l3 = nn.Sequential(nn.Conv2d(w[2], w[3], 3, 2, 1), DiCoBlock(w[3]), DiCoBlock(w[3]))
    def forward(self, x):
        out = []
        for layer in (self.l0, self.l1, self.l2, self.l3):
            x = layer[0](x)
            for blk in layer[1:]: x = run_block(blk, x)
            out.append(x)
        return out


class Up(nn.Module):
    def __init__(self, cin, cout):
        super().__init__(); self.conv = nn.Conv2d(cin, cout, 3, padding=1)
    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode='nearest'))


class GLPFlowNet(nn.Module):
    def __init__(self, c):
        super().__init__(); self.c = c
        w, tok, heads, cond = c['widths']['local'], c['widths']['token'], c['heads'], 256
        self.local = LocalEncoder(w)
        self.global_stream = GlobalStream(c['widths']['global_'], tok, heads) if c['global_stream'] else None
        self.cond = nn.Sequential(nn.Linear(128+16, cond), nn.SiLU(), nn.Linear(cond, cond))
        self.level = nn.Embedding(4, 16)
        self.stem = nn.Conv2d(13, w[0], 3, padding=1)
        self.e0 = nn.ModuleList([DiCoBlock(w[0], cond, spatial=False)])
        self.d1, self.e1 = nn.Conv2d(w[0], w[1], 3, 2, 1), nn.ModuleList([DiCoBlock(w[1], cond, spatial=False)])
        self.d2, self.e2 = nn.Conv2d(w[1], w[2], 3, 2, 1), nn.ModuleList([DiCoBlock(w[2], cond) for _ in range(2)])
        self.d3, self.e3 = nn.Conv2d(w[2], w[3], 3, 2, 1), nn.ModuleList([NABlock(w[3], cond, heads, c['na_kernel']) for _ in range(2)])
        self.fuse = nn.ModuleList([nn.Conv2d(ch, ch, 1) for ch in w])
        self.d4, self.rgb_token = nn.Conv2d(w[3], tok, 3, 2, 1), nn.Conv2d(w[3], tok, 3, 2, 1)
        self.joint = nn.ModuleList([JointBlock(tok, cond, heads, 3 if c['global_stream'] else 2) for _ in range(2)])
        self.u3, self.r3 = Up(tok, w[3]), nn.ModuleList([NABlock(w[3], cond, heads, c['na_kernel'])])
        self.u2, self.r2 = Up(w[3], w[2]), nn.ModuleList([DiCoBlock(w[2], cond)])
        self.u1, self.r1 = Up(w[2], w[1]), nn.ModuleList([DiCoBlock(w[1], cond, spatial=False)])
        self.u0, self.r0 = Up(w[1], w[0]), nn.ModuleList([DiCoBlock(w[0], cond, spatial=False)])
        self.head = nn.Sequential(nn.GroupNorm(8, w[0]), nn.SiLU(), nn.Conv2d(w[0], 4, 1))

    # -- global context and window prior
    def global_context(self, thumbs):
        if self.global_stream is None: return None
        out = []
        for th in thumbs:  # orientation may differ across a batch
            tokens, prior, grid = self.global_stream(th[None])
            out.append((tokens[0], prior[0], grid))
        return out

    @staticmethod
    def window_prior(prior_logits, box, size):
        """Bilinear ROI of a full-image prior; box normalized (y0,x0,y1,x1)."""
        y0, x0, y1, x1 = [float(v) for v in box]
        ys = torch.linspace(y0, y1, size+1, device=prior_logits.device)[:-1]+(y1-y0)/(2*size)
        xs = torch.linspace(x0, x1, size+1, device=prior_logits.device)[:-1]+(x1-x0)/(2*size)
        gy, gx = torch.meshgrid(ys*2-1, xs*2-1, indexing='ij')
        grid = torch.stack([gx, gy], -1)[None]
        return F.grid_sample(prior_logits[None], grid, mode='bilinear', padding_mode='border', align_corners=False)[0]

    def denoise(self, feats, x, prior, umap, selfcond, t, level, gtok, gpos, box):
        cond = self.cond(torch.cat([timestep_embedding(t), self.level(level)], 1))
        h = self.stem(torch.cat([x, umap, prior, selfcond], 1))
        skips = []
        for i, (down, blocks) in enumerate([(None, self.e0), (self.d1, self.e1), (self.d2, self.e2), (self.d3, self.e3)]):
            if down is not None: h = down(h)
            h = h+self.fuse[i](feats[i])
            for blk in blocks: h = run_block(blk, h, cond, umap)
            skips.append(h)
        s = self.d4(h); r = self.rgb_token(feats[3]); b, cc, hh, ww = s.shape
        pos = torch.stack([sincos_2d(hh, ww, cc, bx, s.device) for bx in box])
        streams = [s.flatten(2).transpose(1, 2)+pos, r.flatten(2).transpose(1, 2)+pos]
        if gtok is not None: streams.append(gtok+gpos)
        for blk in self.joint: streams = run_block(blk, streams, cond)
        h = streams[0].transpose(1, 2).reshape(b, cc, hh, ww)
        for up, blocks, skip in [(self.u3, self.r3, skips[3]), (self.u2, self.r2, skips[2]), (self.u1, self.r1, skips[1]), (self.u0, self.r0, skips[0])]:
            h = mp_sum(up(h), skip)
            for blk in blocks: h = run_block(blk, h, cond, umap)
        return self.head(h)


def run_block(blk, *args):
    """Activation recomputation during training only; numerically identical forward."""
    if torch.is_grad_enabled() and blk.training:
        return checkpoint(blk, *args, use_reentrant=False)
    return blk(*args)


def encode_targets(mask):
    return 2*F.one_hot(mask, 4).permute(0, 3, 1, 2).float()-1


def entropy_map(p):
    return (-(p.clamp_min(1e-6)*p.clamp_min(1e-6).log()).sum(1, keepdim=True)/math.log(4)).clamp(0, 1)


def make_source(c, prior_p, umap, eps):
    if c['source'] == 'prior_uncertainty': return (2*prior_p-1)+c['source_noise_scale']*umap*eps
    if c['source'] == 'gaussian_noise': return eps
    if c['source'] == 'prior_deterministic': return 2*prior_p-1
    return torch.zeros_like(eps)


def window_conditions(model, c, ctx, boxes, size, device):
    """Prior probabilities, uncertainty, global tokens/positions for each window."""
    b = len(boxes)
    if ctx is None:
        z = torch.zeros(b, 4, size, size, device=device)
        return z, z[:, :1], None, None, None
    with torch.no_grad():  # the window prior is a detached condition; prior loss acts on the global grid
        priors = torch.stack([model.window_prior(ctx[i][1].detach(), boxes[i], size) for i in range(b)])
        p = priors.softmax(1)
    gtok = torch.stack([ctx[i][0] for i in range(b)])
    gpos = torch.stack([sincos_2d(*ctx[i][2], gtok.shape[-1], None, device) for i in range(b)])
    return p, entropy_map(p), gtok, gpos, priors


def dice_loss(logits, mask, weight=None):
    probs = logits.softmax(1); gt = F.one_hot(mask, 4).permute(0, 3, 1, 2).float(); dims = (0, 2, 3)
    if weight is not None:
        probs, gt = probs*weight[:, None, None, None], gt*weight[:, None, None, None]
    return 1-((2*(probs*gt).sum(dims)+1e-6)/(probs.sum(dims)+gt.sum(dims)+1e-6))[1:].mean()


def euler(x, y_state, t, dt):
    return x+(dt/(1-t))[:, None, None, None]*(y_state-x)


def train_step(model, opt, batch, c, device):
    image, mask, boxes = batch['image'].to(device), batch['mask'].to(device), batch['box']
    b, s = image.shape[0], image.shape[-1]
    feats = model.local(image)
    ctx = model.global_context([t.to(device) for t in batch['thumb']])
    p, umap, gtok, gpos, _ = window_conditions(model, c, ctx, boxes, s, device)
    p, umap = p.detach(), umap.detach()
    terms = {}
    prior_loss = image.new_zeros(())
    if ctx is not None:
        for (_, pl, _), soft in zip(ctx, batch['soft']):
            soft = soft.to(device)
            prior_loss = prior_loss+(-(soft*pl.log_softmax(0)).sum(0).mean()+dice_loss(pl[None], soft.argmax(0)[None]))/2
        prior_loss = prior_loss/b
    y = encode_targets(mask)
    t = torch.zeros(b, device=device); level = torch.zeros(b, dtype=torch.long, device=device)
    selfcond = torch.zeros_like(p); flow_w = torch.ones(b, device=device); short_w = torch.zeros(b, device=device)
    target_short = None
    if c['flow']:
        eps = torch.randn_like(y)
        x0 = make_source(c, p, umap, eps)
        n_short = int(round(b*c['shortcut_fraction']))
        perm = torch.randperm(b, device=device); short_idx = perm[:n_short].sort().values
        flow_w[short_idx] = 0.; short_w[short_idx] = 1.
        t = torch.sigmoid(c['t_logit_mean']+c['t_logit_std']*torch.randn(b, device=device)).clamp(0, .98)
        if n_short:
            lv = torch.randint(1, 4, (n_short,), device=device); size = 2.0**lv/8
            grid_index = (torch.rand(n_short, device=device)*(1/size)).floor()
            t[short_idx], level[short_idx] = grid_index*size, lv
        x = (1-t)[:, None, None, None]*x0+t[:, None, None, None]*y
        if torch.rand((), device=device) < c['self_condition_prob']:
            with torch.no_grad():
                selfcond = model.denoise(feats, x, p, umap, selfcond, t, level, gtok, gpos, boxes).softmax(1)
        if n_short:
            with torch.no_grad():  # two half steps of the next-finer level define the shortcut target
                i, half = short_idx, (2.0**level[short_idx]/8)/2
                sub = lambda v: None if v is None else v[i]
                feats_i = [f[i] for f in feats]; boxes_i = [boxes[int(j)] for j in i]
                xs, ts, lv0 = x[i], t[i], level[i]-1
                y1 = 2*model.denoise(feats_i, xs, p[i], umap[i], selfcond[i], ts, lv0, sub(gtok), sub(gpos), boxes_i).softmax(1)-1
                xm = euler(xs, y1, ts, half)
                y2 = 2*model.denoise(feats_i, xm, p[i], umap[i], (y1+1)/2, ts+half, lv0, sub(gtok), sub(gpos), boxes_i).softmax(1)-1
                xe = euler(xm, y2, ts+half, half)
                target_short = torch.zeros_like(y)
                target_short[short_idx] = (xs+((1-ts)/(2*half))[:, None, None, None]*(xe-xs)).clamp(-1, 1)
    else:
        x = make_source(c, p, umap, torch.zeros_like(y))
    logits = model.denoise(feats, x, p, umap, selfcond, t, level, gtok, gpos, boxes)
    ce = F.cross_entropy(logits, mask, reduction='none').mean((1, 2))
    terms['loss_ce'] = (ce*flow_w).sum()/flow_w.sum(); terms['loss_dice_fg'] = dice_loss(logits, mask, flow_w)
    total = .5*terms['loss_ce']+.5*terms['loss_dice_fg']
    if c['flow'] and target_short is not None:
        mse = (2*logits.softmax(1)-1-target_short).square().mean((1, 2, 3))
        terms['loss_shortcut'] = (mse*short_w).sum()/short_w.sum()
        total = total+terms['loss_shortcut']
    if ctx is not None:
        terms['loss_prior'] = prior_loss; total = total+c['prior_loss_weight']*prior_loss
    terms['loss_total'] = total
    if not torch.isfinite(total): raise FloatingPointError('Nonfinite loss')
    opt.zero_grad(set_to_none=True); total.backward()
    grad = nn.utils.clip_grad_norm_(model.parameters(), c['grad_clip'], error_if_nonfinite=True)
    opt.step()
    return {k: float(v.detach()) for k, v in terms.items()}, float(grad)


# ----------------------------------------------------------------------------- inference

def level_for_steps(steps):
    return {8: 0, 4: 1, 2: 2, 1: 3}[steps]


@torch.no_grad()
def infer_windows(model, c, img, boxes, ctx, eps, steps):
    """Window batch -> final logits. ctx: per-window global context list (or None); eps: source noise."""
    device = img.device; b, sz = img.shape[0], img.shape[-1]
    feats = model.local(img)
    p, umap, gtok, gpos, _ = window_conditions(model, c, ctx, boxes, sz, device)
    zeros = torch.zeros(b, device=device)
    if not c['flow']:
        return model.denoise(feats, make_source(c, p, umap, torch.zeros_like(p)), p, umap, torch.zeros_like(p), zeros,
                             zeros.long(), gtok, gpos, boxes), b
    x = make_source(c, p, umap, eps); sc = torch.zeros_like(p)
    lv = torch.full((b,), level_for_steps(steps), device=device, dtype=torch.long)
    for k in range(steps):
        t = torch.full((b,), k/steps, device=device)
        logits = model.denoise(feats, x, p, umap, sc, t, lv, gtok, gpos, boxes)
        sc = logits.softmax(1); x = euler(x, 2*sc-1, t, torch.full_like(t, 1/steps))
    return logits, b*steps


@torch.no_grad()
def predict(model, c, image, thumb, ident, steps=None, sampling_seed=None, return_prob=False):
    """Only RGB, its thumbnail and the image ID enter prediction; no ground truth argument."""
    device = next(model.parameters()).device
    steps = c['sampling_steps'] if steps is None else steps
    sampling_seed = c['sampling_seed'] if sampling_seed is None else sampling_seed
    h, w = image.shape[-2:]; sz = c['crop_size']
    hw = ((h//64)*64, (w//64)*64)
    ctx = model.global_context([thumb.to(device)])
    field = None
    if c['flow']:
        gen = torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(f'{ident}:{sampling_seed}'.encode()).digest()[:8], 'little') % (2**63-1))
        field = torch.randn((4, h, w), generator=gen)
    coords = [(y, x) for y in positions(h, sz, c['stride']) for x in positions(w, sz, c['stride'])]
    total, counts = torch.zeros((4, h, w)), torch.zeros((h, w)); nfe = 0
    for start in range(0, len(coords), c['window_batch_size']):
        part = coords[start:start+c['window_batch_size']]; b = len(part)
        img = torch.stack([image[:, y:y+sz, x:x+sz] for y, x in part]).to(device)
        boxes = torch.tensor([[y/hw[0], x/hw[1], (y+sz)/hw[0], (x+sz)/hw[1]] for y, x in part])
        eps = None if field is None else torch.stack([field[:, y:y+sz, x:x+sz] for y, x in part]).to(device)
        logits, calls = infer_windows(model, c, img, boxes, None if ctx is None else ctx*b, eps, steps); nfe += calls
        prob = logits.softmax(1).cpu()
        for pr, (y, x0) in zip(prob, part):
            total[:, y:y+sz, x0:x0+sz] += pr; counts[y:y+sz, x0:x0+sz] += 1
    assert bool((counts > 0).all())
    prob = total/counts
    pred = prob.argmax(0).numpy().astype(np.uint8)
    return (pred, nfe, prob) if return_prob else (pred, nfe)


@torch.no_grad()
def evaluate(model, c, split='val', steps=None, sampling_seed=None, limit=None, on_image=None):
    model.eval(); ds = GLPDataset(c, split, train=False)
    n = len(ds) if limit is None else min(limit, len(ds))
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); start = time.perf_counter()
    matrices, bounds, ids, previews, nfe = [], [], [], {}, 0
    for i in range(n):
        d = ds[i]
        pred, calls = predict(model, c, d['image'], d['thumb'], d['id'], steps, sampling_seed)
        truth = d['mask'].numpy(); nfe += calls
        matrices.append(confusion(truth, pred)); bounds.append(boundary_counts(truth, pred, c['boundary_tolerance'])); ids.append(d['id'])
        if i < c['preview_count']: previews[d['id']] = preview(d['image'], truth, pred)
        if on_image: on_image(d['id'], pred)
    torch.cuda.synchronize()
    m, bc = np.sum(matrices, 0), np.sum(bounds, 0)
    metrics = {**classification_metrics(m), **summarize_boundaries(bc), 'seconds': time.perf_counter()-start,
               'image_count': len(ids), 'nfe_total': nfe, 'peak_reserved_mib': torch.cuda.max_memory_reserved()/2**20}
    return metrics, dict(ids=np.array(ids), per_image=np.array(matrices), global_matrix=m, boundary_counts=bc,
                         per_image_boundary=np.array(bounds)), previews


def classification_metrics(m):
    out = summarize(m)
    for i, name in enumerate(('background', 'monolayer', 'fewlayer', 'multilayer')):
        out['precision_'+name] = float(m[i, i]/m[:, i].sum()) if m[:, i].sum() else float('nan')
        out['recall_'+name] = float(m[i, i]/m[i].sum()) if m[i].sum() else float('nan')
    out['thin_mean_iou'] = (out['iou_monolayer']+out['iou_fewlayer'])/2
    out['few_to_mono_rate'] = float(m[2, 1]/m[2].sum()); out['mono_to_background_rate'] = float(m[1, 0]/m[1].sum())
    return out


# ----------------------------------------------------------------------------- training

class PowerEMA:
    """EDM2 power-function EMA profile with fixed sigma_rel (online; no post-hoc snapshots)."""
    def __init__(self, model, sigma_rel):
        self.gamma = self.solve(sigma_rel); self.shadow = copy.deepcopy(model).eval().requires_grad_(False); self.n = 0
    @staticmethod
    def solve(sigma_rel):
        lo, hi = 0., 1000.
        for _ in range(200):
            g = (lo+hi)/2; s = math.sqrt((g+1)/((g+2)**2*(g+3)))
            lo, hi = (g, hi) if s > sigma_rel else (lo, g)
        return (lo+hi)/2
    @torch.no_grad()
    def update(self, model):
        self.n += 1; beta = (1-1/self.n)**(self.gamma+1)
        for e, p in zip(self.shadow.parameters(), model.parameters()): e.lerp_(p.detach(), 1-beta)
        for e, p in zip(self.shadow.buffers(), model.buffers()): e.copy_(p)
    def state_dict(self): return dict(n=self.n, gamma=self.gamma, model=self.shadow.state_dict())
    def load_state_dict(self, s): self.n = s['n']; self.shadow.load_state_dict(s['model'])


def lr_lambda(c):
    def f(epoch):
        if epoch < c['warmup_epochs']: return (epoch+1)/c['warmup_epochs']
        prog = min(1., (epoch-c['warmup_epochs'])/max(1, c['epochs']-c['warmup_epochs']-1)); lo = c['min_lr']/c['lr']
        return lo+(1-lo)*.5*(1+math.cos(math.pi*prog))
    return f


def build(c, device='cuda'):
    configure(c['seed'])
    model = GLPFlowNet(c).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=c['lr'], weight_decay=c['weight_decay'])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(c))
    return model, opt, sched, PowerEMA(model, c['ema_sigma_rel'])


def make_loader(ds, c, epoch):
    ds.epoch = epoch
    return DataLoader(ds, batch_size=c['batch_size'], shuffle=True, num_workers=c['workers'], collate_fn=collate,
                      drop_last=False, pin_memory=True, generator=torch.Generator().manual_seed(c['seed']+epoch),
                      persistent_workers=False)


def checkpoint_payload(model, opt, sched, ema, c, epoch, step, best, plan_sha):
    return dict(model=model.state_dict(), optimizer=opt.state_dict(), scheduler=sched.state_dict(), ema=ema.state_dict(),
                rng=rng_state(), config=c, epoch=epoch, global_step=step, best_miou=best, plan_sha256=plan_sha)


def train(c, run, plan_sha='unfrozen', max_batches=None, stop_after_epoch=None, guard_fn=None):
    run = Path(run); device = c['device']
    model, opt, sched, ema = build(c, device)
    last = run/'checkpoints/last.pt'; epoch0 = step = 0; best = -1.
    sg = StorageGuard(ROOT, run, c['storage'])
    if last.exists():
        state = torch.load(last, map_location='cpu', weights_only=False)
        assert state['config'] == c and state['plan_sha256'] == plan_sha, 'resume requires identical config/plan'
        model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer']); sched.load_state_dict(state['scheduler'])
        ema.load_state_dict(state['ema']); restore_rng(state['rng'])
        epoch0, step, best = state['epoch'], state['global_step'], state['best_miou']; del state
    else:
        (run/'checkpoints').mkdir(parents=True, exist_ok=True); atomic_json(run/'config.json', c)
        n = sum(p.numel() for p in model.parameters())
        atomic_json(run/'run_metadata.json', dict(parameters=n, fp32_bytes=4*n, arm=c['arm'], plan_sha256=plan_sha,
                    torch=torch.__version__, gpu=torch.cuda.get_device_name()))
        atomic_checkpoint(last, checkpoint_payload(model, opt, sched, ema, c, 0, 0, best, plan_sha), sg)
    if epoch0 >= c['epochs']:
        return dict(status='completed', epoch=epoch0)
    ds = GLPDataset(c, 'train')
    logger = ExperimentLogger(run, sg, step if epoch0 else None, epoch0 if epoch0 else None)
    atomic_json(run/'status.json', dict(status='running', last_completed_epoch=epoch0, global_step=step))
    try:
        for epoch in range(epoch0+1, c['epochs']+1):
            if guard_fn: guard_fn()
            model.train(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.perf_counter()
            sums, seen, lr = {}, 0, opt.param_groups[0]['lr']
            for bi, batch in enumerate(make_loader(ds, c, epoch)):
                if max_batches is not None and bi >= max_batches: break
                terms, grad = train_step(model, opt, batch, c, device); ema.update(model)
                n = batch['image'].shape[0]; seen += n; step += 1
                for k, v in terms.items(): sums[k] = sums.get(k, 0.)+v*n
                if step % c['log_every'] == 0:
                    logger.scalars('steps', {'train_step/'+k: v for k, v in dict(**terms, grad_norm=grad, lr=lr).items()}, epoch, step)
                    atomic_json(run/'progress.json', dict(epoch=epoch, global_step=step, loss=terms['loss_total'],
                                updated_at_utc=datetime.now(timezone.utc).isoformat()))
            torch.cuda.synchronize(); secs = time.perf_counter()-t0
            vals = {'train_epoch/'+k: v/seen for k, v in sums.items()}
            vals.update({'resources/train_seconds': secs, 'resources/samples_per_second': seen/secs,
                         'resources/peak_reserved_mib': torch.cuda.max_memory_reserved()/2**20, 'optim/lr': lr})
            logger.scalars('epochs', vals, epoch, step); metrics = None
            if max_batches is None and (epoch % c['val_every'] == 0 or epoch == c['epochs']):
                metrics, mats, previews = evaluate(ema.shadow, c)
                logger.scalars('val', {'val/'+k: v for k, v in metrics.items()}, epoch, step); logger.images(previews, epoch)
                atomic_json(run/f'validation_epoch{epoch}.json', metrics)
                if metrics['miou_4class'] > best:
                    best = metrics['miou_4class']
                    atomic_checkpoint(run/'checkpoints/best.pt', dict(model=ema.shadow.state_dict(), config=c, epoch=epoch,
                                      global_step=step, validation_metrics=metrics, plan_sha256=plan_sha, weights='power_ema'), sg)
                    atomic_json(run/'best_val_metrics.json', dict(epoch=epoch, metrics=metrics))
                    np.savez_compressed(run/'best_val_confusion.npz', **mats)
            sched.step(); logger.flush()
            atomic_checkpoint(last, checkpoint_payload(model, opt, sched, ema, c, epoch, step, best, plan_sha), sg)
            atomic_json(run/'status.json', dict(status='completed' if epoch == c['epochs'] else 'running',
                        last_completed_epoch=epoch, global_step=step, best_miou=best))
            print(dict(arm=c['arm'], epoch=epoch, step=step, loss=round(sums['loss_total']/seen, 5), sec=round(secs, 1),
                       val=None if metrics is None else round(metrics['miou_4class'], 5)), flush=True)
            if stop_after_epoch is not None and epoch >= stop_after_epoch: break
    except BaseException as e:
        atomic_json(run/'status.json', dict(status='interrupted', error=str(e), global_step=step)); raise
    finally:
        logger.close()
    return dict(status='completed' if epoch == c['epochs'] else 'stopped', epoch=epoch, best=best)
