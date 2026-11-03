"""Fixed, label-free image conditions and train-only optical class statistics."""
from pathlib import Path
import hashlib
import json
import os
import time
import numpy as np
import torch
from PIL import Image
from .data import MEAN, STD, read_manifest
from .static_semantic import StaticSemanticModel
from .runtime import write_json


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_direct(checkpoint, device='cpu'):
    """Complete exact mapping of the completed Direct; no old file is changed."""
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    prefixes = {'input.':'rgb_encoder.input.', 'enc.':'rgb_encoder.blocks.',
                'down.':'rgb_encoder.down.', 'up.':'decoder.up.',
                'dec.':'decoder.blocks.', 'out.':'head.'}
    model = StaticSemanticModel('S0', seed=42)
    mapped = {'stage_timesteps':model.stage_timesteps.clone()}
    for key,value in state['model'].items():
        names = [new+key[len(old):] for old,new in prefixes.items() if key.startswith(old)]
        if len(names) != 1: raise ValueError('Unmapped Direct tensor: '+key)
        mapped[names[0]] = value
    model.load_state_dict(mapped, strict=True)
    return model.to(device).eval().requires_grad_(False)


def substrate(rgb, prior):
    """Robust linear background field; uses RGB and predicted BG only, never GT."""
    h,w = rgb.shape[:2]
    yy,xx = np.mgrid[0:h:16,0:w:16]
    design = np.stack([np.ones_like(xx),2*xx/max(w-1,1)-1,2*yy/max(h-1,1)-1],-1).reshape(-1,3)
    colors = (rgb[::16,::16].astype(np.float64)/255).reshape(-1,3)
    selected = (np.asarray(prior[0,::16,::16]).reshape(-1) >= .95)
    selected &= (colors.min(1)>.02)&(colors.max(1)<.98)
    n = int(selected.sum()); support = min(1.,n/512.)
    if n < 32:
        # RGB median is an explicitly unreliable fallback, not assumed substrate.
        coeff = np.zeros((3,3)); coeff[0] = np.median(colors,axis=0)
        return dict(coefficients=coeff.tolist(),reliability=0.,candidate_count=n,rmse=None)
    a,v = design[selected],colors[selected]; weights = np.ones(n)
    for _ in range(4):
        coeff = np.linalg.solve(a.T@(weights[:,None]*a)+np.eye(3)*1e-6,a.T@(weights[:,None]*v))
        residual = np.linalg.norm(a@coeff-v,axis=1)
        cutoff = max(.01,2.5*float(np.median(residual)))
        weights = np.minimum(1.,cutoff/np.maximum(residual,1e-8))
    rmse = float(np.sqrt(np.average(((a@coeff-v)**2).mean(1),weights=weights)))
    return dict(coefficients=coeff.tolist(),reliability=float(support*np.exp(-rmse/.05)),candidate_count=n,rmse=rmse)


def conditions_for_region(rgb, prior, meta, y=0, x=0, height=None, width=None):
    """Return C3,r1,P4; retain full-image coordinates before crop/augmentation."""
    full_h,full_w = rgb.shape[:2]
    height = full_h-y if height is None else height
    width = full_w-x if width is None else width
    patch = rgb[y:y+height,x:x+width].astype(np.float32)/255
    yy = (2*np.arange(y,y+height,dtype=np.float32)/max(full_h-1,1)-1)[:,None]
    xx = (2*np.arange(x,x+width,dtype=np.float32)/max(full_w-1,1)-1)[None,:]
    coeff = np.asarray(meta['coefficients'],dtype=np.float32)
    background = np.clip(coeff[0]+xx[:,:,None]*coeff[1]+yy[:,:,None]*coeff[2],.02,1.)
    contrast = np.clip((background-patch)/(background+1e-6),-1.,1.)
    reliable = np.full(patch.shape[:2],meta['reliability'],dtype=np.float32)
    reliable *= (patch.min(2)>.02)&(patch.max(2)<.98)
    p = np.asarray(prior[:,y:y+height,x:x+width],dtype=np.float32).copy()
    p = np.maximum(p,1e-4);p /= p.sum(0,keepdims=True)
    return np.concatenate([contrast.transpose(2,0,1),reliable[None],p],0)


class ConditionDataset(torch.utils.data.Dataset):
    """Same seed/epoch/id crop and augmentation RNG calls as MaterialDataset."""
    def __init__(self, manifest, cache, split='train', crop=512, seed=42):
        self.manifest=read_manifest(manifest);self.root=Path(self.manifest['dataset_root'])
        if split not in ['train','val']:raise ValueError('Only train/val authorized')
        self.ids=Path(self.manifest['splits'][split]['snapshot_path']).read_text().split()
        self.cache=Path(cache);self.index=json.loads((self.cache/'index.json').read_text())
        self.crop,self.seed,self.epoch=crop,seed,0

    def __len__(self):return len(self.ids)

    def __getitem__(self,index):
        ident=self.ids[index]
        with Image.open(self.root/self.manifest['image_pattern'].format(id=ident)) as im:rgb=np.asarray(im.convert('RGB')).copy()
        with Image.open(self.root/self.manifest['mask_pattern'].format(id=ident)) as im:mask=np.asarray(im).copy()
        prior=np.load(self.cache/(ident+'.npy'),mmap_mode='r')
        h,w=mask.shape;y=x=0
        if self.crop:
            if min(h,w)<self.crop:raise ValueError('Unexpected padded training input')
            rng=np.random.default_rng(np.random.SeedSequence([self.seed,self.epoch,index]))
            y=int(rng.integers(h-self.crop+1));x=int(rng.integers(w-self.crop+1))
            aux=conditions_for_region(rgb,prior,self.index['images'][ident]['substrate'],y,x,self.crop,self.crop)
            rgb,mask=rgb[y:y+self.crop,x:x+self.crop],mask[y:y+self.crop,x:x+self.crop]
            k=int(rng.integers(4));rgb,mask,aux=np.rot90(rgb,k),np.rot90(mask,k),np.rot90(aux,k,axes=(1,2))
            for axis in [0,1]:
                if rng.random()<.5:rgb,mask,aux=np.flip(rgb,axis),np.flip(mask,axis),np.flip(aux,axis+1)
        else:aux=conditions_for_region(rgb,prior,self.index['images'][ident]['substrate'])
        image=torch.from_numpy(np.ascontiguousarray(rgb)).permute(2,0,1).float()/255
        return (image-MEAN)/STD,torch.from_numpy(np.ascontiguousarray(aux)),torch.from_numpy(np.ascontiguousarray(mask)).long(),ident


def prepare_conditions(config, cache):
    """Fixed prior cache: RGB-only train/val; labels accessed only for train stats."""
    from .static_experiment import predict, configure
    from .inference import positions
    configure(config);cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    manifest=read_manifest(config['manifest']);root=Path(manifest['dataset_root'])
    teacher=load_direct(config['prior_checkpoint'],'cuda');teacher_sha=digest(config['prior_checkpoint'])
    cfg=dict(config,mode='S0',window_batch_size=1,return_probabilities=True)
    index_path=cache/'index.json'
    index=json.loads(index_path.read_text()) if index_path.exists() else dict(status='preparing',prior_sha256=teacher_sha,images={})
    assert index['prior_sha256']==teacher_sha
    sums=np.zeros((4,3),np.float64);outer=np.zeros((4,3,3),np.float64);counts=np.zeros(4,np.int64)
    for split in ['train','val']:
        ids=Path(manifest['splits'][split]['snapshot_path']).read_text().split()
        for ident in ids:
            image_path=root/manifest['image_pattern'].format(id=ident)
            before=digest(image_path)
            with Image.open(image_path) as im:rgb=np.asarray(im.convert('RGB')).copy()
            entry=index['images'].get(ident);path=cache/(ident+'.npy')
            if entry is not None:
                assert entry['image_sha256']==before and digest(path)==entry['prior_file_sha256']
            else:
                begin=time.perf_counter();image=torch.from_numpy(rgb).permute(2,0,1).float()/255
                image=(image-MEAN)/STD
                _,calls,timing,prob=predict(teacher,None,image,ident,cfg,True)
                p=prob.numpy();p=(1-4e-4)*p+1e-4;p/=p.sum(0,keepdims=True)
                pending=path.with_suffix('.npy.tmp')
                with pending.open('wb') as f:np.save(f,p.astype(np.float16));f.flush();os.fsync(f.fileno())
                pending.replace(path)
                prior=np.load(path,mmap_mode='r')
                entry=dict(split=split,image_sha256=before,prior_file_sha256=digest(path),
                           shape=list(p.shape),substrate=substrate(rgb,prior),prior_calls=calls,
                           seconds_including_readout_storage=time.perf_counter()-begin,prior_timing=timing)
                index['images'][ident]=entry;write_json(index_path,index)
                print(dict(stage='prior_cache',id=ident,split=split,completed=len(index['images']),seconds=entry['seconds_including_readout_storage']),flush=True)
            assert digest(image_path)==before
            if split=='train':
                mask_path=root/manifest['mask_pattern'].format(id=ident);mask_sha=digest(mask_path)
                with Image.open(mask_path) as im:mask=np.asarray(im).copy()
                prior=np.load(path,mmap_mode='r')
                # Bounded train-only sample: one pixel every 8 native pixels.
                aux=conditions_for_region(rgb,prior,entry['substrate'])[:3,::8,::8].transpose(1,2,0)
                target=mask[::8,::8]
                for k in range(4):
                    values=aux[target==k].astype(np.float64)
                    counts[k]+=len(values);sums[k]+=values.sum(0);outer[k]+=values.T@values
                assert digest(mask_path)==mask_sha;entry['train_mask_sha256']=mask_sha
    assert (counts>0).all() and len(index['images'])==145
    mean=sums/counts[:,None]
    covariance=outer/counts[:,None,None]-mean[:,:,None]*mean[:,None,:]
    covariance=.9*covariance+.1*np.eye(3)[None]*np.trace(covariance,axis1=1,axis2=2)[:,None,None]/3+np.eye(3)[None]*.0025
    distance=((mean[:,None]-mean[None,:])**2).sum(-1)
    variance=(np.trace(covariance,axis1=1,axis2=2)[:,None]+np.trace(covariance,axis1=1,axis2=2)[None,:])/3
    similarity=np.exp(-distance/(2*variance+1e-8))+.05
    relation=similarity/similarity.sum(1,keepdims=True)
    statistics=dict(train_counts=counts.tolist(),mean=mean.tolist(),covariance=covariance.tolist(),
                    similarity=similarity.tolist(),global_relation=relation.tolist(),pixel_stride=8,
                    covariance_floor=.0025,class_prior='equal, not background-frequency weighted',
                    prior_sha256=teacher_sha,scope='train labels only; val RGB prior cache only, no val labels')
    write_json(cache/'optical_statistics.json',statistics)
    index.update(status='completed',image_count=145,statistics_sha256=digest(cache/'optical_statistics.json'),
                 cache_bytes=sum(f.stat().st_size for f in cache.glob('*.npy')))
    write_json(index_path,index)
    assert digest(config['prior_checkpoint'])==teacher_sha
    return index
