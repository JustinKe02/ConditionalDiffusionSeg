import csv
import hashlib
import json
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

GIB = 1024**3


def write_json(path, value):
    # JSON null denotes missing/undefined values; never emit nonstandard NaN.
    def clean(v):
        if isinstance(v, dict):
            return {str(k): clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [clean(x) for x in v]
        if isinstance(v, (float, np.floating)) and not np.isfinite(v):
            return None
        if isinstance(v, np.generic):
            return v.item()
        return v
    Path(path).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2)+'\n')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if torch.cuda.is_available() and state['cuda']:
        torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


def directory_size(path):
    total, seen = 0, set()
    for p in Path(path).rglob('*'):
        if p.is_symlink() or not p.is_file():
            continue
        s = p.stat()
        key = s.st_dev, s.st_ino
        if key not in seen:
            total += s.st_blocks*512
            seen.add(key)
    return total


class StorageGuard:
    def __init__(self, project, run, config):
        self.project, self.run, self.config = Path(project), Path(run), config

    def check(self, additional_bytes=0):
        free = shutil.disk_usage(self.project).free
        project_size, run_size = directory_size(self.project), directory_size(self.run)
        c = self.config
        if free-additional_bytes < c['min_free_gib']*GIB:
            raise RuntimeError('Disk reserve would be violated; preserving last successful checkpoint')
        if project_size+additional_bytes > c['project_cap_gib']*GIB:
            raise RuntimeError('Project disk limit would be exceeded')
        if run_size+additional_bytes > c['run_cap_gib']*GIB:
            raise RuntimeError('Run disk limit would be exceeded; review storage budget')
        if directory_size(self.run/'tensorboard') > c['tensorboard_cap_mib']*1024**2:
            raise RuntimeError('TensorBoard budget exceeded; review before continuing')
        if free < c['warning_free_gib']*GIB:
            print('WARNING: disk free space below warning threshold', flush=True)
        return {'disk_free_gib': free/GIB, 'project_size_gib': project_size/GIB, 'run_size_gib': run_size/GIB}


def atomic_checkpoint(path, payload, guard):
    tensor_bytes = sum(t.numel()*t.element_size() for t in payload['model'].values())
    # A last checkpoint holds weights + 2 Adam moments. Allow extra metadata/serialization.
    estimate = tensor_bytes*(3 if 'optimizer' in payload else 1)+8*1024**2
    guard.check(estimate)
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    if temp.exists():
        raise RuntimeError(f'Unresolved checkpoint temporary file: {temp}')
    try:
        with temp.open('wb') as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    except BaseException:
        # Only the temporary file created by this invocation is removed.
        temp.unlink(missing_ok=True)
        raise


class ExperimentLogger:
    def __init__(self, run, guard, resume_step=None, resume_epoch=None):
        self.run, self.guard = Path(run), guard
        if resume_step is not None:
            self._reconcile(resume_step, resume_epoch)
        self.steps = SummaryWriter(str(self.run/'tensorboard/steps'), flush_secs=30,
                                   purge_step=None if resume_step is None else resume_step+1)
        self.epochs = SummaryWriter(str(self.run/'tensorboard/epochs'), flush_secs=30,
                                    purge_step=None if resume_epoch is None else resume_epoch+1)
        self.steps.add_text('run/config', (self.run/'config.json').read_text(), 0)

    def _reconcile(self, step, epoch):
        # Back up before any mutation, including TensorBoard files subject to purge markers.
        files = list(self.run.glob('metrics_*.csv'))
        tb = self.run/'tensorboard'
        needed = sum(p.stat().st_size for p in files)+directory_size(tb)
        self.guard.check(needed+1024**2)
        backup = self.run/'resume_backups'/str(time.time_ns())
        backup.mkdir(parents=True, exist_ok=False)
        if tb.exists():
            shutil.copytree(tb, backup/'tensorboard')
        for p in files:
            shutil.copy2(p, backup/p.name)
            with p.open(newline='') as f:
                reader = csv.DictReader(f)
                fields = reader.fieldnames
                rows = [r for r in reader if int(r['global_step']) <= step and int(r['epoch']) <= epoch]
            with p.open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)

    def scalars(self, category, values, epoch, step):
        self.guard.check(256*1024)
        path = self.run/('metrics_'+category+'.csv')
        exists = path.exists()
        fields = ['wall_time_unix', 'epoch', 'global_step', 'tag', 'value']
        writer_tb = self.steps if category == 'steps' else self.epochs
        axis = step if category == 'steps' else epoch
        with path.open('a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if not exists:
                writer.writeheader()
            for tag, value in values.items():
                value = float(value)
                writer.writerow(dict(wall_time_unix=time.time(), epoch=epoch, global_step=step,
                                     tag=tag, value=value if np.isfinite(value) else ''))
                if np.isfinite(value):
                    writer_tb.add_scalar(tag, value, axis)

    def images(self, previews, epoch):
        self.guard.check(16*1024**2)
        for ident, panels in previews.items():
            for name, arr in panels.items():
                im = Image.fromarray(arr)
                im.thumbnail((512, 512), Image.Resampling.BILINEAR if name == 'rgb' else Image.Resampling.NEAREST)
                self.epochs.add_image(f'val_images/{ident}/{name}', np.asarray(im), epoch, dataformats='HWC')
        self.flush()

    def flush(self):
        self.steps.flush()
        self.epochs.flush()

    def close(self):
        self.steps.close()
        self.epochs.close()


def source_hashes(project):
    return {str(p.relative_to(project)): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ['src', 'scripts', 'configs'] for p in (Path(project)/folder).rglob('*')
            if p.is_file() and p.suffix in {'.py', '.sh', '.json'}}
