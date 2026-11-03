"""Record and validate the fixed Python 3.10 diffusion training environment."""
import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import numpy
import PIL
import tensorboard
import matplotlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from conditional_diffusion.runtime import write_json, source_hashes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    if sys.version_info[:2] != (3, 10) or not torch.cuda.is_available():
        raise RuntimeError('Require Python 3.10 and CUDA-enabled PyTorch')
    versions = {'torch': torch.__version__, 'numpy': numpy.__version__, 'Pillow': PIL.__version__,
                'tensorboard': tensorboard.__version__, 'matplotlib': matplotlib.__version__}
    expected = {'torch': '2.5.1+cu124', 'numpy': '2.1.3', 'Pillow': '11.0.0',
                'tensorboard': '2.18.0', 'matplotlib': '3.9.2'}
    if versions != expected:
        raise RuntimeError(f'Environment differs from frozen recipe: {versions}')
    executable = Path(sys.executable).resolve()
    if executable.is_relative_to('/root/autodl-tmp'):
        raise RuntimeError('Training environment must live on the system disk')
    system_free = shutil.disk_usage(executable).free/1024**3
    data_free = shutil.disk_usage(ROOT).free/1024**3
    if system_free < 2 or data_free < 12:
        raise RuntimeError('Insufficient free disk space before training')
    free, total = torch.cuda.mem_get_info()
    if free < 20*1024**3:
        raise RuntimeError('Less than 20 GiB GPU memory free; check other GPU processes')
    subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True)
    freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
    args.output.with_suffix('.pip-freeze.txt').write_text(freeze)
    report = {'python': platform.python_version(), 'executable': str(executable), 'packages': versions,
              'cuda_runtime': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
              'gpu_free_gib': free/1024**3, 'gpu_total_gib': total/1024**3,
              'system_free_gib': system_free, 'data_free_gib': data_free,
              'source_sha256': source_hashes(ROOT), 'status': 'passed'}
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
