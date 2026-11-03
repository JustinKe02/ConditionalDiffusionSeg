#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${DIFFUSION_PYTHON:-/root/miniconda3/envs/diffusion/bin/python}"
export TMPDIR=/tmp
export PIP_CACHE_DIR=/root/.cache/pip
"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 10), sys.version'
if ! "$PYTHON" -c 'import torch; assert torch.__version__ == "2.5.1+cu124"' 2>/dev/null; then
    wheel_dir="$(mktemp -d /tmp/diffusion-wheels.XXXXXXXX)"
    trap 'rm -rf -- "$wheel_dir"' EXIT
    "$PYTHON" -m pip download --no-cache-dir --no-deps --dest "$wheel_dir" \
        --index-url https://download.pytorch.org/whl/cu124 'torch==2.5.1'
    "$PYTHON" -m pip install --no-cache-dir --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
        "$wheel_dir"/torch-2.5.1+cu124-*.whl -r "$ROOT/requirements-diffusion.txt"
else
    "$PYTHON" -m pip install --no-cache-dir --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
        -r "$ROOT/requirements-diffusion.txt"
fi
"$PYTHON" -m pip check
"$PYTHON" -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
