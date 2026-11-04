#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/ConditionalDiffusionSeg
HERE=$(cd "$(dirname "$0")" && pwd)
PY=/root/miniconda3/envs/diffusion/bin/python
OUT="$HERE/results"
mkdir -p "$OUT"
exec 9>"$HERE/.run.lock"
flock -n 9 || exit 75
if [ -f "$OUT/assessment.json" ] && grep -q '"phase": "completed"' "$OUT/status.json"; then exit 0; fi
echo '{"phase":"waiting_for_original_queue"}' > "$OUT/status.json"
# A shared lock waits behind the original exclusive queue, including its final evaluation.
exec 8<"$ROOT/outputs/.diffusion-seg-queue.lock"
flock -s 8
if ! "$PY" -c 'import json,sys;sys.exit(json.load(open(sys.argv[1]))["status"]!="completed")' "$ROOT/outputs/diffusion_seg_20261103/status.json"; then
  echo '{"phase":"blocked","reason":"Original batch has not completed"}' > "$OUT/status.json"
  exit 3
fi
echo '{"phase":"waiting_for_idle_gpu"}' > "$OUT/status.json"
until nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i 0 | awk -F, 'NF==2 && $1+0<750 && $2+0<10 {ok=1} END{exit !ok}'; do sleep 60; done
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$ROOT/src"
cd "$HERE"
exec "$PY" -u run.py "$ROOT" "$OUT"
