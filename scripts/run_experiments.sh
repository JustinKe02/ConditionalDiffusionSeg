#!/usr/bin/env bash
# P0 and the three P1 pilots only. Does not run formal experiments or test evaluation.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${DIFFUSION_PYTHON:-/root/miniconda3/envs/diffusion/bin/python}"
STAGE=all
BATCH_ID="$(date -u +%Y%m%dT%H%M%SZ)"
while (($#)); do
    case "$1" in
        --stage) STAGE="${2:?Missing stage}"; shift 2 ;;
        --batch-id) BATCH_ID="${2:?Missing batch ID}"; shift 2 ;;
        -h|--help)
            echo 'Usage: bash scripts/run_experiments.sh [--stage all|p0|pilot] [--batch-id ID]'
            echo 'Reuse the same batch ID to resume. pilot requires a passed P0 in that batch.'
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ "$STAGE" =~ ^(all|p0|pilot)$ ]] || { echo 'Invalid stage' >&2; exit 2; }
[[ "$BATCH_ID" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Invalid batch ID' >&2; exit 2; }
cd "$ROOT"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
BATCH="$ROOT/outputs/launches/$BATCH_ID"
mkdir -p "$BATCH"
exec 9>"$ROOT/outputs/.experiments.lock"
flock -n 9 || { echo 'Another experiment batch holds the GPU lock.' >&2; exit 2; }
CURRENT_STAGE=preflight
write_status() {
    "$PYTHON" - "$BATCH/status.json" "$1" "$CURRENT_STAGE" "${2:-0}" <<'PY'
import json, os, sys
from datetime import datetime, timezone
from pathlib import Path
path = Path(sys.argv[1])
value = {'status': sys.argv[2], 'stage': sys.argv[3], 'exit_code': int(sys.argv[4]),
         'updated_at_utc': datetime.now(timezone.utc).isoformat(), 'shell_pid': os.getppid()}
temp = path.with_suffix('.tmp')
temp.write_text(json.dumps(value, indent=2)+'\n')
temp.replace(path)
PY
}
on_exit() {
    local code=$?
    trap - EXIT
    if ((code == 0)); then write_status completed 0; else write_status failed "$code"; fi
    exit "$code"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
write_status running
echo "Batch: $BATCH | stage: $STAGE | interpreter: $PYTHON"
"$PYTHON" scripts/check_environment.py --output "$BATCH/environment.json"
"$PYTHON" scripts/check_storage.py --runs 6 --parameters 5132676 > "$BATCH/storage_preflight.json"
"$PYTHON" -m unittest discover -s tests -v
"$PYTHON" scripts/smoke_resources.py --output "$BATCH/resource_smoke.json"
"$PYTHON" scripts/verify_pipeline.py --output "$BATCH/pipeline_verification.json"

run_one() {
    local config="$1" method="$2" phase="$3" action
    local run="$ROOT/outputs/$phase/$method/seed_42/$BATCH_ID"
    action="$("$PYTHON" - "$config" "$run" "$method" <<'PY'
import json, sys
from pathlib import Path
config, run, method = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
if not run.exists():
    print('new')
else:
    expected = json.loads(config.read_text())
    expected.update(method=method, seed=42)
    if not (run/'config.json').exists() or json.loads((run/'config.json').read_text()) != expected:
        raise SystemExit('Existing run configuration differs; preserve it and use a new batch ID')
    sys.path.insert(0, str(Path.cwd()/'src'))
    from conditional_diffusion.runtime import source_hashes
    if json.loads((run/'run_metadata.json').read_text())['source_sha256'] != source_hashes(Path.cwd()):
        raise SystemExit('Source changed since this run started; preserve it and use a new batch ID')
    status = json.loads((run/'status.json').read_text())
    if not (run/'checkpoints/last.pt').exists():
        raise SystemExit('Existing run has no recovery checkpoint; inspect it and use a new batch ID')
    print('completed' if status['status'] == 'completed' else 'resume')
PY
)"
    case "$action" in
        new) "$PYTHON" scripts/train.py --config "$config" --method "$method" --seed 42 --run-id "$BATCH_ID" ;;
        resume) "$PYTHON" scripts/train.py --resume "$run/checkpoints/last.pt" ;;
        completed) echo "Already completed: $run" ;;
        *) echo 'Unexpected run state' >&2; return 2 ;;
    esac
}

if [[ "$STAGE" == all || "$STAGE" == p0 ]]; then
    CURRENT_STAGE=p0_overfit
    write_status running
    "$PYTHON" scripts/prepare_p0.py
    cp configs/p0.json "$BATCH/p0_config.json"
    for method in direct iterative diffusion; do
        run_one configs/p0.json "$method" P0
    done
    "$PYTHON" scripts/check_p0.py --runs \
        "$ROOT/outputs/P0/direct/seed_42/$BATCH_ID" \
        "$ROOT/outputs/P0/iterative/seed_42/$BATCH_ID" \
        "$ROOT/outputs/P0/diffusion/seed_42/$BATCH_ID" --output "$BATCH/p0_gate.json"
    CURRENT_STAGE=p0_full_val_benchmark
    write_status running
    "$PYTHON" scripts/evaluate.py \
        --checkpoint "$ROOT/outputs/P0/diffusion/seed_42/$BATCH_ID/checkpoints/best.pt" \
        --split val --full-val --window-size 512 --stride 384 --steps 20 \
        | tee "$BATCH/full_val_benchmark.log"
    "$PYTHON" scripts/summarize_batch.py --batch-id "$BATCH_ID" --benchmark-only
fi

if [[ "$STAGE" == all || "$STAGE" == pilot ]]; then
    "$PYTHON" - "$BATCH" <<'PY'
import hashlib, json, sys
from pathlib import Path
batch = Path(sys.argv[1])
sys.path.insert(0, str(Path.cwd()/'src'))
from conditional_diffusion.runtime import source_hashes
gate = json.loads((batch/'p0_gate.json').read_text())
if not gate['passed']:
    raise SystemExit('P0 memorization gate failed; stop before P1')
if gate['source_sha256'] != source_hashes(Path.cwd()):
    raise SystemExit('Code/config changed since P0 gate; rerun P0 checks before P1')
if gate['fixed_crops_sha256'] != hashlib.sha256(Path('data_manifests/p0_fixed_crops.json').read_bytes()).hexdigest():
    raise SystemExit('Fixed crop specification changed since P0 gate')
if not json.loads((batch/'benchmark_summary.json').read_text())['passed']:
    raise SystemExit('P0 complete-val timing benchmark missing or failed')
PY
    CURRENT_STAGE=p1_pilot
    write_status running
    cp configs/pilot.json "$BATCH/pilot_config.json"
    for method in direct iterative diffusion; do
        run_one configs/pilot.json "$method" P1
    done
    "$PYTHON" scripts/summarize_batch.py --batch-id "$BATCH_ID"
fi
echo "Requested stages completed: $BATCH"
