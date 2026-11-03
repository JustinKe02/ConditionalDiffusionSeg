#!/usr/bin/env bash
# One immutable serial batch. Launch this script with nohup after pushing HEAD.
set -euo pipefail
export GIT_TERMINAL_PROMPT=0
export GIT_SSH_COMMAND="${GIT_SSH_COMMAND:-ssh -o BatchMode=yes -o ConnectTimeout=20}"

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=python
BATCH=
EXPECTED_COMMIT=
while (($#)); do
    case "$1" in
        --python) PYTHON=$2; shift 2 ;;
        --batch) BATCH=$2; shift 2 ;;
        --expected-commit) EXPECTED_COMMIT=$2; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
if [[ -z "$BATCH" || ! "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
    echo 'Usage: run_diffusion_seg.sh --batch outputs/NAME --expected-commit FULL_SHA [--python PATH]' >&2
    exit 2
fi
cd -- "$ROOT"
command -v flock >/dev/null
mkdir -p outputs
exec 9>outputs/.diffusion-seg-queue.lock
if ! flock -n 9; then
    echo 'Another diffusion segmentation queue owns the project lock.' >&2
    exit 75
fi
BATCH=$("$PYTHON" - "$ROOT" "$BATCH" <<'PY'
from pathlib import Path
import sys
root, batch = Path(sys.argv[1]), Path(sys.argv[2]).resolve()
if not batch.is_relative_to(root / 'outputs') or batch == root / 'outputs':
    raise SystemExit('Batch must be a named directory under project/outputs')
batch.mkdir(parents=True, exist_ok=True)
print(batch)
PY
)

# The small control plane stores no model weights and never changes configuration.
control() {
    "$PYTHON" - "$ROOT" "$BATCH" "$EXPECTED_COMMIT" "$@" <<'PY'
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import sys
import tempfile

root, batch, commit = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
action, *args = sys.argv[4:]
sys.path.insert(0, str(root / 'src'))

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def sha(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            checksum.update(chunk)
    return checksum.hexdigest()

def atomic_json(path, value):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix=path.name + '.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

status_path = batch / 'status.json'
status = read(status_path) if status_path.exists() else dict(status='initializing', arms={})
if action == 'status':
    phase, arm, message = args
    if status['status'] == 'completed' or (status['status'] == 'failed' and phase == 'failed'):
        raise SystemExit(0)
    status.update(status=phase, current_arm=arm or None, message=message, expected_commit=commit,
                  pid=os.getppid(), updated_at_utc=datetime.now(timezone.utc).isoformat())
    if arm:
        record = status['arms'].setdefault(arm, dict(retries=0))
        record.update(status=phase, message=message)
        if phase == 'retrying':
            record['retries'] += 1
    status['completed_arms'] = [name for name, record in status['arms'].items()
                                if record['status'] == 'arm_completed']
    atomic_json(status_path, status)
elif action == 'prepare':
    from conditional_diffusion.diffusion_seg_experiment import provenance, validate_config
    order = ['ordinal', 'onehot', 'permuted', 'direct']
    snapshots, runs, budgets = {}, {}, {}
    protocol = ('manifest', 'device', 'workers', 'cpu_threads', 'epochs', 'batch_size', 'crop_size', 'stride', 'crops_per_image', 'thin_crop_fraction',
                'photometric_gain', 'photometric_gamma', 'lr', 'min_lr', 'warmup_epochs',
                'weight_decay', 'grad_clip', 'timesteps', 'sampling_steps', 'sampling_seed',
                'window_batch_size', 'val_every', 'boundary_tolerance', 'seed', 'ce_weight', 'dice_weight',
                'reconstruction_weight', 'prior_loss_weight', 'model')
    common = None
    (batch / 'configs').mkdir(exist_ok=True)
    (batch / 'logs').mkdir(exist_ok=True)
    (batch / 'evaluations').mkdir(exist_ok=True)
    for arm in order:
        source = root / 'configs' / f'diffusion_seg_{arm}.json'
        config = read(source)
        validate_config(config)
        settings = {name: config[name] for name in protocol}
        if common is not None and settings != common:
            raise SystemExit(f'Matched training protocol differs for {arm}')
        common = settings
        destination = batch / 'configs' / f'{arm}.json'
        if destination.exists():
            if destination.read_bytes() != source.read_bytes():
                raise SystemExit(f'Immutable config differs for {arm}')
        else:
            with destination.open('xb') as stream:
                stream.write(source.read_bytes())
                stream.flush()
                os.fsync(stream.fileno())
        snapshots[arm] = dict(path=str(destination), sha256=sha(destination))
        runs[arm] = str(batch / arm)
        manifest = read(Path(config['manifest']))
        exposures = manifest['splits']['train']['count'] * config['crops_per_image']
        validation_epochs = sorted(set(range(config['val_every'], config['epochs'] + 1, config['val_every']))
                                   | {config['epochs']})
        budgets[arm] = dict(epochs=config['epochs'], batch_size=config['batch_size'],
                           updates_per_epoch=math.ceil(exposures / config['batch_size']),
                           total_updates=config['epochs'] * math.ceil(exposures / config['batch_size']),
                           total_crop_exposures=config['epochs'] * exposures,
                           validation_epochs=validation_epochs, training_validation_passes=len(validation_epochs),
                           final_best_validation_passes=1)
    plan = dict(schema_version=1, expected_commit=commit, order=order, configs=snapshots,
                runs=runs, budgets=budgets, protocol=common, provenance=provenance(config, root),
                selection='validation four-class mIoU; evaluate best before next arm',
                retry_policy='At most one automatic checkpoint resume per arm; stop immediately on CUDA OOM')
    plan_path = batch / 'plan.json'
    if plan_path.exists() and read(plan_path) != plan:
        raise SystemExit('Immutable batch plan differs from requested batch')
    if not plan_path.exists():
        atomic_json(plan_path, plan)
elif action == 'probe':
    import torch
    from conditional_diffusion.diffusion_seg_experiment import provenance
    arm = args[0]
    plan = read(batch / 'plan.json')
    config_path = Path(plan['configs'][arm]['path'])
    if sha(config_path) != plan['configs'][arm]['sha256']:
        raise SystemExit('Snapshot configuration hash changed')
    config = read(config_path)
    if provenance(config, root) != plan['provenance']:
        raise SystemExit('Training source or split hashes changed')
    run = Path(plan['runs'][arm])
    complete = False
    if (run / 'last.pt').exists():
        state = torch.load(run / 'last.pt', map_location='cpu', weights_only=False)
        if state['config'] != config or state['provenance'] != plan['provenance']:
            raise SystemExit('Checkpoint configuration or provenance differs from plan')
        if state['epoch'] > config['epochs']:
            raise SystemExit('Checkpoint epoch exceeds the immutable schedule')
        if state['epoch'] == config['epochs'] and state['best'] is not None and (run / 'best.pt').exists():
            try:
                selected = torch.load(run / 'best.pt', map_location='cpu', weights_only=False)
                complete = (selected['config'] == config and selected['provenance'] == plan['provenance']
                            and selected['epoch'] == state['best']['epoch']
                            and selected['model'].keys() == state['best']['model'].keys()
                            and all(torch.equal(value, selected['model'][key])
                                    for key, value in state['best']['model'].items()))
            except (OSError, RuntimeError, EOFError, ValueError, KeyError, pickle.UnpicklingError):
                complete = False  # The trainer repairs best.pt from committed last.pt.
    evaluation = batch / 'evaluations' / f'{arm}.json'
    evaluated = (complete and evaluation.exists() and
                 read(evaluation).get('checkpoint_sha256') == sha(run / 'best.pt') and
                 read(evaluation).get('plan_sha256') == sha(batch / 'plan.json'))
    print('evaluated' if evaluated else ('trained' if complete else 'pending'))
elif action == 'retries':
    print(status['arms'].get(args[0], {}).get('retries', 0))
elif action == 'evaluation':
    arm = args[0]
    temporary = batch / 'evaluations' / f'{arm}.json.pending'
    result = read(temporary)
    result.update(checkpoint_sha256=sha(batch / arm / 'best.pt'), plan_sha256=sha(batch / 'plan.json'),
                  expected_commit=commit)
    atomic_json(batch / 'evaluations' / f'{arm}.json', result)
    temporary.unlink()
else:
    raise SystemExit(f'Unknown control action {action}')
PY
}

CURRENT_ARM=
finish() {
    code=$?
    if ((code != 0)); then
        control status failed "$CURRENT_ARM" "Queue exited with code $code; see per-arm stderr/stdout logs" || true
    fi
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

verify_uploaded_source() {
    [[ "$(git rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || {
        echo 'HEAD differs from the expected uploaded commit.' >&2; return 1;
    }
    local remote_commit attempt
    for attempt in 1 2 3; do
        if remote_commit=$(git ls-remote --exit-code --heads origin refs/heads/feat/diffusion-seg | awk '{print $1}'); then
            break
        fi
        if ((attempt == 3)); then
            echo 'Remote commit verification failed after three network attempts.' >&2
            return 1
        fi
        echo "Retrying remote commit verification after network attempt $attempt." >&2
        sleep "$((attempt * 5))"
    done
    [[ "$remote_commit" == "$EXPECTED_COMMIT" ]] || {
        echo 'origin/feat/diffusion-seg does not contain the exact expected HEAD.' >&2; return 1;
    }
    git diff --quiet HEAD -- src scripts configs || {
        echo 'Training source or configuration has staged/unstaged changes.' >&2; return 1;
    }
    [[ -z "$(git ls-files --others --exclude-standard -- src scripts configs)" ]] || {
        echo 'Untracked training source/config files must be published first.' >&2; return 1;
    }
}

control status initializing '' 'Checking uploaded code and immutable configuration'
verify_uploaded_source
control prepare
for CURRENT_ARM in ordinal onehot permuted direct; do
    verify_uploaded_source
    config="$BATCH/configs/$CURRENT_ARM.json"
    run="$BATCH/$CURRENT_ARM"
    state=$(control probe "$CURRENT_ARM")
    if [[ "$state" == 'evaluated' ]]; then
        control status arm_completed "$CURRENT_ARM" 'Complete checkpoint and matching best evaluation already exist'
        continue
    fi
    if [[ "$state" != 'trained' ]]; then
        resume=()
        [[ ! -d "$run" ]] || resume=(--resume)
        control status running "$CURRENT_ARM" 'Training with frozen configuration'
        if "$PYTHON" scripts/diffusion_seg.py train --config "$config" --run "$run" "${resume[@]}" \
            >>"$BATCH/logs/$CURRENT_ARM.train.stdout.log" 2>>"$BATCH/logs/$CURRENT_ARM.train.stderr.log"; then
            :
        else
            code=$?
            if tail -n 100 "$BATCH/logs/$CURRENT_ARM.train.stderr.log" "$BATCH/logs/$CURRENT_ARM.train.stdout.log" \
                | grep -Eiq 'CUDA out of memory|torch\.OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED|CUDNN_STATUS_ALLOC_FAILED'; then
                control status failed "$CURRENT_ARM" 'CUDA allocation failure; configuration is frozen and automatic retry is disabled'
                exit "$code"
            fi
            if [[ ! -f "$run/last.pt" ]] || (( $(control retries "$CURRENT_ARM") >= 1 )); then
                control status failed "$CURRENT_ARM" 'Training failed without a retryable checkpoint or exhausted its single automatic retry'
                exit "$code"
            fi
            control status retrying "$CURRENT_ARM" 'Resuming once from the last committed epoch with the same configuration'
            "$PYTHON" scripts/diffusion_seg.py train --config "$config" --run "$run" --resume \
                >>"$BATCH/logs/$CURRENT_ARM.train.stdout.log" 2>>"$BATCH/logs/$CURRENT_ARM.train.stderr.log"
        fi
    fi
    [[ "$(control probe "$CURRENT_ARM")" == 'trained' ]] || {
        control status failed "$CURRENT_ARM" 'Training command returned without a complete committed checkpoint'; exit 1;
    }
    control status evaluating "$CURRENT_ARM" 'Evaluating the selected best checkpoint on the full validation split'
    "$PYTHON" scripts/diffusion_seg.py evaluate --config "$config" --run "$run" \
        >"$BATCH/evaluations/$CURRENT_ARM.json.pending" 2>>"$BATCH/logs/$CURRENT_ARM.evaluate.stderr.log"
    control evaluation "$CURRENT_ARM"
    control status arm_completed "$CURRENT_ARM" 'Training and full best-checkpoint validation completed'
done
control status completed '' 'All four experiments and best-checkpoint evaluations completed'
