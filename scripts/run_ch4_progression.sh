#!/usr/bin/env bash
# Continuous serial runner for Chapter 4 diffusion progression (S0 check -> S1..S9 -> report).
# Usage:
#   nohup bash scripts/run_ch4_progression.sh --expected-commit <40-hex-sha> \
#       > outputs/ch4_progression_20261108/logs/queue.stdout.log \
#       2> outputs/ch4_progression_20261108/logs/queue.stderr.log &
set -euo pipefail

export GIT_TERMINAL_PROMPT=0
export GIT_SSH_COMMAND="${GIT_SSH_COMMAND:-ssh -o BatchMode=yes -o ConnectTimeout=20}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON="${PYTHON:-/root/miniconda3/envs/diffusion/bin/python}"
OUT="outputs/ch4_progression_20261108"
EXPECTED_COMMIT=""
QUEUE="all"
SKIP_REMOTE_CHECK=0

while (($#)); do
    case "$1" in
        --python) PYTHON=$2; shift 2 ;;
        --out) OUT=$2; shift 2 ;;
        --expected-commit) EXPECTED_COMMIT=$2; shift 2 ;;
        --queue) QUEUE=$2; shift 2 ;;
        --skip-remote-check) SKIP_REMOTE_CHECK=1; shift 1 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

case "$QUEUE" in
    all|night|day|check-only) ;;
    *) echo "Invalid --queue '$QUEUE' (expected all|night|day|check-only)" >&2; exit 2 ;;
esac

cd -- "$ROOT"
if [[ -z "$EXPECTED_COMMIT" ]]; then
    EXPECTED_COMMIT=$(git rev-parse HEAD)
fi
if [[ ! "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
    echo "EXPECTED_COMMIT must be a 40-char git SHA: got '$EXPECTED_COMMIT'" >&2
    exit 2
fi
if [[ ! -x "$PYTHON" ]] && ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "Python executable not found: $PYTHON" >&2
    exit 2
fi

command -v flock >/dev/null
mkdir -p outputs
exec 9>outputs/.ch4-progression.lock
if ! flock -n 9; then
    echo "Another Chapter 4 progression queue owns outputs/.ch4-progression.lock" >&2
    exit 75
fi

OUT_ABS=$("$PYTHON" - "$ROOT" "$OUT" <<'PY'
from pathlib import Path
import sys
root, out = Path(sys.argv[1]).resolve(), (Path(sys.argv[1]) / sys.argv[2]).resolve()
if not out.is_relative_to(root / 'outputs') or out == root / 'outputs':
    raise SystemExit(f'--out must be a named directory under project/outputs: {out}')
(out / 'logs').mkdir(parents=True, exist_ok=True)
print(out)
PY
)

control() {
    "$PYTHON" - "$ROOT" "$OUT_ABS" "$EXPECTED_COMMIT" "$@" <<'PY'
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import tempfile

root, out, commit = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
action, *args = sys.argv[4:]
sys.path.insert(0, str(root / 'src'))

def bjt_now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def atomic_json(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=p.parent,
                                         prefix=p.name + '.', suffix='.tmp', delete=False) as f:
            tmp = Path(f.name)
            json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)

status_path = out / 'status.json'
status = read(status_path) if status_path.exists() else dict(
    status='initializing', expected_commit=commit, stages={}, completed_stages=[],
)

if action == 'status':
    overall, stage, msg = args
    if status.get('status') == 'failed' and overall == 'failed':
        raise SystemExit(0)
    status.update(
        status=overall, current_stage=stage or None, message=msg,
        expected_commit=commit, pid=os.getppid(),
        updated_at_bjt=bjt_now(), updated_at_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'),
    )
    if stage:
        rec = status['stages'].setdefault(stage, dict(retries=0))
        rec.update(status=overall, message=msg, updated_at_bjt=bjt_now())
        if overall == 'retrying':
            rec['retries'] += 1
        if overall == 'stage_completed' and stage not in status['completed_stages']:
            status['completed_stages'].append(stage)
    atomic_json(status_path, status)
elif action == 'probe_train':
    import torch
    from conditional_diffusion.diffusion_seg_experiment import provenance
    arm = args[0]
    plan = read(out / 'plan.json')
    cfg_path = Path(plan['configs'][arm]['path'])
    h = hashlib.sha256(cfg_path.read_bytes()).hexdigest()
    if h != plan['configs'][arm]['sha256']:
        raise SystemExit(f'Snapshot config hash mismatch for {arm}')
    cfg = read(cfg_path)
    if provenance(cfg, root) != plan['provenance']:
        raise SystemExit('Source or train/val provenance changed')
    run = Path(plan['runs'][arm])
    complete = False
    if (run / 'last.pt').exists():
        st = torch.load(run / 'last.pt', map_location='cpu', weights_only=False)
        if st['config'] != cfg or st['provenance'] != plan['provenance']:
            raise SystemExit(f'Checkpoint config/provenance mismatch for {arm}')
        if st['epoch'] > cfg['epochs']:
            raise SystemExit(f'Checkpoint epoch exceeds schedule for {arm}')
        if st['epoch'] == cfg['epochs'] and st['best'] is not None and (run / 'best.pt').exists():
            try:
                sel = torch.load(run / 'best.pt', map_location='cpu', weights_only=False)
                complete = (
                    sel['config'] == cfg and sel['provenance'] == plan['provenance']
                    and sel['epoch'] == st['best']['epoch']
                    and sel['model'].keys() == st['best']['model'].keys()
                    and all(torch.equal(v, sel['model'][k]) for k, v in st['best']['model'].items())
                )
            except (OSError, RuntimeError, EOFError, ValueError, KeyError, pickle.UnpicklingError):
                complete = False
    print('trained' if complete else 'pending')
elif action == 'retries':
    print(status['stages'].get(args[0], {}).get('retries', 0))
elif action == 'json_completed':
    p = Path(args[0])
    ok = False
    if p.exists():
        try:
            ok = read(p).get('status') == 'completed'
        except Exception:
            ok = False
    print('yes' if ok else 'no')
else:
    raise SystemExit(f'Unknown control action: {action}')
PY
}

CURRENT_STAGE=""
finish() {
    local code=$?
    if ((code != 0)); then
        control status failed "$CURRENT_STAGE" "Queue exited with code $code; inspect $OUT_ABS/logs/" || true
    fi
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

verify_source() {
    [[ "$(git rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || {
        echo "HEAD $(git rev-parse HEAD) != expected $EXPECTED_COMMIT" >&2; return 1;
    }
    git diff --quiet HEAD -- src scripts configs || {
        echo "Uncommitted modifications in src/, scripts/, or configs/." >&2; return 1;
    }
    [[ -z "$(git ls-files --others --exclude-standard -- src scripts configs)" ]] || {
        echo "Untracked files exist in src/, scripts/, or configs/." >&2; return 1;
    }
    if (( SKIP_REMOTE_CHECK == 0 )); then
        local remote_commit attempt
        for attempt in 1 2 3; do
            if remote_commit=$(git ls-remote --exit-code --heads origin refs/heads/feat/diffusion-seg | awk '{print $1}'); then
                break
            fi
            if ((attempt == 3)); then
                echo "Remote commit verification failed after 3 attempts (pass --skip-remote-check if offline)." >&2
                return 1
            fi
            sleep "$((attempt * 5))"
        done
        [[ "$remote_commit" == "$EXPECTED_COMMIT" ]] || {
            echo "origin/feat/diffusion-seg ($remote_commit) != $EXPECTED_COMMIT" >&2; return 1;
        }
    fi
}

check_gpu_idle() {
    local apps
    apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | awk '{$1=$1};1')
    if [[ -n "$apps" ]]; then
        echo "GPU is occupied by PIDs: $apps" >&2
        return 1
    fi
}

run_train_arm() {
    local stage=$1
    local arm=$2
    CURRENT_STAGE="$stage"
    verify_source
    local state
    state=$(control probe_train "$arm")
    if [[ "$state" == "trained" ]]; then
        control status stage_completed "$stage" "$arm training already completed"
        return 0
    fi
    check_gpu_idle
    local resume_flag=()
    local run_dir="$OUT_ABS/$arm"
    if [[ -d "$run_dir" ]] && [[ -f "$run_dir/last.pt" || -f "$run_dir/config.json" ]]; then
        resume_flag=(--resume)
    fi
    control status running "$stage" "Training $arm"
    local stdout_log="$OUT_ABS/logs/${stage}_${arm}.stdout.log"
    local stderr_log="$OUT_ABS/logs/${stage}_${arm}.stderr.log"
    if "$PYTHON" scripts/ch4_progression.py train --out "$OUT_ABS" --arm "$arm" "${resume_flag[@]}" \
        >>"$stdout_log" 2>>"$stderr_log"; then
        :
    else
        local rc=$?
        if tail -n 120 "$stderr_log" "$stdout_log" | grep -Eiq 'CUDA out of memory|torch\.OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED|CUDNN_STATUS_ALLOC_FAILED'; then
            control status failed "$stage" "CUDA OOM during $arm; automatic batch change/retry forbidden"
            return "$rc"
        fi
        if [[ ! -f "$run_dir/last.pt" ]] || (( $(control retries "$stage") >= 1 )); then
            control status failed "$stage" "$arm failed without retryable checkpoint or exceeded 1 retry"
            return "$rc"
        fi
        control status retrying "$stage" "Resuming $arm once from last.pt"
        "$PYTHON" scripts/ch4_progression.py train --out "$OUT_ABS" --arm "$arm" --resume \
            >>"$stdout_log" 2>>"$stderr_log"
    fi
    [[ "$(control probe_train "$arm")" == "trained" ]] || {
        control status failed "$stage" "$arm returned without complete verified checkpoints"; return 1;
    }
    control status stage_completed "$stage" "$arm training finished"
}

run_eval_once() {
    local stage=$1
    local result_name=$2
    shift 2
    CURRENT_STAGE="$stage"
    local target="$OUT_ABS/eval/$result_name"
    if [[ "$(control json_completed "$target")" == "yes" ]]; then
        control status stage_completed "$stage" "Evaluation $result_name already completed (skipped)"
        return 0
    fi
    verify_source
    check_gpu_idle
    control status running "$stage" "Running evaluation $result_name"
    "$PYTHON" scripts/ch4_progression.py evaluate --out "$OUT_ABS" --result-name "$result_name" "$@" \
        >>"$OUT_ABS/logs/${stage}_${result_name%.json}.stdout.log" \
        2>>"$OUT_ABS/logs/${stage}_${result_name%.json}.stderr.log"
    [[ "$(control json_completed "$target")" == "yes" ]] || {
        control status failed "$stage" "Evaluation $result_name missing or incomplete"; return 1;
    }
    control status stage_completed "$stage" "Evaluation $result_name completed"
}

run_diag_arm() {
    local stage=$1
    local arm=$2
    CURRENT_STAGE="$stage"
    local target="$OUT_ABS/diagnostics/state_gap_${arm}.json"
    if [[ "$(control json_completed "$target")" == "yes" ]]; then
        control status stage_completed "$stage" "Diagnosis $arm already completed"
        return 0
    fi
    verify_source
    check_gpu_idle
    control status running "$stage" "Diagnosing state gap for $arm"
    "$PYTHON" scripts/ch4_progression.py diagnose --out "$OUT_ABS" --arm "$arm" \
        >>"$OUT_ABS/logs/${stage}.stdout.log" 2>>"$OUT_ABS/logs/${stage}.stderr.log"
    [[ "$(control json_completed "$target")" == "yes" ]] || {
        control status failed "$stage" "Diagnosis $arm incomplete"; return 1;
    }
    control status stage_completed "$stage" "Diagnosis $arm completed"
}

# ---- Execution starts ----
control status initializing '' "Verifying committed source, storage, and plan"
verify_source
"$PYTHON" scripts/ch4_progression.py prepare --out "$OUT_ABS" --expected-commit "$EXPECTED_COMMIT" \
    >>"$OUT_ABS/logs/prepare.log" 2>&1

if [[ "$QUEUE" == "all" || "$QUEUE" == "night" || "$QUEUE" == "check-only" ]]; then
    CURRENT_STAGE="S0_check"
    if [[ ! -f "$OUT_ABS/engineering_check.json" ]]; then
        check_gpu_idle
        control status running "S0_check" "Running S0 engineering verification"
        "$PYTHON" scripts/ch4_progression.py check --out "$OUT_ABS" \
            >"$OUT_ABS/logs/S0_check.stdout.log" 2>"$OUT_ABS/logs/S0_check.stderr.log"
    fi
    control status stage_completed "S0_check" "Engineering checks passed"
    if [[ "$QUEUE" == "check-only" ]]; then
        CURRENT_STAGE=""
        control status completed '' "S0 prepare and check completed"
        exit 0
    fi

    CURRENT_STAGE="preflight_storage"
    "$PYTHON" scripts/check_storage.py --runs 2 --reserve-gib 8.0 >"$OUT_ABS/logs/storage_preflight.json"

    run_train_arm "S1_g_onehot" "g_onehot"
    run_train_arm "S6_g_ordinal" "g_ordinal"

    run_eval_once "S2_steps_g_onehot" "steps_val_g_onehot.json" --model g_onehot --split val --steps 1 2 4 10 20 50 --seed 3407
    run_eval_once "S2_steps_g_ordinal" "steps_val_g_ordinal.json" --model g_ordinal --split val --steps 1 2 4 10 20 50 --seed 3407
    run_eval_once "S7_steps_I" "steps_val_I.json" --model I --split val --steps 1 2 4 --seed 3407
    run_eval_once "S8_seed3408_g_onehot" "seed3408_g_onehot.json" --model g_onehot --split val --steps 20 --seed 3408
    run_eval_once "S8_seed3408_g_ordinal" "seed3408_g_ordinal.json" --model g_ordinal --split val --steps 20 --seed 3408

    CURRENT_STAGE="S3_register_test"
    verify_source
    control status running "S3_register_test" "Locking pre-registered test27 protocol in plan.json"
    "$PYTHON" scripts/ch4_progression.py register-test --out "$OUT_ABS" \
        >>"$OUT_ABS/logs/S3_register.log" 2>&1
    control status stage_completed "S3_register_test" "test_registration locked"

    run_eval_once "S3_test_g_onehot_k20" "test27_g_onehot_k20.json" --model g_onehot --split test --steps 20 --seed 3407
    run_eval_once "S3_test_g_onehot_k4" "test27_g_onehot_k4.json" --model g_onehot --split test --steps 4 --seed 3407
    run_eval_once "S3_test_g_ordinal_k20" "test27_g_ordinal_k20.json" --model g_ordinal --split test --steps 20 --seed 3407
    run_eval_once "S3_test_g_ordinal_k4" "test27_g_ordinal_k4.json" --model g_ordinal --split test --steps 4 --seed 3407
    run_eval_once "S3_test_I_k4" "test27_I_k4.json" --model I --split test --steps 4 --seed 3407

    CURRENT_STAGE="S3_bootstrap"
    control status running "S3_bootstrap" "Computing 10,000 paired bootstrap intervals & LOO sensitivity"
    "$PYTHON" scripts/ch4_progression.py bootstrap --out "$OUT_ABS" \
        >>"$OUT_ABS/logs/bootstrap.log" 2>&1
    control status stage_completed "S3_bootstrap" "Paired bootstrap & LOO completed"
fi

if [[ "$QUEUE" == "all" || "$QUEUE" == "day" ]]; then
    run_diag_arm "S9_diag_g_onehot" "g_onehot"
    run_diag_arm "S9_diag_g_ordinal" "g_ordinal"
    run_diag_arm "S9_diag_I" "I"
    run_diag_arm "S9_diag_R" "R"

    CURRENT_STAGE="S4_bench_infer"
    if [[ ! -f "$OUT_ABS/benchmark/inference.json" ]]; then
        verify_source
        check_gpu_idle
        control status running "S4_bench_infer" "Running single-session unified inference benchmark (13 objects)"
        "$PYTHON" scripts/ch4_progression.py bench-infer --out "$OUT_ABS" \
            >"$OUT_ABS/logs/S4_bench_infer.stdout.log" 2>"$OUT_ABS/logs/S4_bench_infer.stderr.log"
    fi
    control status stage_completed "S4_bench_infer" "Unified inference benchmark completed"

    CURRENT_STAGE="S5_bench_train"
    if [[ ! -f "$OUT_ABS/benchmark/training.json" ]]; then
        verify_source
        check_gpu_idle
        control status running "S5_bench_train" "Measuring training step time & VRAM across 7 objects"
        "$PYTHON" scripts/ch4_progression.py bench-train --out "$OUT_ABS" \
            >"$OUT_ABS/logs/S5_bench_train.stdout.log" 2>"$OUT_ABS/logs/S5_bench_train.stderr.log"
    fi
    control status stage_completed "S5_bench_train" "Training resource measurement completed"

    CURRENT_STAGE="S11_report"
    control status running "S11_report" "Generating Chapter 4/5 tables, figures, and §13 narrative verdict"
    "$PYTHON" scripts/ch4_progression.py report --out "$OUT_ABS" \
        >"$OUT_ABS/logs/S11_report.stdout.log" 2>"$OUT_ABS/logs/S11_report.stderr.log"
    control status stage_completed "S11_report" "Tables, figures, and narrative verdict generated"
fi

CURRENT_STAGE=""
control status completed '' "Chapter 4 progression queue ($QUEUE) completed cleanly"
