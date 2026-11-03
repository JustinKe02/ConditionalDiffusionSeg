"""One authorized batch: wait for the pinned preflight, launch A/B, then exit."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import hashlib
import json
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from conditional_diffusion.runtime import source_hashes, write_json

BATCH = ROOT / 'outputs/launches/p1_reference_adapter_20261018'


def read(path):
    return json.loads(Path(path).read_text())


def alive(pid):
    try:
        proc = Path(f'/proc/{pid}')
        return 'verify_reference_adapter.py' in (proc / 'cmdline').read_text() and ') Z ' not in (proc / 'stat').read_text()
    except FileNotFoundError:
        return False


def main():
    request = read(BATCH / 'queued_request.json')
    pid = request['preflight_pid']
    count = -1
    stage = 'waiting_for_preflight'
    try:
        while alive(pid):
            progress_path = BATCH / 'fallback_verification_partial.json'
            current = len(read(progress_path)['results']) if progress_path.exists() else 0
            if current != count:
                count = current
                write_json(BATCH / 'status.json', dict(status='preflight_running', stage='native_rejected_fallback_audit',
                           pid=os.getpid(), preflight_pid=pid, verified_images=count, expected_image_seed_pairs=20,
                           training_started=False, updated_at_utc=datetime.now(timezone.utc).isoformat()))
                print(f'PREFLIGHT_PROGRESS {count}/20; A/B queued after complete checks', flush=True)
            time.sleep(2)
        assert read(BATCH / 'preflight_completed.json')['status'] == 'passed', 'Preflight did not complete successfully; inspect preflight_attempt1.log'
        assert source_hashes(ROOT) == request['source_sha256'], 'Source/config changed during queued preflight'
        for name, digest in request['workflow_sha256'].items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
        stage = 'launching_training'
        write_json(BATCH / 'status.json', dict(status='starting_training', stage=stage, pid=os.getpid(),
                   training_started=False, updated_at_utc=datetime.now(timezone.utc).isoformat()))
        print('PREFLIGHT_COMPLETE; starting exactly A then B', flush=True)
        subprocess.run([sys.executable, '-u', 'work/launch_p1_reference_adapter.py'], cwd=ROOT, check=True)
    except BaseException as exc:
        previous = read(BATCH / 'status.json') if (BATCH / 'status.json').exists() else {}
        if previous.get('status') != 'failed':
            write_json(BATCH / 'status.json', dict(status='failed', stage=stage, pid=os.getpid(), error=str(exc),
                       updated_at_utc=datetime.now(timezone.utc).isoformat()))
        traceback.print_exc(); raise


if __name__ == '__main__':
    main()
