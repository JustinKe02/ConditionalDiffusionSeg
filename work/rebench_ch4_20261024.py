"""Re-run the full chapter-4 benchmark (17 objects) in one batch into a new output directory.

The 2026-10-24 benchmark was measured in two batches whose whole-image fusion times differ ~2.5x
(host CPU contention), so all objects are re-timed together. Reuses the frozen benchmark code and the
patched FLOP counter of complete_ch4_20261024.py; records host load average around every object.
The old outputs/analysis/ch4_benchmark_20261024 is left untouched.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('CH4_BENCH_OUT', '/root/autodl-tmp/ConditionalDiffusionSeg/outputs/analysis/ch4_benchmark_rerun_20261024')
import sys, json, fcntl, subprocess
from pathlib import Path
from datetime import datetime, timezone, timedelta
ROOT = Path('/root/autodl-tmp/ConditionalDiffusionSeg')
sys.path[:0] = [str(ROOT / 'work'), str(ROOT / 'src')]
import complete_ch4_20261024 as C  # patches B.flops_of
B = C.B
B.__file__ = str(Path(__file__).resolve())
OUT = B.OUT
LOADS = OUT / 'host_load.json'


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')


def load():
    return dict(time_bjt=now(), loadavg=list(os.getloadavg()), cpu_count=os.cpu_count())


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--one':
        B.one(sys.argv[2])
        sys.exit(0)
    assert OUT != ROOT / 'outputs/analysis/ch4_benchmark_20261024'
    lock = open(ROOT / 'outputs/.experiments.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    OUT.mkdir(parents=True, exist_ok=True)
    meta = dict(started_bjt=now(), counter_checks=C.verify_counter(), runner_sha256=B.G.digest(Path(__file__)),
                benchmark_sha256=B.G.digest(ROOT / 'work/benchmark_ch4_20261024.py'),
                completion_sha256=B.G.digest(ROOT / 'work/complete_ch4_20261024.py'),
                reason='single-batch re-timing; previous run split into two batches with different host CPU load',
                loads={})
    if LOADS.exists():
        meta['loads'] = json.loads(LOADS.read_text())['loads']
    for name in B.OBJECTS:
        if (OUT / 'objects' / f'{name}.json').exists():
            continue
        apps = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                              capture_output=True, text=True, check=True).stdout.split()
        assert not apps, f'GPU is busy before benchmark {name}: {apps}'
        before = load()
        with open(OUT / f'log_{name}.txt', 'a') as log:
            rc = subprocess.call([sys.executable, '-B', '-W', 'ignore', __file__, '--one', name],
                                 stdout=log, stderr=subprocess.STDOUT)
        meta['loads'][name] = dict(before=before, after=load(), returncode=rc)
        B.G.atomic_json(LOADS, meta)
        if rc != 0:
            raise SystemExit(f'benchmark failed for {name} (rc={rc}); see log_{name}.txt')
        B.G.atomic_json(LOADS, meta)
    B.main()  # all objects present: writes benchmark.json/csv/README for the full set
    meta['finished_bjt'] = now()
    B.G.atomic_json(LOADS, meta)
    print('REBENCH DONE', flush=True)
