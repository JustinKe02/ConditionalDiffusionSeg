"""Read-only disk preflight for the planned experiments; does not start training."""
import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

GIB = 1024**3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=int, default=24, help='Number of additional planned runs')
    parser.add_argument('--parameters', type=int, default=20_000_000,
                        help='Budget assumption, not a measured model parameter count')
    parser.add_argument('--reserve-gib', type=float, default=8.0)
    args = parser.parse_args()
    if args.runs <= 0 or args.parameters <= 0 or args.reserve_gib < 0:
        parser.error('runs/parameters must be positive and reserve must be nonnegative')
    root = Path(__file__).resolve().parents[1]
    disk = shutil.disk_usage(root)
    project_bytes = 0
    seen = set()
    for path in [root, *root.rglob('*')]:
        if path.is_symlink():
            continue
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino)
        if identity not in seen:
            project_bytes += stat.st_blocks * 512
            seen.add(identity)
    # FP32 best weights (4P) + rolling AdamW checkpoint (weights + two moments, 12P).
    # Optimizer scalar state, serialization headers, buffers, etc. use metadata/shared allowance.
    per_run = 16 * args.parameters + (.25 + .25 + .03125) * GIB
    persistent = args.runs * per_run + 2 * GIB
    atomic_temp = 12 * args.parameters
    required = persistent + atomic_temp + args.reserve_gib * GIB
    cap = 26 * GIB
    disk_ok = disk.free >= required
    cap_ok = project_bytes + persistent + atomic_temp <= cap
    result = {
        'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'project_root': str(root),
        'measurement': {'disk_total_gib': disk.total / GIB, 'disk_used_gib': disk.used / GIB,
                        'disk_free_gib': disk.free / GIB, 'project_allocated_gib': project_bytes / GIB},
        'assumptions': {'additional_runs': args.runs, 'parameters_per_model': args.parameters,
                        'parameter_count_measured': False, 'tensorboard_gib_per_run': .25,
                        'prediction_gib_per_run': .25, 'metadata_gib_per_run': .03125,
                        'shared_debug_reports_and_misc_gib': 2, 'reserve_gib': args.reserve_gib,
                        'project_cap_gib': 26},
        'estimate': {'per_run_gib': per_run / GIB, 'additional_persistent_gib': persistent / GIB,
                     'atomic_write_extra_gib': atomic_temp / GIB,
                     'required_free_including_reserve_gib': required / GIB},
        'capacity_ok_under_assumptions': disk_ok and cap_ok,
        'disk_budget_ok': disk_ok, 'project_cap_ok': cap_ok,
        'limitations': ['Estimate uses a configurable parameter assumption and size formulas, not measured sizes of all future artifacts.',
                       'Single-run runtime checks are implemented in trainer; this script only estimates total planned capacity.',
                       'Concurrent disk writers and extra reruns can invalidate this snapshot.'],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if disk_ok and cap_ok else 2


if __name__ == '__main__':
    raise SystemExit(main())
