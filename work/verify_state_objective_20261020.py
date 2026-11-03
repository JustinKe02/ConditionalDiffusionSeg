#!/usr/bin/env python3
"""CPU-only P0/P1 contract and legacy-compatibility tests; JSON on stdout.

No effect-training run, historical checkpoint read, real-data inference, result
file write or GPU allocation is performed. The parent launcher preserves stdout.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import unittest

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tests'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--new-only', action='store_true')
    args = parser.parse_args()
    import torch
    from conditional_diffusion import static_experiment as shared
    from conditional_diffusion.static_semantic import StaticSemanticModel
    from conditional_diffusion.state_objective import StateObjectiveModel
    torch.set_num_threads(2)
    assert not torch.cuda.is_available()
    original_build, original_sampler = shared.build, shared.sample_cached
    paths = [ROOT / 'src/conditional_diffusion' / name for name in
             ['state_objective.py', 'static_semantic.py', 'static_experiment.py',
              'diffusion.py', 'objectives.py', 'recycling.py', 'runtime.py']]
    paths += [ROOT / 'work/state_objective_worker.py', ROOT / 'tests/test_state_objective.py', Path(__file__).resolve()]
    def hashes():
        return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    before = hashes()
    begin = time.monotonic()
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'),
                    pattern='test_state_objective.py' if args.new_only else 'test_*.py')
    result = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    legacy_bindings_intact = shared.build is original_build and shared.sample_cached is original_sampler
    reference = StaticSemanticModel('S2').state_dict()
    models = {}
    for variant, mapping in [('P0', 'tanh'), ('P1', 'simplex')]:
        model = StateObjectiveModel(mapping)
        state = model.state_dict()
        models[variant] = {'mapping': mapping, 'parameter_count': sum(p.numel() for p in model.parameters()),
                           'initial_state_exactly_equals_historical_S2': all(torch.equal(v, state[k]) for k, v in reference.items()),
                           'state_keys_exactly_equal': state.keys() == reference.keys(),
                           'additional_learned_parameters': 0, 'final_readout': 'raw_logits'}
    after = hashes()
    passed = result.wasSuccessful() and legacy_bindings_intact and before == after
    report = {'status': 'passed' if passed else 'failed', 'tests_run': result.testsRun,
              'failures': len(result.failures), 'errors': len(result.errors), 'skipped': len(result.skipped),
              'includes_60_legacy_tests': not args.new_only, 'seconds': time.monotonic()-begin,
              'cpu_threads': torch.get_num_threads(), 'cuda_available': torch.cuda.is_available(),
              'process_local_bindings_restored': legacy_bindings_intact, 'models': models,
              'sources_unchanged_during_verification': before == after, 'source_sha256': after,
              'scope': 'Tiny synthetic CPU32 P0/P1 unit and restore checks; no effect training, dataset score, historical artifact modification or GPU use.',
              'limitations': 'No causal conclusion about gradient alignment or segmentation quality; real FP32 throughput/VRAM and full-image evaluation remain separate preflight steps.'}
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
