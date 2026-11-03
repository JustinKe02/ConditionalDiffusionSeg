import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from conditional_diffusion.data import audit_dataset
from conditional_diffusion.runtime import write_json

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=ROOT/'data_manifests/full_audit.json')
    args = p.parse_args()
    result = audit_dataset(ROOT/'data_manifests/source_dataset.json')
    write_json(args.output, result)
    print(result)
