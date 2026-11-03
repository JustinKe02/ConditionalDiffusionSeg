import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
import sys,json,hashlib,fcntl,shutil
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from conditional_diffusion.material_conditions import prepare_conditions
from conditional_diffusion.runtime import directory_size
B=R/'outputs/launches/p1_material_categorical_20261020'
lock=(R/'outputs/.experiments.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
before=json.loads((B/'source_before.json').read_text())
for p,h in before['protected_history_sha256'].items():
 assert hashlib.sha256((R/p).read_bytes()).hexdigest()==h,p
assert shutil.disk_usage(R).free>15*1024**3 and directory_size(R)+6*1024**3<26*1024**3
c=json.loads((R/'configs/pilot_static_semantic_s2.json').read_text())
c.update(epochs=100,prior_checkpoint=str(R/'outputs/P1/direct/seed_42/p1_cosine_20261017/checkpoints/best.pt'))
prepare_conditions(c,B/'conditions')
