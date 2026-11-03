"""Small base8 fixed-crop integration check of stratified training and resume."""
import os,sys,json,argparse,time,subprocess
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
ROOT=Path.cwd()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
 c=json.loads((ROOT/'configs/p0_high_noise.json').read_text());c.update(base_channels=8,epochs=2,val_every=2,log_every=1,preview_count=0)
 stamp=str(time.time_ns());config=ROOT/'work'/('resume_check_'+stamp+'.json');config.write_text(json.dumps(c))
 runs=[]
 for label,extra in [('full',[]),('resume',['--stop-after-epoch','1'])]:
  ident='stratified_debug_'+label+'_'+stamp;run=ROOT/'outputs/P0/diffusion/seed_42'/ident;runs.append(run)
  subprocess.run([sys.executable,'scripts/train.py','--config',str(config),'--run-id',ident,*extra],check=True)
 subprocess.run([sys.executable,'scripts/train.py','--resume',str(runs[1]/'checkpoints/last.pt')],check=True)
 a,b=[torch.load(run/'checkpoints/last.pt',map_location='cpu',weights_only=False) for run in runs]
 assert a['global_step']==b['global_step']==4
 assert all(torch.equal(v,b['model'][k]) for k,v in a['model'].items())
 for k,vals in a['optimizer']['state'].items():
  for key,v in vals.items():assert torch.equal(v,b['optimizer']['state'][k][key])
 assert a['scheduler']==b['scheduler']
 args.output.write_text(json.dumps(dict(status='passed',weights_bitwise_equal=True,optimizer_bitwise_equal=True,scheduler_equal=True,
  scope='base8, 2 epochs, fixed 256 crops, stratified P0 plumbing only',runs=list(map(str,runs))),indent=2))
 print('HIGH_NOISE_RESUME_CHECK_PASSED',flush=True)
if __name__=='__main__':main()
