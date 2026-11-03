import os,sys,json,time,argparse
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
ROOT=Path.cwd();sys.path.insert(0,str(ROOT/'src'))
from conditional_diffusion.trainer import train

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    c=json.loads((ROOT/'configs/pilot_high_noise.json').read_text())
    c.update(base_channels=8,crop_size=32,stride=32,workers=0,preview_count=0,log_every=1)
    stamp=str(time.time_ns());base=ROOT/'outputs/integration'/('p1_high_noise_resume_'+stamp)
    runs=[]
    for name,stop in [('full',2),('resume',1)]:
        run=base/name;run.mkdir(parents=True);runs.append(run)
        train(dict(c),ROOT,run,stop_after_epoch=stop)
    train(dict(c),ROOT,runs[1],resume=runs[1]/'checkpoints/last.pt',stop_after_epoch=2)
    a,b=[torch.load(run/'checkpoints/last.pt',map_location='cpu',weights_only=False) for run in runs]
    assert a['epoch']==b['epoch']==2 and a['global_step']==b['global_step']==120
    assert all(torch.equal(v,b['model'][k]) for k,v in a['model'].items())
    for key,vals in a['optimizer']['state'].items():
        for name,v in vals.items():assert torch.equal(v,b['optimizer']['state'][key][name])
    assert a['scheduler']==b['scheduler']
    args.output.write_text(json.dumps(dict(status='passed',model_weights_bitwise_equal=True,optimizer_bitwise_equal=True,scheduler_equal=True,global_step=120,
       scope='Registered P1 high-noise path, actual full120 train split, base8/crop32 plumbing check, controlled stop at2 of40 epochs, no validation/test',runs=list(map(str,runs))),indent=2))
    print('P1_HIGH_NOISE_RESUME_PASSED',flush=True)

if __name__=='__main__':main()
