"""Native FP32 benchmark, fixed K4 conditions and true early return at1/2/4."""
import sys,time,json,socket
from pathlib import Path
import numpy as np
import torch
import core as C
G=C.G

def measure(fn,n,warmup):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); times=[]
    for _ in range(n):
        t=time.perf_counter(); fn(); torch.cuda.synchronize(); times.append(time.perf_counter()-t)
    return np.asarray(times)

@torch.no_grad()
def main(a,stop):
    torch.set_num_threads(20)
    model,c,meta=C.load_best(a); ds=G.GLPDataset(c,'val',train=False)
    plan=json.loads((C.BATCH/'execution_plan.json').read_text())
    assert ds.ids[0]==plan['benchmark_warmup_image'] and ds.ids[1:]==plan['benchmark_timed_ids']
    d0=ds[0]
    ref,_=C.predict(model,c,d0['image'],d0['thumb'],d0['id'],stop_after=stop)
    torch.use_deterministic_algorithms(False); torch.backends.cudnn.deterministic=False; torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    pred,_=C.predict(model,c,d0['image'],d0['thumb'],d0['id'],stop_after=stop)
    agreement=float((pred==ref).mean()); assert agreement>=.9999,agreement
    ctx=model.global_context([d0['thumb'].cuda()])
    image=d0['image'][:,:512,:512][None].cuda()
    boxes=torch.tensor([[0.,0.,512/1920,512/2560]])
    eps=torch.randn((1,4,512,512),device='cuda')
    window=lambda:C.infer_windows(model,c,image,boxes,ctx,eps,stop_after=stop)
    lat=measure(window,200,50)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    total=[]; net=[]; ids=[]
    for i in range(len(ds)):
        d=ds[i]
        torch.cuda.synchronize(); t=time.perf_counter()
        C.predict(model,c,d['image'],d['thumb'],d['id'],stop_after=stop)
        torch.cuda.synchronize(); elapsed=time.perf_counter()-t
        timing={}
        C.predict(model,c,d['image'],d['thumb'],d['id'],stop_after=stop,timing=timing)
        if i:
            ids.append(d['id']); total.append(elapsed); net.append(timing['network_seconds'])
        print('BENCHMARK',a,stop,i+1,len(ds),round(elapsed,3),flush=True)
    result=dict(meta,stop_after=stop,grid_K=4,level=1,params=sum(p.numel() for p in model.parameters()),
                environment=dict(host=socket.gethostname(),gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
                                 cpu_threads=20,precision='FP32 TF32 disabled',deterministic=False,cudnn_benchmark=True),
                window_warmup=50,window_iterations=200,window_latency_ms_median=float(np.median(lat)*1000),
                window_latency_ms_p95=float(np.percentile(lat,95)*1000),agreement_with_as_evaluated=agreement,
                image_predict_s_median=float(np.median(total)),image_network_s_median=float(np.median(net)),
                image_fusion_and_transfer_s_median=float(np.median(total)-np.median(net)),
                images_timed=len(ids),image_ids=ids,raw_total_seconds=total,raw_network_seconds=net,
                infer_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                infer_peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                timing_scope='Total: separate uninstrumented complete prediction. Network: second pass synchronized global/window calls, excludes tensor assembly/transfer/fusion. Difference of medians is an estimate.',
                method_scope='Early return along fixed K4 trajectory, not independent K1/K2 models')
    G.atomic_json(C.ANALYSIS/'benchmark'/f'{a}_stop{stop}.json',result)

if __name__=='__main__': main(sys.argv[sys.argv.index('--arm')+1],int(sys.argv[sys.argv.index('--stop')+1]))
