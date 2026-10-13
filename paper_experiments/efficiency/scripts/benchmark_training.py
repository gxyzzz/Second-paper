from __future__ import annotations

import argparse, csv, json, statistics, sys, time
from itertools import cycle
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,str(ROOT))
import pipelines.preference_pipeline as pp
from modules.preference_diffusion import DiffusionSchedule, PreferenceDiffusion, curriculum_ids, direction_inner_step, latent_stats, mine_current_ids, predict_directions, weighted_bpr
from paper_experiments.pdbr_ablation.scripts.run_pdbr_ablation import _prepare_shared_init

WARMUP=20; TIMED=100

def sync(device): torch.cuda.synchronize(device)

def batch_to_device(batch,device): return [x.to(device) for x in batch]

def run_baseline(dataset,gpu_id,run_dir):
    _prepare_shared_init(dataset,gpu_id,run_dir)
    model,config,_,loader,_=pp.instantiate_shared(dataset,gpu_id,run_dir)
    opt=torch.optim.Adam(model.parameters(),lr=float(config['learning_rate']))
    device=model.device; it=cycle(loader)
    def step():
        inter=batch_to_device(next(it),device); model.train(); opt.zero_grad(set_to_none=True)
        loss=model.calculate_loss(inter); loss.backward(); opt.step(); return int(inter[0].shape[0])
    for _ in range(WARMUP): step()
    sync(device); torch.cuda.reset_peak_memory_stats(device); total_samples=0; t0=time.perf_counter()
    for _ in range(TIMED): total_samples+=step()
    sync(device); dt=time.perf_counter()-t0
    return {'ms_per_batch':dt/TIMED*1000,'samples_per_second':total_samples/dt,
            'peak_cuda_allocated_GB':torch.cuda.max_memory_allocated(device)/1024**3,
            'peak_cuda_reserved_GB':torch.cuda.max_memory_reserved(device)/1024**3,
            'timed_batches':TIMED,'warmup_batches':WARMUP,'batch_size_config':int(config['train_batch_size'])}

def run_pdbr(dataset,gpu_id,run_dir):
    _prepare_shared_init(dataset,gpu_id,run_dir)
    model,config,_,loader,_=pp.instantiate_shared(dataset,gpu_id,run_dir)
    device=model.device; bundle=pp.build_bundle(dataset,model); latent=latent_stats(model)
    net=PreferenceDiffusion().to(device); schedule=DiffusionSchedule(device=device)
    opt=torch.optim.Adam(model.parameters(),lr=float(config['learning_rate'])); dopt=torch.optim.Adam(net.parameters(),lr=1e-3)
    it=cycle(loader); step_index=0
    def step():
        nonlocal step_index
        inter=batch_to_device(next(it),device); batch=int(inter[0].shape[0]); model.train(); opt.zero_grad(set_to_none=True)
        normal=model.calculate_loss(inter)
        ids=mine_current_ids(model,inter,latent,bundle.all_items,curriculum_ids(3),pp.rank_seed(1000,30,step_index))
        for inner in range(2):
            direction_inner_step(net,dopt,schedule,bundle,inter,latent,ids,pp.local_seed(1000,30,step_index,inner),'formal')
        dirs,_,_=predict_directions(net,schedule,bundle,inter,latent,ids,pp.local_seed(1000,30,step_index,100),'formal')
        deltas,_=pp._calibrate_bands(bundle,inter,ids,dirs,.25,5.0)
        fu=bundle['fu'][inter[0]]; pos=bundle['fi'][inter[1]]; negs=[bundle['fi'][ids[j]]+deltas[j] for j in range(3)]
        aux=weighted_bpr(fu,pos,negs); loss=normal+.20*aux
        loss.backward(); opt.step(); step_index+=1; return batch
    for _ in range(WARMUP): step()
    sync(device); torch.cuda.reset_peak_memory_stats(device); total_samples=0; t0=time.perf_counter()
    for _ in range(TIMED): total_samples+=step()
    sync(device); dt=time.perf_counter()-t0
    return {'ms_per_batch':dt/TIMED*1000,'samples_per_second':total_samples/dt,
            'peak_cuda_allocated_GB':torch.cuda.max_memory_allocated(device)/1024**3,
            'peak_cuda_reserved_GB':torch.cuda.max_memory_reserved(device)/1024**3,
            'timed_batches':TIMED,'warmup_batches':WARMUP,'batch_size_config':int(config['train_batch_size']),
            'benchmark_mode':'full-active upper-bound','active_boundary_bands':3,'active_steps':2}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset',choices=['baby','sports','elec','all'],default='all'); ap.add_argument('--gpu-id',type=int,default=0); a=ap.parse_args()
    ds=['baby','sports','elec'] if a.dataset=='all' else [a.dataset]; rows=[]
    for d in ds:
        print('[Training benchmark]',d,flush=True); base=run_baseline(d,a.gpu_id,ROOT/f'paper_experiments/efficiency/outputs/{d}/baseline'); full=run_pdbr(d,a.gpu_id,ROOT/f'paper_experiments/efficiency/outputs/{d}/pdbr')
        for method,res in [('MSCA',base),('MSCA+CoLiftRec',dict(base)),('MSCA+CoLiftRec+PDBR',full)]:
            res=dict(res); res.update({'dataset':d,'method':method}); rows.append(res)
        full['time_overhead_percent']=(full['ms_per_batch']/base['ms_per_batch']-1)*100
        full['memory_overhead_percent']=(full['peak_cuda_allocated_GB']/base['peak_cuda_allocated_GB']-1)*100
    ev=ROOT/'paper_experiments/efficiency/evidence'; ev.mkdir(parents=True,exist_ok=True)
    fields=['dataset','method','ms_per_batch','samples_per_second','peak_cuda_allocated_GB','peak_cuda_reserved_GB','timed_batches','warmup_batches','batch_size_config','benchmark_mode','active_boundary_bands','active_steps']
    with (ev/'TRAINING_BENCHMARK.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writeheader(); w.writerows(rows)
    overhead=[]
    for d in ds:
        b=next(r for r in rows if r['dataset']==d and r['method']=='MSCA'); p=next(r for r in rows if r['dataset']==d and r['method'].endswith('PDBR'))
        overhead.append({'dataset':d,'time_overhead_percent':(p['ms_per_batch']/b['ms_per_batch']-1)*100,'memory_overhead_percent':(p['peak_cuda_allocated_GB']/b['peak_cuda_allocated_GB']-1)*100})
    (ev/'TRAINING_OVERHEAD.json').write_text(json.dumps(overhead,indent=2)+'\n')
    print('wrote',ev/'TRAINING_BENCHMARK.csv')
if __name__=='__main__': main()
