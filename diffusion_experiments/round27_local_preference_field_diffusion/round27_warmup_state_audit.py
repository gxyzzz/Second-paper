from __future__ import annotations
import argparse, hashlib, json, gc
from pathlib import Path
import numpy as np
import torch
from diffusion_experiments.round27_local_preference_field_diffusion import round27_core as c

R=c.RDIR; E=c.EVID; A=c.ASSETS
SEED=999

def sd_hash(sd):
    h=hashlib.sha256()
    for k in sorted(sd):
        v=sd[k].detach().cpu().contiguous(); h.update(k.encode()); h.update(v.numpy().tobytes())
    return h.hexdigest()

def run(tag):
    model,config,td,vd,pack=c.instantiate_shared(SEED)
    wd=float(config['weight_decay']) if config['weight_decay'] is not None else 0.
    opt=torch.optim.Adam(model.parameters(),lr=float(config['learning_rate']),weight_decay=wd)
    fac=config['learning_rate_scheduler']; sched=torch.optim.lr_scheduler.LambdaLR(opt,lr_lambda=lambda e:fac[0]**(e/fac[1]))
    plans=[]; losses=[]
    for ep in range(10):
        hh=hashlib.sha256(); el=[]; model.train()
        for interaction in td:
            for x in interaction: hh.update(x.detach().cpu().numpy().tobytes())
            opt.zero_grad(set_to_none=True); b=c.forward_bundle(model); loss=c.msca_loss_from_bundle(model,interaction,b); loss.backward()
            if config['clip_grad_norm']: torch.nn.utils.clip_grad_norm_(model.parameters(),**config['clip_grad_norm'])
            opt.step(); el.append(float(loss.detach()))
        sched.step(); plans.append(hh.hexdigest()); losses.append(float(np.mean(el)))
        print(json.dumps({'tag':tag,'epoch':ep,'loss':losses[-1]}),flush=True)
    sd={k:v.detach().cpu() for k,v in model.state_dict().items()}; path=A/f'ROUND27_WARMUP_STATE_{tag}.pt'; torch.save({'state_dict':sd,'state_hash':sd_hash(sd),'normal_plan_hashes':plans,'losses':losses},path)
    print(json.dumps({'tag':tag,'state_hash':sd_hash(sd),'path':str(path)}),flush=True)
    del model,opt; torch.cuda.empty_cache(); gc.collect()

def audit():
    pa=torch.load(A/'ROUND27_WARMUP_STATE_A.pt',map_location='cpu',weights_only=False); pb=torch.load(A/'ROUND27_WARMUP_STATE_B.pt',map_location='cpu',weights_only=False)
    diffs=[]; rel=[]
    for k in sorted(pa['state_dict']):
        a=pa['state_dict'][k].double(); b=pb['state_dict'][k].double(); d=(a-b).abs(); diffs.append(float(d.max())); rel.append(float(d.max()/(a.abs().max()+1e-30)))
    model,config,td,vd,pack=c.instantiate_shared(SEED); events=c.TrainEvents(model); mask=c.valid_mask(td,model.n_items,model.device); interaction=next(iter(td)); tiny=[x[:256] for x in interaction]
    def scores(state):
        model.load_state_dict(state,strict=True); model.eval()
        with torch.no_grad():
            b=c.forward_bundle(model); s=(b['fu'][tiny[0]].double() @ b['fi'].double().T); s[:,~mask]=-torch.inf
            for r,(u,p) in enumerate(zip(tiny[0].cpu().tolist(),tiny[1].cpu().tolist())):
                seen=events.histories[int(u)]
                if seen: s[r,torch.as_tensor(seen,device=s.device,dtype=torch.long)]=-torch.inf
                s[r,int(p)]=-torch.inf
            vals,ids=torch.sort(s,dim=1,descending=True,stable=True)
            return s.cpu(),vals[:,:101].cpu(),ids[:,:101].cpu()
    sa,va,ia=scores(pa['state_dict']); sb,vb,ib=scores(pb['state_dict'])
    finite=torch.isfinite(sa)&torch.isfinite(sb); cross=(sa[finite]-sb[finite]).abs().numpy()
    gaps_a=(va[:,:100]-va[:,1:101]).numpy().reshape(-1); gaps_b=(vb[:,:100]-vb[:,1:101]).numpy().reshape(-1)
    top100_overlap=np.mean([len(set(ia[r,:100].tolist())&set(ib[r,:100].tolist()))/100. for r in range(len(ia))])
    out={'protocol':c.PROTOCOL,'seed':SEED,'deterministic_algorithms':True,'normal_plan_exact':pa['normal_plan_hashes']==pb['normal_plan_hashes'],'warmup_state_hash_exact':pa['state_hash']==pb['state_hash'],'warmup_state_hash_A':pa['state_hash'],'warmup_state_hash_B':pb['state_hash'],'max_parameter_abs_diff':float(max(diffs)),'max_parameter_relative_diff':float(max(rel)),'score_cross_run_abs_diff':c.stats(cross),'rank_adjacent_gap_A':c.stats(gaps_a),'rank_adjacent_gap_B':c.stats(gaps_b),'top100_set_overlap_mean':float(top100_overlap),'top100_order_exact_fraction':float(np.mean(np.all(ia[:,:100].numpy()==ib[:,:100].numpy(),axis=1))),'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
    c.json_write(E/'ROUND27_WARMUP_STATE_AUDIT.json',out); print(json.dumps(out,indent=2))
    del model; torch.cuda.empty_cache(); gc.collect()

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('cmd',choices=['A','B','audit']); z=ap.parse_args(); audit() if z.cmd=='audit' else run(z.cmd)
