from __future__ import annotations
import gc,json
import numpy as np
import torch
from diffusion_experiments.round25_catalog_anchored_diffusion import round25_core as c
from diffusion_experiments.round25_catalog_anchored_diffusion.round25_preflight_warmup import run as warmup

def run():
    grad=json.loads((c.EVID/'ROUND25_GRADIENT_AUDIT.json').read_text())
    model,net,dopt,sched,events,td,warm=warmup(); mask=c.valid_mask(td,model.n_items,model.device)
    u,p=events.sample(1024,20286300); n=events.random_unobserved(u,p,20286301); dev=model.device
    inter=[torch.as_tensor(u,device=dev),torch.as_tensor(p,device=dev),torch.as_tensor(n,device=dev)]
    s=c.latent_stats(model); b=c.forward_bundle(model); ids,_,_=c.mine_current_ids(b,inter,events,mask,999,10,0,1)
    rng=np.random.default_rng(20286302); losses=[]
    for step in range(150):
        ix=rng.choice(len(u),size=64,replace=False); ti=torch.as_tensor(ix,device=dev)
        it=[x[ti] for x in inter]; jj=[ids[0][ti]]
        losses.append(c.diffusion_inner_step(net,dopt,sched,b,it,s,jj,202864000+step))
    batches=[]
    for st in range(0,len(u),128):
        en=min(st+128,len(u)); it=[x[st:en] for x in inter]; jj=[ids[0][st:en]]
        _,diag=c.generate_refinement(net,sched,b,it,s,jj,202865000+st); batches.append(diag)
    mech=c.summarize_mechanism(batches,losses); reasons=[]
    if not grad['PASS']: reasons.append('GRADIENT_TOPOLOGY_WRONG')
    if not mech or not np.isfinite(mech['L_total']['mean']): reasons.append('NONFINITE_DIFFUSION_LOSS')
    if not mech.get('angle_bound_pass',False): reasons.append('ANGLE_BOUND_BROKEN')
    if mech and mech['true_vs_shuffled_residual_l2']['max']==0: reasons.append('TRUE_BASE_SHUF_BRANCH_IMPLEMENTATION_COLLAPSE')
    if grad['D2']['selected_real_negative_grad_norm']<=0: reasons.append('REAL_NEGATIVE_GRAD_ZERO')
    out={'protocol':c.PROTOCOL,'seed':999,'TRAIN_ONLY':True,'events':1024,'online_diffusion_updates':150,'warmup':warm,'mechanism':mech,'gradient_audit_pass':grad['PASS'],'PREFLIGHT_PASS':not reasons,'BLOCK_REASONS':reasons,'FORMAL_ALLOWED':not reasons,'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
    c.json_write(c.EVID/'ROUND25_PREFLIGHT.json',out)
    del model,net,dopt; torch.cuda.empty_cache(); gc.collect(); return out

if __name__=='__main__': print(json.dumps(run(),indent=2))
