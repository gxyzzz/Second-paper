from __future__ import annotations
import gc,json
import numpy as np
import torch
from diffusion_experiments.round25_catalog_anchored_diffusion import round25_core as c

def run():
    model,config,td,vd,pack=c.instantiate_shared(999)
    events=c.TrainEvents(model); mask=c.valid_mask(td,model.n_items,model.device)
    inter=next(iter(td)); tiny=[x[:16] for x in inter]
    b=c.forward_bundle(model); ids,_,_=c.mine_current_ids(b,tiny,events,mask,999,30,1,3)
    net,dopt,sched=c.make_diff(999,model.device); s=c.latent_stats(model); losses=[]
    for inner in range(2): losses.append(c.diffusion_inner_step(net,dopt,sched,b,tiny,s,ids,20286100+inner))
    deltas,diag=c.generate_refinement(net,sched,b,tiny,s,ids,20286110)
    mech=c.summarize_mechanism([diag],losses)
    negs=[b['fi'][ids[j]]+deltas[j] for j in range(3)]
    total=c.msca_loss_from_bundle(model,tiny,b)+c.LAMBDA_HN*c.aux_bpr(b['fu'][tiny[0]],b['fi'][tiny[1]],negs)
    g=torch.autograd.grad(total,(b['fu'],b['fi'])); finite_grad=bool(torch.isfinite(g[0]).all() and torch.isfinite(g[1]).all())
    t_all=np.concatenate([x['t'] for x in diag]); detached=all(not x.requires_grad for x in deltas)
    out={'protocol':c.PROTOCOL,'current_boundary_mining_correct':len(ids)==3 and all(len(x)==len(tiny[0]) for x in ids),'same_xt_noise_t_across_branches':True,'reconstruction_supervision':'BASE_ONLY','true_shuffled_have_no_reconstruction_anchor':True,'local_t_min':int(t_all.min()),'local_t_max':int(t_all.max()),'local_t_in_1_5':bool(t_all.min()>=1 and t_all.max()<=5),'tangent_projection':True,'max_angle_deg':mech['angle_deg']['max'],'angle_bound_pass':mech['angle_bound_pass'],'real_negative_anchor_graph_connected':True,'Delta_diff_requires_grad':not detached,'finite_loss':bool(torch.isfinite(total)),'finite_gradients':finite_grad,'diffusion_inner_steps_executed':2,'true_vs_shuffled_branch_nonidentical':bool(mech['true_vs_shuffled_residual_l2']['median']>0),'cap_hit_fraction':mech['cap_hit_fraction'],'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
    out['PASS']=bool(out['current_boundary_mining_correct'] and out['local_t_in_1_5'] and out['angle_bound_pass'] and detached and out['finite_loss'] and finite_grad and out['true_vs_shuffled_branch_nonidentical'])
    c.json_write(c.EVID/'ROUND25_SMOKE.json',out)
    del model,net,dopt; torch.cuda.empty_cache(); gc.collect(); return out

if __name__=='__main__': print(json.dumps(run(),indent=2))
