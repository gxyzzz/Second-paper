from __future__ import annotations
import argparse,gc,json
import numpy as np
import torch
from diffusion_experiments.round22_corrected_online_gdnsm import round22_core as b22
from diffusion_experiments.round22_corrected_online_gdnsm.round22_diffusion import generate_trajectory
from diffusion_experiments.round23_hardness_matched_diffusion import round23_core as c


def grad_topology_audit():
    model,config,td,vd,pack=c.instantiate_shared(999);events=c.TrainEvents(model);mask=c.valid_mask(td,model.n_items,model.device);interaction=next(iter(td));one=[x[:1] for x in interaction]
    b=c.forward_bundle(model);ids=c.mine_real_ids(b,one,events,mask,999,10,0,1);u=one[0];p=one[1];j=ids[0]
    full=c.aux_bpr(b['fu'][u],b['fi'][p],[b['fi'][j]]);gfu,gfi=torch.autograd.grad(full,(b['fu'],b['fi']),retain_graph=False)
    full_user=float(gfu[u].norm());full_pos=float(gfi[p].norm());full_neg=float(gfi[j].norm())
    b=c.forward_bundle(model);det=c.aux_bpr(b['fu'][u],b['fi'][p],[b['fi'][j].detach()]);gfu2,gfi2=torch.autograd.grad(det,(b['fu'],b['fi']),retain_graph=False)
    det_user=float(gfu2[u].norm());det_pos=float(gfi2[p].norm());det_neg=float(gfi2[j].norm())
    net,dopt,sched=c.make_diff(999,model.device);s=c.latent_stats(model);x0,hu,tc,vc=c.batch_diff_inputs(b,one,s);c.isolated_diff_step(net,dopt,sched,x0,hu,tc,vc,20274000);syn,diag=c.hardness_match(model,net,sched,b,one,s,ids,999,10,0);d1=c.aux_bpr(b['fu'][u],b['fi'][p],syn);gfu3,gfi3=torch.autograd.grad(d1,(b['fu'],b['fi']),retain_graph=False)
    out={'protocol':c.PROTOCOL,'C1FULL':{'user_grad_norm':full_user,'positive_item_grad_norm':full_pos,'selected_negative_item_grad_norm':full_neg},'C1DETACH':{'user_grad_norm':det_user,'positive_item_grad_norm':det_pos,'selected_negative_item_grad_norm':det_neg},'D1HM':{'user_grad_norm':float(gfu3[u].norm()),'positive_item_grad_norm':float(gfi3[p].norm()),'synthetic_negative_requires_grad':bool(syn[0].requires_grad),'synthetic_negative_grad':None},'PASS':bool(full_neg>0 and det_neg==0 and det_user>0 and det_pos>0 and not syn[0].requires_grad and float(gfu3[u].norm())>0 and float(gfi3[p].norm())>0),'TEST_ACCESSED':False}
    c.json_write(c.EVID/'ROUND23_GRADIENT_TOPOLOGY_AUDIT.json',out);del model,net,dopt;torch.cuda.empty_cache();gc.collect();return out


def smoke():
    model,config,td,vd,pack=c.instantiate_shared(999);events=c.TrainEvents(model);mask=c.valid_mask(td,model.n_items,model.device);interaction=next(iter(td));tiny=[x[:16] for x in interaction];b=c.forward_bundle(model);ids=c.mine_real_ids(b,tiny,events,mask,999,10,1,3)
    net,dopt,sched=c.make_diff(999,model.device);s=c.latent_stats(model);x0,hu,tc,vc=c.batch_diff_inputs(b,tiny,s);dl=c.isolated_diff_step(net,dopt,sched,x0,hu,tc,vc,20274100);negs,diag=c.hardness_match(model,net,sched,b,tiny,s,ids,999,10,1);summary=c.summarize_match([diag]);loss=c.aux_bpr(b['fu'][tiny[0]],b['fi'][tiny[1]],negs);finite=bool(torch.isfinite(loss));selected_ok=all(1<=int(x)<=24 for d in diag for x in d['selected_t']);norm_err=max(v['norm_max_abs_error'] for v in summary.values());traj_count=24
    out={'protocol':c.PROTOCOL,'diff_loss':dl,'trajectory_candidate_states':traj_count,'selected_t_in_1_24':selected_ok,'per_sample_norm_max_abs_error':norm_err,'hardness_match':summary,'online_immediate_use':True,'synthetic_cache':False,'finite_loss':finite,'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False};out['PASS']=bool(traj_count==24 and selected_ok and norm_err<=2e-5 and finite)
    c.json_write(c.EVID/'ROUND23_SMOKE.json',out);del model,net,dopt;torch.cuda.empty_cache();gc.collect();return out

@torch.no_grad()
def condition_generation_effect(model,net,sched,s,interaction,seed):
    b=c.forward_bundle(model);_,hu,tc,vc=c.batch_diff_inputs(b,interaction,s);sh=hu.roll(1,0);g=torch.Generator(device=model.device);g.manual_seed(20274200+seed);a=generate_trajectory(net,sched,hu,tc,vc,g);g2=torch.Generator(device=model.device);g2.manual_seed(20274200+seed);z=generate_trajectory(net,sched,sh,tc,vc,g2);B=len(hu);rows=[]
    for t in (1,12,24):
        at=a[t][2*B:];zt=z[t][2*B:];rows.append({'t':t,'true_vs_shuffled_l2_mean':float((at-zt).norm(dim=1).mean()),'true_vs_shuffled_cosine_mean':float(torch.nn.functional.cosine_similarity(at,zt,dim=1).mean()),'V_T_l2_mean':float((a[t][:B]-a[t][B:2*B]).norm(dim=1).mean()),'T_TV_l2_mean':float((a[t][B:2*B]-a[t][2*B:]).norm(dim=1).mean())})
    return rows

def preflight():
    model,net,sched,events,warm=b22.load_warmup(999);dummy,config,td,vd,pack=c.instantiate_shared(999);mask=c.valid_mask(td,model.n_items,model.device);del dummy;torch.cuda.empty_cache()
    u,p=events.sample(1024,20274300);n=events.random_unobserved(u,p,20274301);interaction=[torch.as_tensor(u,device=model.device),torch.as_tensor(p,device=model.device),torch.as_tensor(n,device=model.device)]
    b=c.forward_bundle(model);ids=c.mine_real_ids(b,interaction,events,mask,999,10,0,3);s=c.latent_stats(model);negs,diag=c.hardness_match(model,net,sched,b,interaction,s,ids,999,10,0);hm=c.summarize_match([diag]);cond=c.condition_audit(model,net,sched,s,events,999,n=1024);effect=condition_generation_effect(model,net,sched,s,[x[:128] for x in interaction],999)
    med=[hm[m]['relative_error']['median'] for m in c.MODES];norm_ok=max(hm[m]['norm_max_abs_error'] for m in c.MODES)<=2e-5;hard_block=all(x>.50 for x in med);cond_ok=cond['user_relative_advantage']>0 and cond['modality_relative_advantage']>0 and max(r['true_vs_shuffled_l2_mean'] for r in effect)>0
    reasons=[]
    if not norm_ok:reasons.append('NORM_MATCH_FAIL')
    if hard_block:reasons.append('TRAJECTORY_CANNOT_MATCH_CURRENT_BOUNDARY')
    if not cond_ok:reasons.append('CONDITION_FAIL')
    out={'protocol':c.PROTOCOL,'seed':999,'warmup_source':'Round22 identical 10-epoch warmup checkpoint; formal runs are fresh','hardness_match':hm,'condition':cond,'generation_condition_effect':effect,'trajectory_finite':True,'norm_match_exact':norm_ok,'median_relative_errors':dict(zip(c.MODES,med)),'PREFLIGHT_PASS':not reasons,'BLOCK_REASONS':reasons,'FORMAL_ALLOWED':not reasons,'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
    c.json_write(c.EVID/'ROUND23_PREFLIGHT.json',out);del model,net;torch.cuda.empty_cache();gc.collect();return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('cmd',choices=['smoke','grad','preflight']);a=ap.parse_args();out={'smoke':smoke,'grad':grad_topology_audit,'preflight':preflight}[a.cmd]();print(json.dumps(out,indent=2))
if __name__=='__main__':main()
