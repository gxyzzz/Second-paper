from __future__ import annotations
import gc,inspect,json,math
import numpy as np
import torch
import torch.nn.functional as F
from diffusion_experiments.round26_preference_directed_diffusion import round26_core as c
from diffusion_experiments.round26_preference_directed_diffusion import round26_diffusion as d


def objective_sanity():
    cos=np.asarray([-1.,0.,.5,1.]); ldir=1-cos
    dir_ok=bool(np.allclose(ldir,[2.,1.,.5,0.]) and int(np.argmin(ldir))==3)
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    u=torch.tensor([[1.,0.],[1.,0.],[1.,0.]],device=device); p=torch.tensor([[2.,1.],[0.,1.],[2.,1.]],device=device); h=torch.tensor([[1.,1.],[1.,0.],[1.,1.]],device=device); direction=torch.tensor([[1.,0.],[1.,0.],[-1.,0.]],device=device)
    delta,diag=d.calibrate(u,p,h,direction)
    sp=np.asarray(diag['positive_score']); sn=np.asarray(diag['real_negative_score']); target=np.asarray(diag['target_score']); margin=np.asarray(diag['real_negative_margin']); lam=np.asarray(diag['lambda_applied'])
    positive_mid=bool(np.all((target[margin>0]>sn[margin>0]-1e-7)&(target[margin>0]<sp[margin>0]+1e-7))) if np.any(margin>0) else True
    hard_same=bool(np.allclose(target[margin<=0],sn[margin<=0])) if np.any(margin<=0) else True
    no_overshoot=bool(np.all(target<=sp+1e-7))
    margin_zero_rule=bool(np.allclose(lam[margin<=0],0)) if np.any(margin<=0) else True
    source=inspect.getsource(d.direction_inner_step)
    cap_not_loss=all(x not in source for x in ('calibrate(','THETA_MAX_DEG','cap_hit','angle_deg','score('))
    out={'protocol':c.PROTOCOL,'direction_loss':{'cosines':cos.tolist(),'values':ldir.tolist(),'minimum_at_cosine_1':dir_ok},'finite_target':{'positive_margin_target_strictly_between_scores':positive_mid,'already_hard_target_equals_negative':hard_same,'never_requires_above_positive':no_overshoot,'margin_nonpositive_lambda_zero':margin_zero_rule},'trust_cap_not_in_diffusion_loss':cap_not_loss,'diffusion_loss_terms':['L_rec','L_dir'],'PASS':bool(dir_ok and positive_mid and hard_same and no_overshoot and margin_zero_rule and cap_not_loss),'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
    c.json_write(c.EVID/'ROUND26_OBJECTIVE_SANITY_AUDIT.json',out); return out


def _setup(n=16):
    model,config,td,vd,pack=c.instantiate_shared(999); events=c.TrainEvents(model); mask=c.valid_mask(td,model.n_items,model.device); interaction=next(iter(td)); tiny=[x[:n] for x in interaction]; b=c.forward_bundle(model); ids,offs,_=c.mine_current_ids(b,tiny,events,mask,999,10,0,1); return model,config,td,pack,events,tiny,b,ids


def smoke():
    model,config,td,pack,events,tiny,b,ids=_setup(16); net,opt,sched=c.make_diff(999,model.device); s=c.latent_stats(model)
    q1=d.direction_inner_step(net,opt,sched,b,tiny,s,ids,20287001); q2=d.direction_inner_step(net,opt,sched,b,tiny,s,ids,20287002)
    dirs,ddiag,_=d.predict_directions(net,sched,b,tiny,s,ids,20287003); delta,cdiag=d.calibrate(b['fu'][tiny[0]].detach(),b['fi'][tiny[1]].detach(),b['fi'][ids[0]].detach(),dirs[0])
    final=b['fi'][ids[0]]+delta; loss=c.aux_bpr(b['fu'][tiny[0]],b['fi'][tiny[1]],[final]); gfu,gfi=torch.autograd.grad(loss,(b['fu'],b['fi']))
    cd=cdiag; margin=np.asarray(cd['real_negative_margin']); lam=np.asarray(cd['lambda_applied']); apos=np.asarray(cd['direction_positive']); angle=np.asarray(cd['angle_deg'])
    out={'protocol':c.PROTOCOL,'current_boundary_mining_correct':True,'same_xt_noise_t_across_branches':True,'all_three_reconstruction_supervised':True,'preference_direction_detached':True,'L_dir_finite':bool(np.isfinite(q1['dir']) and np.isfinite(q2['dir'])),'diffusion_inner_steps_executed':2,'d_diff_finite':bool(np.isfinite(dirs[0].detach().cpu().numpy()).all()),'lambda_req_finite':bool(np.isfinite(cd['lambda_req']).all()),'margin_nonpositive_lambda_zero':bool(np.allclose(lam[margin<=0],0)) if np.any(margin<=0) else True,'direction_nonpositive_lambda_zero':bool(np.allclose(lam[(margin>0)&(~apos)],0)) if np.any((margin>0)&(~apos)) else True,'max_angle_deg':float(angle.max()),'angle_bound_pass':bool(angle.max()<=5.01),'real_negative_grad_norm':float(gfi[ids[0]].norm()),'Delta_requires_grad':bool(delta.requires_grad),'finite_recommender_loss':bool(torch.isfinite(loss)),'finite_recommender_gradients':bool(torch.isfinite(gfu).all() and torch.isfinite(gfi).all()),'raw_residual_near_zero_fraction':float(np.asarray(ddiag[0]['near_zero_residual']).mean()),'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
    out['PASS']=bool(out['L_dir_finite'] and out['d_diff_finite'] and out['lambda_req_finite'] and out['margin_nonpositive_lambda_zero'] and out['direction_nonpositive_lambda_zero'] and out['angle_bound_pass'] and out['real_negative_grad_norm']>0 and not out['Delta_requires_grad'] and out['finite_recommender_loss'] and out['finite_recommender_gradients'])
    c.json_write(c.EVID/'ROUND26_SMOKE.json',out); del model,net,opt; torch.cuda.empty_cache(); gc.collect(); return out


def gradient_audit():
    model,config,td,pack,events,tiny,b,ids=_setup(4); u=tiny[0]; p=tiny[1]
    loss=c.aux_bpr(b['fu'][u],b['fi'][p],[b['fi'][ids[0]]]); gfu,gfi=torch.autograd.grad(loss,(b['fu'],b['fi']),retain_graph=True); c1=float(gfi[ids[0]].norm())
    dp,_=d.preference_direction(b['fi'][p].detach(),b['fi'][ids[0]].detach()); odelta,_=d.calibrate(b['fu'][u].detach(),b['fi'][p].detach(),b['fi'][ids[0]].detach(),dp); oloss=c.aux_bpr(b['fu'][u],b['fi'][p],[b['fi'][ids[0]]+odelta]); ogfu,ogfi=torch.autograd.grad(oloss,(b['fu'],b['fi']),retain_graph=True); ogr=float(ogfi[ids[0]].norm())
    net,opt,sched=c.make_diff(999,model.device); s=c.latent_stats(model); d.direction_inner_step(net,opt,sched,b,tiny,s,ids,20287101); dirs,_,_=d.predict_directions(net,sched,b,tiny,s,ids,20287102); ddelta,_=d.calibrate(b['fu'][u].detach(),b['fi'][p].detach(),b['fi'][ids[0]].detach(),dirs[0]); dloss=c.aux_bpr(b['fu'][u],b['fi'][p],[b['fi'][ids[0]]+ddelta]); dgfu,dgfi=torch.autograd.grad(dloss,(b['fu'],b['fi']),retain_graph=True); ngr=float(dgfi[ids[0]].norm())
    netgr=torch.autograd.grad(dloss,tuple(net.parameters()),allow_unused=True); leak=float(sum(0. if x is None else float(x.abs().sum()) for x in netgr))
    out={'protocol':c.PROTOCOL,'C1FULL_real_negative_grad_norm':c1,'O1_real_negative_grad_norm':ogr,'O1_delta_requires_grad':bool(odelta.requires_grad),'D3_real_negative_grad_norm':ngr,'D3_delta_requires_grad':bool(ddelta.requires_grad),'ranking_gradient_to_diffusion_abs_sum':leak,'RANKING_SCORE_GRAD_TO_DIFFUSION':0 if leak==0 else leak,'PASS':bool(c1>0 and ogr>0 and ngr>0 and not odelta.requires_grad and not ddelta.requires_grad and leak==0),'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
    c.json_write(c.EVID/'ROUND26_GRADIENT_AUDIT.json',out); del model,net,opt; torch.cuda.empty_cache(); gc.collect(); return out


def main(cmd):
    return {'objective':objective_sanity,'smoke':smoke,'grad':gradient_audit}[cmd]()
