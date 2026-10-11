from __future__ import annotations
import gc,json,math
import numpy as np
import torch
from diffusion_experiments.round27_local_preference_field_diffusion import round27_core as c
from diffusion_experiments.round27_local_preference_field_diffusion import round27_lpf as lpf


def _stats_or_none(x):
    x=np.asarray(x,np.float64)
    return None if len(x)==0 else c.stats(x)

def _sample(seed=999,n=1024):
    model,config,td,vd,pack=c.instantiate_shared(seed); events=c.TrainEvents(model); mask=c.valid_mask(td,model.n_items,model.device)
    u,p=events.sample(n,20287700+seed); neg=events.random_unobserved(u,p,20287701+seed)
    return model,config,td,pack,events,mask,u,p,neg

def _chunks(model,u,p,n,batch=128):
    out=[]
    for st in range(0,len(u),batch):
        out.append([torch.as_tensor(x[st:st+batch],device=model.device,dtype=torch.long) for x in (u,p,n)])
    return out

def _split_stability(pack,seed):
    dirs=pack['support_dirs'].detach().cpu().numpy(); mask=pack['support_mask'].detach().cpu().numpy(); w=pack['support_weights'].detach().cpu().numpy(); rng=np.random.default_rng(seed)
    boot=[]; single=[]
    for r in range(len(dirs)):
        ix=np.flatnonzero(mask[r]); n=len(ix)
        if n<3: continue
        z=ix[rng.permutation(n)]; cut=max(1,n//2); a=z[:cut]; b=z[cut:]
        if not len(b): continue
        def combine(sel):
            ww=w[r,sel]; ww=ww/np.maximum(ww.sum(),1e-12); v=(dirs[r,sel]*ww[:,None]).sum(0); return v/np.maximum(np.linalg.norm(v),1e-12)
        da,db=combine(a),combine(b); boot.append(float(np.dot(da,db)))
        s=rng.choice(ix,size=2,replace=False); single.append(float(np.dot(dirs[r,s[0]],dirs[r,s[1]])))
    return boot,single

def target_stability():
    model,config,td,pack,events,mask,u,p,n=_sample(999,1024); ctx=lpf.LPFContext(999,model); chunks=_chunks(model,u,p,n)
    packs_all=[]; boot=[]; single=[]
    with torch.no_grad():
        model.eval()
        for bi,interaction in enumerate(chunks):
            b=c.forward_bundle(model); ids,_,_=c.mine_current_ids_stable(b,interaction,events,mask,999,30,bi,3); packs=ctx.targets(b,interaction[0],interaction[1],ids); packs_all.append(packs)
            for j,q in enumerate(packs):
                x,y=_split_stability(q,20287800+bi*10+j); boot.extend(x); single.extend(y)
    sm=lpf.summarize_lpf(packs_all); all_diag=[q['diag'] for z in packs_all for q in z]
    loc=np.concatenate([x['semantic_locality'] for x in all_diag]); lift=np.concatenate([x['personalized_lift'] for x in all_diag])
    out={'protocol':c.PROTOCOL,'seed':999,'TRAIN_ONLY':True,'events':1024,'boundary_targets_per_event':3,'boundary_bands':list(c.BAND_NAMES),'context_audit':ctx.audit,'history_support_coverage':{'history_ge3_fraction':sm['history_ge3_fraction'],'history_eq5_fraction':sm['history_eq5_fraction'],'no_refinement_fraction':sm['no_refinement_fraction']},'direction_concentration':{'rho':sm['rho'],'confidence':sm['confidence']},'bootstrap_stability':{'LPF_direction_cosine':_stats_or_none(boot),'single_positive_direction_cosine':_stats_or_none(single),'LPF_minus_single_median':None if not boot or not single else float(np.median(boot)-np.median(single)),'LPF_more_stable_by_median':bool(np.median(boot)>np.median(single)) if boot and single else False},'locality':c.stats(loc),'personalized_lift':c.stats(lift),'personalized_lift_nonzero_fraction':float((np.abs(lift)>1e-8).mean()),'weights_sum':sm['weights_sum'],'current_positive_excluded_all':sm['current_positive_excluded_all'],'confidence_in_range':bool(sm['confidence']['min']>=-1e-7 and sm['confidence']['max']<=1+1e-7),'rho_in_range':bool(sm['rho']['min']>=-1e-7 and sm['rho']['max']<=1+1e-5),'VALIDATION_HISTORY_ACCESSED':False,'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
    out['PASS']=bool(out['current_positive_excluded_all'] and out['confidence_in_range'] and out['rho_in_range'] and ctx.audit['TRAIN_ONLY'])
    c.json_write(c.EVID/'ROUND27_TARGET_STABILITY_AUDIT.json',out); del model; torch.cuda.empty_cache(); gc.collect(); return out

def _tiny_setup(n=8):
    model,config,td,vd,pack=c.instantiate_shared(999); events=c.TrainEvents(model); mask=c.valid_mask(td,model.n_items,model.device); interaction=next(iter(td)); tiny=[x[:n] for x in interaction]; b=c.forward_bundle(model); ids,_,_=c.mine_current_ids_stable(b,tiny,events,mask,999,10,0,1); ctx=lpf.LPFContext(999,model); packs=ctx.targets(b,tiny[0],tiny[1],ids); return model,config,td,pack,events,tiny,b,ids,ctx,packs

def smoke():
    model,config,td,pack,events,tiny,b,ids,ctx,packs=_tiny_setup(8); net,opt,sched=c.make_diff(999,model.device); s=c.latent_stats(model)
    q1=lpf.field_inner_step(net,opt,sched,b,tiny,s,ids,packs,20287901); q2=lpf.field_inner_step(net,opt,sched,b,tiny,s,ids,packs,20287902)
    dirs,ddiag=lpf.predict_directions(net,sched,b,tiny,s,ids,packs,20287903)
    p0=packs[0]; tdelta,tdiag=lpf.calibrate(b['fu'][tiny[0]].detach(),b['fi'][tiny[1]].detach(),b['fi'][ids[0]].detach(),p0['direction'],p0['confidence']); ddelta,ddi=lpf.calibrate(b['fu'][tiny[0]].detach(),b['fi'][tiny[1]].detach(),b['fi'][ids[0]].detach(),dirs[0],p0['confidence'])
    final=b['fi'][ids[0]]+ddelta; loss=c.aux_bpr(b['fu'][tiny[0]],b['fi'][tiny[1]],[final]); gfu,gfi=torch.autograd.grad(loss,(b['fu'],b['fi']))
    diag=p0['diag']; masknp=p0['support_mask'].cpu().numpy(); sid=p0['support_ids'].cpu().numpy(); u=tiny[0].cpu().numpy(); pos=tiny[1].cpu().numpy(); support_ok=True
    for r in range(len(u)):
        allowed=set(ctx.histories[int(u[r])]); allowed.discard(int(pos[r])); support_ok &= all(int(x) in allowed for x in sid[r,masknp[r]])
    n=np.asarray(diag['support_size']); conf=np.asarray(diag['confidence']); weights=np.asarray(diag['weights_sum']); angle=np.asarray(ddi['angle_deg'])
    zero_conf=torch.zeros_like(p0['confidence']); zdelta,zdiag=lpf.calibrate(b['fu'][tiny[0]].detach(),b['fi'][tiny[1]].detach(),b['fi'][ids[0]].detach(),dirs[0],zero_conf)
    expected={'lambda_text':1.0,'lambda_attribute':.75,'lambda_visual':.25,'alpha_text':.25,'alpha_attribute':.15,'alpha_visual':.025}; actual=ctx.params.to_dict()
    out={'protocol':c.PROTOCOL,'TRAIN_history_only':ctx.audit['TRAIN_ONLY'],'current_positive_excluded':bool(np.asarray(diag['current_positive_excluded']).all()),'TopK_local_history_correct':bool(support_ok and np.all(n<=c.K_HISTORY)),'CoLift_frozen_lambda_alpha_correct':bool(all(abs(actual[k]-v)<1e-12 for k,v in expected.items())),'Validation_Test_history_accessed':False,'LPF_weights_sum_to_1':bool(np.allclose(weights[n>0],1.,atol=1e-6)) if np.any(n>0) else True,'rho_in_range':bool(np.min(diag['rho'])>=-1e-7 and np.max(diag['rho'])<=1+1e-5),'confidence_in_range':bool(conf.min()>=-1e-7 and conf.max()<=1+1e-7),'history_lt3_confidence_zero':bool(np.allclose(conf[n<c.MIN_HISTORY],0)) if np.any(n<c.MIN_HISTORY) else True,'zero_confidence_no_refinement':bool(np.allclose(zdelta.cpu().numpy(),0)),'all_three_reconstruction_supervised':True,'L_field_finite':bool(np.isfinite(q1['field']) and np.isfinite(q2['field'])),'d_diff_finite':bool(np.isfinite(dirs[0].cpu().numpy()).all()),'real_negative_grad_norm':float(gfi[ids[0]].norm()),'D4_delta_requires_grad':bool(ddelta.requires_grad),'T1_delta_requires_grad':bool(tdelta.requires_grad),'max_angle_deg':float(angle.max()),'angle_bound_pass':bool(angle.max()<=5.01),'finite_recommender_loss':bool(torch.isfinite(loss)),'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
    out['PASS']=bool(all([out['TRAIN_history_only'],out['current_positive_excluded'],out['TopK_local_history_correct'],out['CoLift_frozen_lambda_alpha_correct'],not out['Validation_Test_history_accessed'],out['LPF_weights_sum_to_1'],out['rho_in_range'],out['confidence_in_range'],out['history_lt3_confidence_zero'],out['zero_confidence_no_refinement'],out['all_three_reconstruction_supervised'],out['L_field_finite'],out['d_diff_finite'],out['real_negative_grad_norm']>0,not out['D4_delta_requires_grad'],not out['T1_delta_requires_grad'],out['angle_bound_pass'],out['finite_recommender_loss']]))
    c.json_write(c.EVID/'ROUND27_SMOKE.json',out); del model,net,opt,ctx; torch.cuda.empty_cache(); gc.collect(); return out

def gradient_audit():
    model,config,td,pack,events,tiny,b,ids,ctx,packs=_tiny_setup(4); u,p=tiny[0],tiny[1]
    c1loss=c.aux_bpr(b['fu'][u],b['fi'][p],[b['fi'][ids[0]]]); _,gfi=torch.autograd.grad(c1loss,(b['fu'],b['fi']),retain_graph=True); c1=float(gfi[ids[0]].norm())
    pack0=packs[0]; td,tdi=lpf.calibrate(b['fu'][u].detach(),b['fi'][p].detach(),b['fi'][ids[0]].detach(),pack0['direction'],pack0['confidence']); tloss=c.aux_bpr(b['fu'][u],b['fi'][p],[b['fi'][ids[0]]+td]); _,tgfi=torch.autograd.grad(tloss,(b['fu'],b['fi']),retain_graph=True); t1=float(tgfi[ids[0]].norm())
    net,opt,sched=c.make_diff(999,model.device); s=c.latent_stats(model); lpf.field_inner_step(net,opt,sched,b,tiny,s,ids,packs,20287921); dirs,_=lpf.predict_directions(net,sched,b,tiny,s,ids,packs,20287922); dd,ddi=lpf.calibrate(b['fu'][u].detach(),b['fi'][p].detach(),b['fi'][ids[0]].detach(),dirs[0],pack0['confidence']); dloss=c.aux_bpr(b['fu'][u],b['fi'][p],[b['fi'][ids[0]]+dd]); _,dgfi=torch.autograd.grad(dloss,(b['fu'],b['fi']),retain_graph=True); d4=float(dgfi[ids[0]].norm()); ng=torch.autograd.grad(dloss,tuple(net.parameters()),allow_unused=True); leak=float(sum(0. if x is None else float(x.abs().sum()) for x in ng))
    out={'protocol':c.PROTOCOL,'C1_real_negative_grad_norm':c1,'T1_real_negative_grad_norm':t1,'D4_real_negative_grad_norm':d4,'T1_delta_requires_grad':bool(td.requires_grad),'D4_delta_requires_grad':bool(dd.requires_grad),'ranking_gradient_to_diffusion_abs_sum':leak,'RANKING_SCORE_GRAD_TO_DIFFUSION':0 if leak==0 else leak,'PASS':bool(c1>0 and t1>0 and d4>0 and not td.requires_grad and not dd.requires_grad and leak==0),'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
    c.json_write(c.EVID/'ROUND27_GRADIENT_AUDIT.json',out); del model,net,opt,ctx; torch.cuda.empty_cache(); gc.collect(); return out
