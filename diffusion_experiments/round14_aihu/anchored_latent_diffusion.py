from __future__ import annotations
import torch
import torch.nn.functional as F
from diffusion_experiments.round13_iudlp import latent_diffusion as r13

LATENT_DIM=r13.LATENT_DIM
T_LATENT=r13.T_LATENT
RHO=r13.RHO
LAMBDA_REC=r13.LAMBDA_REC
LAMBDA_ANCHOR=0.25
LAMBDA_USER=r13.LAMBDA_USER
USER_MARGIN=r13.USER_MARGIN
T_INFER=r13.T_INFER
INFER_SEEDS=r13.INFER_SEEDS
AIHUModules=r13.IUDLPModules
freeze_recommender=r13.freeze_recommender
parameter_audit=r13.parameter_audit
forward_components=r13.forward_components
fuse_item=r13.fuse_item
condition=r13.condition
noisy_state=r13.noisy_state
tangent_augment=r13.tangent_augment
shuffled_user_ids=r13.shuffled_user_ids
cosine_alpha_bar=r13.cosine_alpha_bar
build_noise_tables=r13.build_noise_tables
inference_residuals=r13.inference_residuals
pair_scores=r13.pair_scores
grad_l2=r13.grad_l2


def _predictions(denoiser, modules, h, user_cf, item_cf, t, alpha_bar, noise, wrong_cf=None):
    x,ht=noisy_state(h,t,alpha_bar,noise)
    z=torch.zeros_like(user_cf)
    ct=condition(modules,user_cf,item_cf)
    c0=condition(modules,z,item_cf)
    pt=denoiser(ht,t,ct)
    p0=denoiser(ht,t,c0)
    ps=None
    if wrong_cf is not None:
        cs=condition(modules,wrong_cf,item_cf)
        ps=denoiser(ht,t,cs)
    return x,ht,pt,p0,ps


def _two_modality_predictions(model,modules,c,user_ids,item_ids,t,alpha_bar,noise_t,noise_v,with_shuf=True):
    ucf=c['collab_user'][user_ids]
    icf=c['collab_item'][item_ids]
    wrong_cf=None
    wrong_ids=None
    if with_shuf:
        base_users=user_ids
        wrong_ids=shuffled_user_ids(base_users,model.n_users)
        wrong_cf=c['collab_user'][wrong_ids]
    tx=_predictions(modules.text,modules,c['text_item'][item_ids],ucf,icf,t,alpha_bar,noise_t,wrong_cf)
    vx=_predictions(modules.visual,modules,c['image_item'][item_ids],ucf,icf,t,alpha_bar,noise_v,wrong_cf)
    return tx,vx,wrong_ids


def anchor_terms(model,modules,c,user_ids,item_ids,t,alpha_bar,noise_t=None,noise_v=None):
    n=len(item_ids)
    if noise_t is None: noise_t=torch.randn((n,LATENT_DIM),device=item_ids.device)
    if noise_v is None: noise_v=torch.randn((n,LATENT_DIM),device=item_ids.device)
    tx,vx,_=_two_modality_predictions(model,modules,c,user_ids,item_ids,t,alpha_bar,noise_t,noise_v,with_shuf=False)
    xt,_,pt,p0t,_=tx; xv,_,pv,p0v,_=vx
    lt=F.mse_loss(pt,xt.detach()); l0t=F.mse_loss(p0t,xt.detach())
    lv=F.mse_loss(pv,xv.detach()); l0v=F.mse_loss(p0v,xv.detach())
    ltrue=.5*(lt+lv); lzero=.5*(l0t+l0v); total=.5*(ltrue+lzero)
    return total,{
      'L_anchor_true':ltrue,'L_anchor_zero':lzero,'L_anchor_total':total,
      'L_anchor_text_true':lt,'L_anchor_text_zero':l0t,'L_anchor_visual_true':lv,'L_anchor_visual_zero':l0v,
      'zero_true_error_ratio':lzero.detach()/ltrue.detach().clamp_min(1e-12),
      't_min':t.min().detach().float(),'t_max':t.max().detach().float(),
    }


def preference_terms(model,modules,c,users,pos,neg,t,alpha_bar,noise_t=None,noise_v=None,return_outputs=False):
    b=len(users); ids=torch.cat([pos,neg]); scorer_users=torch.cat([users,users])
    if t.ndim==1 and len(t)==b: t=torch.cat([t,t])
    n=len(ids)
    if noise_t is None:
        base=torch.randn((b,LATENT_DIM),device=users.device); noise_t=torch.cat([base,base])
    if noise_v is None:
        base=torch.randn((b,LATENT_DIM),device=users.device); noise_v=torch.cat([base,base])
    # Same item/noise/timestep; only condition identity differs.
    tx,vx,wrong_ids=_two_modality_predictions(model,modules,c,scorer_users,ids,t,alpha_bar,noise_t,noise_v,with_shuf=True)
    xt,htt,pt,p0t,pst=tx; xv,htv,pv,p0v,psv=vx
    rt=pt-p0t; rv=pv-p0v; rst=pst-p0t; rsv=psv-p0v
    at=tangent_augment(c['text_item'][ids],rt,RHO); av=tangent_augment(c['image_item'][ids],rv,RHO)
    ast=tangent_augment(c['text_item'][ids],rst,RHO); asv=tangent_augment(c['image_item'][ids],rsv,RHO)
    true_items=fuse_item(model,c['collab_item'][ids],av,at,c['struct_item'][ids])
    shuf_items=fuse_item(model,c['collab_item'][ids],asv,ast,c['struct_item'][ids])
    u=c['final_user'][users]
    mtrue=(u*true_items[:b]).sum(1)-(u*true_items[b:]).sum(1)
    mshuf=(u*shuf_items[:b]).sum(1)-(u*shuf_items[b:]).sum(1)
    lrec=F.softplus(-mtrue).mean()
    luser=F.softplus(mshuf-mtrue+USER_MARGIN).mean()
    # Anchoring values are useful for A1, where the same random-t path is shared.
    lt=F.mse_loss(pt,xt.detach()); l0t=F.mse_loss(p0t,xt.detach())
    lv=F.mse_loss(pv,xv.detach()); l0v=F.mse_loss(p0v,xv.detach())
    ltrue=.5*(lt+lv); lzero=.5*(l0t+l0v); lanchor=.5*(ltrue+lzero)
    out={
      'L_rec':lrec,'L_user':luser,'m_true':mtrue.mean(),'m_shuf':mshuf.mean(),'m_true_minus_shuf':(mtrue-mshuf).mean(),
      'L_anchor_true':ltrue,'L_anchor_zero':lzero,'L_anchor_total':lanchor,
      'true_zero_residual_norm':.5*(rt.norm(dim=1).mean()+rv.norm(dim=1).mean()),
      'shuf_zero_residual_norm':.5*(rst.norm(dim=1).mean()+rsv.norm(dim=1).mean()),
      'residual_ratio_shuf_true':(.5*(rst.norm(dim=1).mean()+rsv.norm(dim=1).mean()))/(.5*(rt.norm(dim=1).mean()+rv.norm(dim=1).mean())).clamp_min(1e-12),
      't_min':t.min().detach().float(),'t_max':t.max().detach().float(),
    }
    if return_outputs:
        out.update({'mtrue_vec':mtrue,'mshuf_vec':mshuf,'rt':rt,'rv':rv,'rst':rst,'rsv':rsv,'pt':pt,'p0t':p0t,'pv':pv,'p0v':p0v,'xt':xt,'xv':xv})
    return out


def sample_hard_negatives(users,hard_lookup):
    cols=torch.randint(0,hard_lookup.shape[1],(len(users),),device=users.device)
    neg=hard_lookup[users,cols]
    if (neg<0).any(): raise RuntimeError('missing hard candidate for TRAIN user')
    return neg,cols+6


def training_loss(model,modules,interaction,c,variant,alpha_bar,hard_lookup=None):
    if variant not in ('A1','A2','A3'): raise ValueError(variant)
    users,pos,orig_neg=interaction[0],interaction[1],interaction[2]
    b=len(users); device=users.device
    pref_neg=orig_neg
    sampled_rank=torch.full((b,),-1,device=device,dtype=torch.long)
    if variant=='A3':
        if hard_lookup is None: raise RuntimeError('A3 requires frozen TRAIN-only hard lookup')
        pref_neg,sampled_rank=sample_hard_negatives(users,hard_lookup)
    if variant=='A1':
        t=torch.randint(1,T_LATENT+1,(b,),device=device)
        pref=preference_terms(model,modules,c,users,pos,pref_neg,t,alpha_bar)
        anch={k:pref[k] for k in ('L_anchor_true','L_anchor_zero','L_anchor_total')}
        recon_t_min=pref['t_min']; recon_t_max=pref['t_max']; pref_t_min=pref['t_min']; pref_t_max=pref['t_max']
    else:
        ids=torch.cat([pos,orig_neg]); uids=torch.cat([users,users])
        td=torch.randint(1,T_LATENT+1,(2*b,),device=device)
        _,anch=anchor_terms(model,modules,c,uids,ids,td,alpha_bar)
        tp=torch.full((b,),T_INFER,device=device,dtype=torch.long)
        pref=preference_terms(model,modules,c,users,pos,pref_neg,tp,alpha_bar)
        recon_t_min=anch['t_min']; recon_t_max=anch['t_max']; pref_t_min=pref['t_min']; pref_t_max=pref['t_max']
    total=LAMBDA_REC*pref['L_rec']+LAMBDA_USER*pref['L_user']+LAMBDA_ANCHOR*anch['L_anchor_total']
    parts={
      'L_rec':pref['L_rec'],'L_anchor_true':anch['L_anchor_true'],'L_anchor_zero':anch['L_anchor_zero'],'L_anchor_total':anch['L_anchor_total'],
      'L_user':pref['L_user'],'total_loss':total,'m_true':pref['m_true'],'m_shuf':pref['m_shuf'],'m_true_minus_shuf':pref['m_true_minus_shuf'],
      'true_zero_residual_norm':pref['true_zero_residual_norm'],'shuf_zero_residual_norm':pref['shuf_zero_residual_norm'],'residual_ratio_shuf_true':pref['residual_ratio_shuf_true'],
      'reconstruction_t_min':recon_t_min,'reconstruction_t_max':recon_t_max,'preference_t_min':pref_t_min,'preference_t_max':pref_t_max,
      'sampled_hard_rank_mean':sampled_rank.float().mean() if variant=='A3' else total.new_tensor(-1.0),
      'hard_rank_6_10_frac':(((sampled_rank>=6)&(sampled_rank<=10)).float().mean() if variant=='A3' else total.new_tensor(0.0)),
      'hard_rank_11_15_frac':(((sampled_rank>=11)&(sampled_rank<=15)).float().mean() if variant=='A3' else total.new_tensor(0.0)),
      'hard_rank_16_20_frac':(((sampled_rank>=16)&(sampled_rank<=20)).float().mean() if variant=='A3' else total.new_tensor(0.0)),
      'hard_rank_21_25_frac':(((sampled_rank>=21)&(sampled_rank<=25)).float().mean() if variant=='A3' else total.new_tensor(0.0)),
      'hard_rank_26_30_frac':(((sampled_rank>=26)&(sampled_rank<=30)).float().mean() if variant=='A3' else total.new_tensor(0.0)),
    }
    return total,parts
