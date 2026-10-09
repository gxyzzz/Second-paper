from __future__ import annotations
import copy, hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_experiments.round13_iudlp import latent_diffusion as r13
from diffusion_experiments.round15_baurp import base_anchored_residual as r15

LATENT_DIM=r13.LATENT_DIM; T_LATENT=r13.T_LATENT; T_INFER=r13.T_INFER; INFER_SEEDS=r13.INFER_SEEDS
RHO=r15.RHO; THETA_MAX_DEG=r15.THETA_MAX_DEG; TANGENT_NORM_CAP=r15.TANGENT_NORM_CAP
USER_MARGIN=.05; LAMBDA_ID=.5; LAMBDA_ZERO=.25; LAMBDA_BOUND=.05

BaseModules=r13.IUDLPModules
freeze_recommender=r13.freeze_recommender
forward_components=r13.forward_components
fuse_item=r13.fuse_item
cosine_alpha_bar=r13.cosine_alpha_bar
build_noise_tables=r13.build_noise_tables
noisy_state=r13.noisy_state
shuffled_user_ids=r13.shuffled_user_ids
grad_l2=r13.grad_l2


class UserModules(nn.Module):
    def __init__(self, context_dim=0):
        super().__init__(); self.context_dim=int(context_dim)
        self.condition=r13.ConditionEncoder(128+self.context_dim,128,64)
        self.text=r13.LatentX0Denoiser(64,64,256,64)
        self.visual=r13.LatentX0Denoiser(64,64,256,64)


def clone_user_from_base(base:BaseModules, context_dim=0):
    u=UserModules(context_dim)
    u.text.load_state_dict(copy.deepcopy(base.text.state_dict()))
    u.visual.load_state_dict(copy.deepcopy(base.visual.state_dict()))
    if int(context_dim)==0:
        u.condition.load_state_dict(copy.deepcopy(base.condition.state_dict()))
    else:
        with torch.no_grad():
            s0=base.condition.net[0]; d0=u.condition.net[0]
            d0.weight.zero_(); d0.weight[:,:128].copy_(s0.weight); d0.bias.copy_(s0.bias)
            u.condition.net[2].load_state_dict(copy.deepcopy(base.condition.net[2].state_dict()))
    return u


def state_sha256(module):
    h=hashlib.sha256()
    for k,v in sorted(module.state_dict().items()):
        h.update(k.encode()); a=v.detach().cpu().contiguous().numpy(); h.update(str(a.dtype).encode()); h.update(np.asarray(a.shape,np.int64).tobytes()); h.update(a.tobytes())
    return h.hexdigest()


def base_condition(base,user_cf,item_cf):
    return base.condition(torch.cat([torch.zeros_like(user_cf),item_cf],dim=-1))


def user_condition(user,user_cf,item_cf,ctx=None):
    if user.context_dim:
        if ctx is None: raise RuntimeError('CoLift context required')
        x=torch.cat([user_cf,item_cf,ctx],dim=-1)
    else: x=torch.cat([user_cf,item_cf],dim=-1)
    return user.condition(x)


def base_reconstruction_loss(base,c,interaction,alpha_bar):
    users,pos,neg=interaction[0],interaction[1],interaction[2]; ids=torch.cat([pos,neg]); uids=torch.cat([users,users])
    t=torch.randint(1,T_LATENT+1,(len(ids),),device=ids.device); icf=c['collab_item'][ids]; ucf=c['collab_user'][uids]
    losses=[]; parts={}
    for name,den,h in [('text',base.text,c['text_item'][ids]),('visual',base.visual,c['image_item'][ids])]:
        noise=torch.randn((len(ids),LATENT_DIM),device=ids.device); x,ht=noisy_state(h,t,alpha_bar,noise)
        pred=den(ht,t,base_condition(base,ucf,icf)); l=F.mse_loss(pred,x.detach()); losses.append(l); parts['L_base_'+name]=l
    total=.5*(losses[0]+losses[1]); parts.update({'L_base':total,'t_min':t.min().detach(),'t_max':t.max().detach()})
    return total,parts


def _bounded_random_augment(h,residual,seed):
    n=h.norm(dim=1,keepdim=True).clamp_min(1e-8); x,tang=r15.tangent_raw(h,residual); raw=tang.norm(dim=1)
    scale=(TANGENT_NORM_CAP/raw.clamp_min(1e-12)).clamp(max=1.0); used_norm=raw*scale
    g=torch.Generator(device=h.device); g.manual_seed(int(seed)); z=torch.randn(h.shape,generator=g,device=h.device,dtype=h.dtype)
    z=z-(z*x).sum(1,keepdim=True)*x; z=z/z.norm(dim=1,keepdim=True).clamp_min(1e-12); tr=z*used_norm.unsqueeze(1)
    out=F.normalize(x+RHO*tr,p=2,dim=1)*n
    return out,{'raw_tangent_norm':raw,'bounded_tangent_norm':used_norm,'angle_deg':r15._angle_deg(h,out),'cap_saturated':raw.gt(TANGENT_NORM_CAP)}


def _augment(h,residual,mode='plus',random_seed=None):
    if mode=='plus': return r15.tangent_augment(h,residual,True,return_diag=True)
    if mode=='minus': return r15.tangent_augment(h,-residual,True,return_diag=True)
    if mode=='random': return _bounded_random_augment(h,residual,random_seed)
    raise ValueError(mode)


def preference_terms(model,base,user,c,users,pos,neg,base_full_scores,alpha_bar,variant,ctx_pos=None,ctx_neg=None,return_outputs=False):
    if variant not in ('A1','A2','A3'): raise ValueError(variant)
    b=len(users); ids=torch.cat([pos,neg]); scorer=torch.cat([users,users]); icf=c['collab_item'][ids]; true_cf=c['collab_user'][scorer]
    wrong=shuffled_user_ids(scorer,model.n_users); wrong_cf=c['collab_user'][wrong]; zero_cf=torch.zeros_like(true_cf)
    ctx=None if ctx_pos is None else torch.cat([ctx_pos,ctx_neg])
    t=torch.full((2*b,),T_INFER,device=ids.device,dtype=torch.long)
    zt=torch.randn((2*b,LATENT_DIM),device=ids.device); zv=torch.randn((2*b,LATENT_DIM),device=ids.device)
    aug={}; minus={}; zeros=[]; bounds=[]; res={}; diags={}
    for name,denb,denu,h,noise in [('text',base.text,user.text,c['text_item'][ids],zt),('visual',base.visual,user.visual,c['image_item'][ids],zv)]:
        _,ht=noisy_state(h,t,alpha_bar,noise)
        with torch.no_grad(): pb=denb(ht,t,base_condition(base,true_cf,icf))
        pt=denu(ht,t,user_condition(user,true_cf,icf,ctx)); ps=denu(ht,t,user_condition(user,wrong_cf,icf,ctx)); pz=denu(ht,t,user_condition(user,zero_cf,icf,ctx))
        rt=pt-pb; rs=ps-pb; at,dt=_augment(h,rt,'plus'); ash,ds=_augment(h,rs,'plus'); am,dm=_augment(h,rt,'minus')
        aug[name]=(at,ash); minus[name]=am; zeros.append(F.mse_loss(pz,pb.detach()))
        bounds.extend([r15.excess_penalty(h,rt),r15.excess_penalty(h,rs)]); res[name]=(rt,rs); diags[name]=(dt,ds,dm)
    u=c['final_user'][scorer]
    base_back=(u*c['final_item'][ids]).sum(1)
    true_item=fuse_item(model,icf,aug['visual'][0],aug['text'][0],c['struct_item'][ids]); true_back=(u*true_item).sum(1)
    shuf_item=fuse_item(model,icf,aug['visual'][1],aug['text'][1],c['struct_item'][ids]); shuf_back=(u*shuf_item).sum(1)
    min_item=fuse_item(model,icf,minus['visual'],minus['text'],c['struct_item'][ids]); min_back=(u*min_item).sum(1)
    dtrue=true_back-base_back; dshuf=shuf_back-base_back; dminus=min_back-base_back
    # Round16R: compute correction margins DIRECTLY. Never form huge_base+tiny_delta then subtract base.
    sbase=base_full_scores
    mbase=sbase[:b]-sbase[b:]
    Dtrue=dtrue[:b]-dtrue[b:]
    Dshuf=dshuf[:b]-dshuf[b:]
    Dminus=dminus[:b]-dminus[b:]
    mtrue=mbase+Dtrue; mshuf=mbase+Dshuf; mminus=mbase+Dminus
    Lrec=F.softplus(-mtrue).mean(); Lgain=F.softplus(-Dtrue).mean(); Lid=F.softplus(Dshuf-Dtrue+USER_MARGIN).mean()
    Lzero=.5*(zeros[0]+zeros[1]); Lbound=.25*sum(bounds); Lrank=Lrec if variant=='A1' else Lgain
    total=Lrank+LAMBDA_ID*Lid+LAMBDA_ZERO*Lzero+LAMBDA_BOUND*Lbound
    lp=F.softplus(-mtrue); lb=F.softplus(-mbase); lm=F.softplus(-mminus)
    out={
        'L_rec':Lrec,'L_gain':Lgain,'L_identity':Lid,'L_zero':Lzero,'L_bound':Lbound,'total_loss':total,
        'm_base':mbase.mean(),'m_true':mtrue.mean(),'m_shuf':mshuf.mean(),
        'Delta_true_mean':Dtrue.mean(),'Delta_true_median':Dtrue.median(),
        'fraction_Delta_true_gt0':(Dtrue>0).float().mean(),
        'Delta_identity_mean':(Dtrue-Dshuf).mean(),
        'fraction_Delta_true_gt_shuf':(Dtrue>Dshuf).float().mean(),
        'L_plus':lp.mean(),'L_pair_base':lb.mean(),'L_minus':lm.mean(),
        'L_plus_minus_base':(lp-lb).mean(),'L_minus_minus_base':(lm-lb).mean(),
        'L_plus_minus_minus':(lp-lm).mean(),
        'fraction_Lplus_lt_base':(lp<lb).float().mean(),
        'fraction_Lplus_lt_minus':(lp<lm).float().mean(),
        'near_zero_action_fraction':(dtrue.abs()<1e-6).float().mean(),
        'small_action_fraction':(dtrue.abs()<1e-4).float().mean(),
        'correction_abs_mean':dtrue.abs().mean(),'correction_abs_max':dtrue.abs().max(),
        'raw_residual_text_true':res['text'][0].norm(dim=1).mean(),
        'raw_residual_visual_true':res['visual'][0].norm(dim=1).mean(),
        'cap_fraction_text':.5*(diags['text'][0]['cap_saturated'].float().mean()+diags['text'][1]['cap_saturated'].float().mean()),
        'cap_fraction_visual':.5*(diags['visual'][0]['cap_saturated'].float().mean()+diags['visual'][1]['cap_saturated'].float().mean()),
        'angle_text_mean':.5*(diags['text'][0]['angle_deg'].mean()+diags['text'][1]['angle_deg'].mean()),
        'angle_text_max':torch.maximum(diags['text'][0]['angle_deg'].max(),diags['text'][1]['angle_deg'].max()),
        'angle_visual_mean':.5*(diags['visual'][0]['angle_deg'].mean()+diags['visual'][1]['angle_deg'].mean()),
        'angle_visual_max':torch.maximum(diags['visual'][0]['angle_deg'].max(),diags['visual'][1]['angle_deg'].max()),
        'preference_t_min':t.min().detach(),'preference_t_max':t.max().detach()
    }
    if return_outputs:
        out.update({'Dtrue_vec':Dtrue,'Dshuf_vec':Dshuf,'Dminus_vec':Dminus,'mtrue_vec':mtrue,'mbase_vec':mbase,'mminus_vec':mminus,
                    'dtrue_item':dtrue,'dshuf_item':dshuf,'rt':res['text'][0],'rv':res['visual'][0]})
    return total,out


@torch.no_grad()
def inference_residuals(base,user,c,condition_user_ids,item_ids,ctx,alpha_bar,noise_tables):
    ucf=c['collab_user'][condition_user_ids]; icf=c['collab_item'][item_ids]; zcf=torch.zeros_like(ucf)
    t=torch.full((len(item_ids),),T_INFER,device=item_ids.device,dtype=torch.long)
    out={'text':[],'visual':[]}
    for seed in INFER_SEEDS:
        for name,denb,denu,h in [('text',base.text,user.text,c['text_item'][item_ids]),('visual',base.visual,user.visual,c['image_item'][item_ids])]:
            noise=noise_tables[(seed,name)][item_ids]; _,ht=noisy_state(h,t,alpha_bar,noise)
            pb=denb(ht,t,base_condition(base,zcf,icf))
            pu=denu(ht,t,user_condition(user,ucf,icf,ctx))
            out[name].append(pu-pb)
    return torch.stack(out['text']).mean(0),torch.stack(out['visual']).mean(0)

@torch.no_grad()
def pair_scores(model,base,user,c,scorer_user_ids,item_ids,ctx,alpha_bar,noise_tables,condition_mode='true',control_seed=None,return_diag=False):
    base_back=(c['final_user'][scorer_user_ids]*c['final_item'][item_ids]).sum(1)
    if condition_mode=='zero':
        z=torch.zeros((len(item_ids),LATENT_DIM),device=item_ids.device); dz=torch.zeros(len(item_ids),device=item_ids.device)
        diag={'text':{'angle_deg':dz,'cap_saturated':dz.bool(),'raw_tangent_norm':dz,'bounded_tangent_norm':dz},
              'visual':{'angle_deg':dz,'cap_saturated':dz.bool(),'raw_tangent_norm':dz,'bounded_tangent_norm':dz}}
        return (base_back,base_back,z,z,diag) if return_diag else (base_back,base_back,z,z)
    cond=scorer_user_ids if condition_mode in ('true','negated','random') else shuffled_user_ids(scorer_user_ids,model.n_users)
    if condition_mode not in ('true','shuffled','negated','random'): raise ValueError(condition_mode)
    rt,rv=inference_residuals(base,user,c,cond,item_ids,ctx,alpha_bar,noise_tables)
    mode='plus' if condition_mode in ('true','shuffled') else ('minus' if condition_mode=='negated' else 'random')
    at,dt=_augment(c['text_item'][item_ids],rt,mode,None if control_seed is None else int(control_seed)+17)
    av,dv=_augment(c['image_item'][item_ids],rv,mode,None if control_seed is None else int(control_seed)+31)
    aug=fuse_item(model,c['collab_item'][item_ids],av,at,c['struct_item'][item_ids]); aug_back=(c['final_user'][scorer_user_ids]*aug).sum(1)
    if return_diag: return base_back,aug_back,rt,rv,{'text':dt,'visual':dv}
    return base_back,aug_back,rt,rv
