from __future__ import annotations
import math
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.preprocessing import normalize

from diffusion_experiments.round27_local_preference_field_diffusion import round27_core as c
from diffusion_experiments.round26_preference_directed_diffusion import round26_diffusion as d26
from diffusion_experiments.round22_corrected_online_gdnsm import round22_core as r22
from modules.attribute import build_item_matrices, build_profiles, attribute_z, _sparse_candidate_scores
from modules.coliftrec import fit_backgrounds
from modules.ranking import histories_csr, l2_normalize_rows, semantic_z_for_candidates, topk_from_embeddings
from pipelines.dataset_config import load_dataset_config
from pipelines.coliftrec import _params


def _masked_z(x,mask,eps=1e-8):
    x=np.asarray(x,np.float32); m=np.asarray(mask,bool); n=m.sum(1,keepdims=True).astype(np.float32)
    den=np.maximum(n,1.0); mean=(x*m).sum(1,keepdims=True)/den
    var=(((x-mean)*m)**2).sum(1,keepdims=True)/den; std=np.sqrt(np.maximum(var,eps**2))
    return (((x-mean)/std)*m).astype(np.float32)

def _masked_softmax(a,mask):
    a=np.asarray(a,np.float32); m=np.asarray(mask,bool); out=np.zeros_like(a)
    for r in range(len(a)):
        ix=np.flatnonzero(m[r])
        if len(ix):
            v=a[r,ix]; e=np.exp(v-v.max()); out[r,ix]=e/e.sum()
    return out

def _dense_scores(feat,profiles,items,batch=256):
    out=np.zeros(items.shape,np.float32)
    for st in range(0,len(items),batch):
        en=min(st+batch,len(items)); out[st:en]=np.einsum('bkd,bd->bk',feat[items[st:en]],profiles[st:en],optimize=True)
    return out

def _norm_rows(x):
    return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)

class LPFContext:
    """TRAIN-only static multimodal context. No Validation/Test labels are consumed."""
    def __init__(self,seed,model):
        self.seed=int(seed); self.cfg=load_dataset_config('baby'); p=self.cfg['resolved_paths']; self.paths=p
        self.ccfg=self.cfg['coliftrec']; self.params=_params(self.ccfg); self.n_items=model.n_items; self.n_users=model.n_users
        df=pd.read_csv(p['interaction'],sep='\t',usecols=['userID','itemID','timestamp','x_label']); df['_row']=np.arange(len(df),dtype=np.int64)
        tr=df[df.x_label==0].copy().sort_values(['userID','timestamp','_row'],kind='stable')
        self.histories=[[] for _ in range(self.n_users)]
        for u,g in tr.groupby('userID',sort=False): self.histories[int(u)]=g.itemID.astype(np.int64).tolist()
        self.counts=np.asarray([len(h) for h in self.histories],np.int64)
        self.pseudo=[list(h[:-1]) if len(h)>=2 else list(h) for h in self.histories]
        self.pusers=np.asarray([u for u,h in enumerate(self.histories) if len(h)>=2],np.int64)
        self.text=l2_normalize_rows(np.asarray(np.load(p['text_feature'],mmap_mode='r'),dtype=np.float32))
        self.visual=l2_normalize_rows(np.asarray(np.load(p['visual_feature'],mmap_mode='r'),dtype=np.float32))
        ac=self.ccfg['attribute']; self.attr_weights=ac.get('weights') or {'title':.45,'brand':.20,'description':.35}
        self.mats,self.attr_audit=build_item_matrices(p['metadata'],self.n_items,min_df=int(ac.get('tfidf_min_df',2)),max_df=float(ac.get('tfidf_max_df',.8)),description_len=int(ac.get('description_len',128)),weights=self.attr_weights)
        # Frozen generic background from TRAIN pseudo-history Top100 only.
        pseudo_profiles=build_profiles(self.mats,self.pseudo,self.n_items)
        model.eval()
        with torch.no_grad(): fu,fi=model.forward(test=True); pi,_=topk_from_embeddings(fu,fi,self.pusers,self.pseudo,100,1024)
        zt,_=semantic_z_for_candidates(p['text_feature'],self.pseudo,self.pusers,pi,256)
        zv,_=semantic_z_for_candidates(p['visual_feature'],self.pseudo,self.pusers,pi,128)
        za,_=attribute_z(self.mats,pseudo_profiles,self.pusers,pi,256,self.attr_weights)
        self.bg=fit_backgrounds(pi,zt,za,zv,self.n_items)
        # Sums permit exact H_u^{-i} profiles without using the current label in target construction.
        H=histories_csr(self.histories,self.n_items,mean=False)
        self.text_sum=np.asarray(H@self.text,dtype=np.float32); self.visual_sum=np.asarray(H@self.visual,dtype=np.float32)
        self.attr_sum={f:(H@self.mats[f]).tocsr() for f in ('title','brand','description')}
        self.audit={'seed':self.seed,'TRAIN_ONLY':True,'train_interactions':int(self.counts.sum()),'users_with_history':int((self.counts>0).sum()),'K_HISTORY':c.K_HISTORY,'MIN_HISTORY':c.MIN_HISTORY,'colift':self.params.to_dict(),'attribute_weights':self.attr_weights,'background_definition':'TRAIN pseudo-history Top100; no Validation/Test interactions','current_positive_excluded_from_target':True,'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}

    def _hminus(self,users,pos):
        hs=[]; occ=np.zeros(len(users),np.int64)
        for r,(u,p) in enumerate(zip(users,pos)):
            h=self.histories[int(u)]; q=[int(x) for x in h if int(x)!=int(p)]; hs.append(q); occ[r]=len(h)-len(q)
        return hs,occ

    def _locality_support(self,hm,anchors):
        B=len(hm); lengths=np.asarray([len(x) for x in hm],np.int64); offsets=np.zeros(B+1,np.int64); np.cumsum(lengths,out=offsets[1:])
        sup=np.zeros((B,c.K_HISTORY),np.int64); mask=np.zeros((B,c.K_HISTORY),bool); loc=np.zeros((B,c.K_HISTORY),np.float32)
        if offsets[-1]==0: return sup,mask,loc,lengths
        flat=np.concatenate([np.asarray(x,np.int64) for x in hm if x]); rows=np.repeat(np.arange(B,dtype=np.int64),lengths); ar=anchors[rows]
        ct=np.einsum('ij,ij->i',self.text[flat],self.text[ar],optimize=True).astype(np.float32)
        cv=np.empty(len(flat),np.float32)
        for st in range(0,len(flat),2048):
            en=min(st+2048,len(flat)); cv[st:en]=np.einsum('ij,ij->i',self.visual[flat[st:en]],self.visual[ar[st:en]],optimize=True)
        ca=np.zeros(len(flat),np.float32)
        for f,w in self.attr_weights.items():
            ca += float(w)*np.asarray(self.mats[f][flat].multiply(self.mats[f][ar]).sum(axis=1)).reshape(-1).astype(np.float32)
        cc=(ct+cv+ca)/3.0
        for r in range(B):
            st,en=offsets[r],offsets[r+1]
            if st==en: continue
            ids=flat[st:en]; val=cc[st:en]; order=np.lexsort((ids,-val))[:c.K_HISTORY]; n=len(order)
            sup[r,:n]=ids[order]; mask[r,:n]=True; loc[r,:n]=val[order]
        return sup,mask,loc,lengths

    def _profiles_and_lift(self,users,pos,occ,support,mask):
        B=len(users); cnt=(self.counts[users]-occ).astype(np.int64)
        tp=self.text_sum[users]-occ[:,None].astype(np.float32)*self.text[pos]; vp=self.visual_sum[users]-occ[:,None].astype(np.float32)*self.visual[pos]
        tp=_norm_rows(tp); vp=_norm_rows(vp)
        raw_t=_dense_scores(self.text,tp,support); raw_v=_dense_scores(self.visual,vp,support)
        field_raw={}
        urow=np.arange(B,dtype=np.int64)
        for f in ('title','brand','description'):
            sub=self.mats[f][pos].multiply(occ[:,None]); prof=(self.attr_sum[f][users]-sub).tocsr()
            if f in ('title','description'): prof=normalize(prof,norm='l2',axis=1,copy=False)
            else:
                scale=np.divide(1.0,np.maximum(cnt,1),dtype=np.float32); prof=prof.multiply(scale[:,None]).tocsr()
            field_raw[f]=_sparse_candidate_scores(prof,self.mats[f],urow,support,batch=256)
        fz={f:_masked_z(v,mask) for f,v in field_raw.items()}
        raw_a=sum(float(self.attr_weights[f])*fz[f] for f in fz); z_a=_masked_z(raw_a,mask)
        z_t=_masked_z(raw_t,mask); z_v=_masked_z(raw_v,mask)
        pt=self.params
        lt=_masked_z(z_t-float(pt.lambda_text)*self.bg['text']['shrunk_mean'][support],mask)
        la=_masked_z(z_a-float(pt.lambda_attribute)*self.bg['attribute']['shrunk_mean'][support],mask)
        lv=_masked_z(z_v-float(pt.lambda_visual)*self.bg['visual']['shrunk_mean'][support],mask)
        lv=_masked_z(z_v-float(pt.lambda_visual)*self.bg['visual']['shrunk_mean'][support],mask)
        q=float(pt.alpha_text)*lt+float(pt.alpha_attribute)*la+float(pt.alpha_visual)*lv
        return q,{'z_text':z_t,'z_attribute':z_a,'z_visual':z_v,'lift_text':lt,'lift_attribute':la,'lift_visual':lv}

    def targets(self,b,users_t,pos_t,target_ids):
        users=users_t.detach().cpu().numpy().astype(np.int64)
        pos=pos_t.detach().cpu().numpy().astype(np.int64)
        hm,occ=self._hminus(users,pos); packs=[]
        for band,ids_t in enumerate(target_ids):
            anchors=ids_t.detach().cpu().numpy().astype(np.int64)
            support,mask,locality,hist_avail=self._locality_support(hm,anchors)
            q,mods=self._profiles_and_lift(users,pos,occ,support,mask)
            zl=_masked_z(locality,mask); zq=_masked_z(q,mask); a=.5*(zl+zq); weights=_masked_softmax(a,mask)
            dev=b['fi'].device; sid=torch.as_tensor(support,device=dev,dtype=torch.long); mt=torch.as_tensor(mask,device=dev); wt=torch.as_tensor(weights,device=dev)
            anchor=b['fi'][ids_t].detach(); sh=b['fi'][sid].detach(); v=sh-anchor[:,None,:]
            denom=(anchor*anchor).sum(1,keepdim=True).unsqueeze(2)+c.EPS
            tan=v-((v*anchor[:,None,:]).sum(2,keepdim=True)/denom)*anchor[:,None,:]
            dn=tan.norm(dim=2,keepdim=True); dirs=tan/(dn+c.EPS); dirs=dirs*mt.unsqueeze(2)
            vf=(dirs*wt.unsqueeze(2)).sum(1); rho=vf.norm(dim=1); dfield=vf/(rho.unsqueeze(1)+c.EPS)
            n=mt.sum(1); eligible=(n>=c.MIN_HISTORY)&(rho>c.EPS)
            conf=rho*(n.float()/float(c.K_HISTORY)); conf=torch.where(eligible,conf,torch.zeros_like(conf)).clamp(0,1)
            dfield=torch.where(eligible[:,None],dfield,torch.zeros_like(dfield))
            arrn=n.detach().cpu().numpy()
            diag={'band':c.BAND_NAMES[band],'history_available':hist_avail,'support_size':arrn,'semantic_locality':locality[mask],'personalized_lift':q[mask],'rho':rho.detach().cpu().numpy(),'confidence':conf.detach().cpu().numpy(),'weights_sum':weights.sum(1),'history_ge3':(arrn>=c.MIN_HISTORY),'history_eq5':(arrn==c.K_HISTORY),'no_refinement':(~eligible).detach().cpu().numpy(),'current_positive_excluded':np.asarray([int(p) not in set(support[r,mask[r]].tolist()) for r,p in enumerate(pos)],bool)}
            packs.append({'direction':dfield.detach(),'confidence':conf.detach(),'support_ids':sid,'support_mask':mt,'support_weights':wt,'support_dirs':dirs.detach(),'diag':diag,'mods':mods})
        return packs
def _concat_active(b,interaction,s,target_ids):
    users=interaction[0]; B=len(users); active=len(target_ids)
    users_all=torch.cat([users for _ in range(active)],0); ids_all=torch.cat(target_ids,0)
    x0=c.z(b['fi'][ids_all].detach(),s['muI'],s['rI']); hu=c.z(b['fu'][users_all].detach(),s['muU'],s['rU'])
    tc=c.z(b['text_item'][ids_all].detach(),s['muT'],s['rT']); vc=c.z(b['image_item'][ids_all].detach(),s['muV'],s['rV'])
    shuf=torch.cat([hu[j*B:(j+1)*B].roll(1,0) for j in range(active)],0)
    anchor=b['fi'][ids_all].detach(); user_raw=b['fu'][users_all].detach(); pos=torch.cat([b['fi'][interaction[1]].detach() for _ in range(active)],0)
    return users_all,ids_all,x0,hu,shuf,tc,vc,user_raw,anchor,pos,B

def warmup_step(net,opt,sched,b,interaction,s,seed):
    return d26.warmup_step(net,opt,sched,b,interaction,s,seed)

def field_inner_step(net,opt,sched,b,interaction,s,target_ids,packs,seed):
    rng=r22.capture_rng(); g=torch.Generator(device=b['fu'].device); g.manual_seed(int(seed))
    _,_,x0,hu,shuf,tc,vc,_,anchor,_,B=_concat_active(b,interaction,s,target_ids)
    td=torch.cat([p['direction'] for p in packs],0).detach(); cf=torch.cat([p['confidence'] for p in packs],0).detach()
    xt,t,eps=d26.local_xt(sched,x0,g); net.train(); opt.zero_grad(set_to_none=True)
    eb,et,es,zb,zt,zs=d26.branches(net,sched,xt,t,hu,shuf,tc,vc)
    hb=c.inv_item(zb,s); ht=c.inv_item(zt,s); rt=d26.tangent(anchor,ht-hb); dd,n=d26.unit(rt)
    cos=(dd*td).sum(1).clamp(-1,1); valid=(cf>0)&(n>c.EPS)
    lv=cf*(1-cos); lfield=lv[valid].mean() if bool(valid.any()) else lv.mean()*0
    lrec=((eb-eps).pow(2).mean()+(et-eps).pow(2).mean()+(es-eps).pow(2).mean())/3.; total=lrec+lfield
    if not torch.isfinite(total): raise RuntimeError('nonfinite Round27 diffusion loss')
    total.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step(); r22.restore_rng(rng)
    return {'total':float(total.detach()),'rec':float(lrec.detach()),'field':float(lfield.detach()),'valid_field_fraction':float(valid.float().mean().detach()),'t_min':int(t.min()),'t_max':int(t.max())}

@torch.no_grad()
def predict_directions(net,sched,b,interaction,s,target_ids,packs,seed):
    g=torch.Generator(device=b['fu'].device); g.manual_seed(int(seed))
    _,_,x0,hu,shuf,tc,vc,_,anchor,_,B=_concat_active(b,interaction,s,target_ids)
    td=torch.cat([p['direction'] for p in packs],0).detach(); cf=torch.cat([p['confidence'] for p in packs],0).detach()
    xt,t,eps=d26.local_xt(sched,x0,g); net.eval(); eb,et,es,zb,zt,zs=d26.branches(net,sched,xt,t,hu,shuf,tc,vc)
    hb=c.inv_item(zb,s); ht=c.inv_item(zt,s); hs=c.inv_item(zs,s)
    rt=d26.tangent(anchor,ht-hb); rs=d26.tangent(anchor,hs-hb); dt,nt=d26.unit(rt); ds,ns=d26.unit(rs)
    at=(dt*td).sum(1).clamp(-1,1); ash=(ds*td).sum(1).clamp(-1,1); da=at-ash
    rec=((eb-eps).pow(2).mean(1)+(et-eps).pow(2).mean(1)+(es-eps).pow(2).mean(1))/3.
    dirs=[]; diags=[]
    for j in range(len(target_ids)):
        sl=slice(j*B,(j+1)*B); dirs.append(dt[sl].detach()); diags.append({'band':c.BAND_NAMES[j],'A_true':at[sl].cpu().numpy(),'A_shuf':ash[sl].cpu().numpy(),'DeltaA':da[sl].cpu().numpy(),'raw_residual_norm':nt[sl].cpu().numpy(),'shuffled_raw_residual_norm':ns[sl].cpu().numpy(),'near_zero_residual':(nt[sl]<c.EPS).cpu().numpy(),'reconstruction_mse':rec[sl].cpu().numpy(),'confidence':cf[sl].cpu().numpy(),'t':t[sl].cpu().numpy()})
    return dirs,diags

@torch.no_grad()
def calibrate(user,pos,anchor,direction,confidence):
    d=direction.detach(); h=anchor.detach(); u=user.detach(); p=pos.detach(); conf=confidence.detach().clamp(0,1)
    sp=(u*p).sum(1); sn=(u*h).sum(1); margin=sp-sn; shift=conf*c.HALF_MARGIN*torch.clamp(margin,min=0); a=(u*d).sum(1)
    eligible=(conf>0)&(margin>0)&(a>0); req=torch.zeros_like(shift); req[eligible]=shift[eligible]/(a[eligible]+c.EPS)
    lmax=math.tan(math.radians(c.THETA_MAX_DEG))*h.norm(dim=1); lam=torch.where(eligible,torch.minimum(req,lmax),torch.zeros_like(req)); delta=(lam[:,None]*d).detach(); refined=h+delta
    sr=(u*refined).sum(1); actual=sr-sn; target=sn+shift; angle=torch.rad2deg(torch.acos(F.cosine_similarity(h,refined,dim=1).clamp(-1,1)))
    cap=eligible&(req>lmax+1e-10); reach=eligible&(req<=lmax+1e-10)
    diag={'positive_margin':(margin>0).cpu().numpy(),'confidence':conf.cpu().numpy(),'direction_positive':(a>0).cpu().numpy(),'direction_miss':((conf>0)&(margin>0)&(a<=0)).cpu().numpy(),'applied_refinement':(lam>0).cpu().numpy(),'requested_score_shift':shift.cpu().numpy(),'actual_score_shift':actual.cpu().numpy(),'target_hit_abs_error':(sr-target).abs().cpu().numpy(),'target_reachable':reach.cpu().numpy(),'cap_hit':cap.cpu().numpy(),'angle_deg':angle.cpu().numpy(),'real_negative_margin':margin.cpu().numpy(),'refined_negative_margin':(sp-sr).cpu().numpy(),'G_hard':actual.cpu().numpy(),'target_never_exceeds_positive':((margin<=0)|(target<=sp+1e-7)).cpu().numpy()}
    return delta,diag

def summarize_lpf(packs_batches):
    rows=[p['diag'] for batch in packs_batches for p in batch]
    if not rows:return {}
    cat=lambda k:np.concatenate([np.asarray(r[k]).reshape(-1) for r in rows])
    return {'support_size':c.stats(cat('support_size')),'history_ge3_fraction':float(cat('history_ge3').mean()),'history_eq5_fraction':float(cat('history_eq5').mean()),'no_refinement_fraction':float(cat('no_refinement').mean()),'semantic_locality':c.stats(cat('semantic_locality')),'personalized_lift':c.stats(cat('personalized_lift')),'rho':c.stats(cat('rho')),'confidence':c.stats(cat('confidence')),'weights_sum':c.stats(cat('weights_sum')),'current_positive_excluded_all':bool(cat('current_positive_excluded').all())}

def summarize_mechanism(packs_batches,dir_batches,cal_batches,diff_losses=None):
    out={'LPF':summarize_lpf(packs_batches)}; cr=[r for batch in cal_batches for r in batch]; catc=lambda k:np.concatenate([r[k] for r in cr])
    out.update({'applied_refinement_fraction':float(catc('applied_refinement').mean()),'requested_score_shift':c.stats(catc('requested_score_shift')),'actual_score_shift':c.stats(catc('actual_score_shift')),'target_hit_abs_error':c.stats(catc('target_hit_abs_error')),'target_reachable_fraction':float(catc('target_reachable').mean()),'cap_hit_fraction':float(catc('cap_hit').mean()),'angle_deg':c.stats(catc('angle_deg')),'real_negative_margin':c.stats(catc('real_negative_margin')),'refined_negative_margin':c.stats(catc('refined_negative_margin')),'G_hard':c.stats(catc('G_hard')),'target_never_exceeds_positive':bool(catc('target_never_exceeds_positive').all())}); out['angle_bound_pass']=bool(out['angle_deg']['max']<=5.01)
    if dir_batches:
        dr=[r for batch in dir_batches for r in batch]; catd=lambda k:np.concatenate([r[k] for r in dr]); da=catd('DeltaA')
        out.update({'A_true':c.stats(catd('A_true')),'A_shuf':c.stats(catd('A_shuf')),'DeltaA':dict(c.stats(da),p_gt_zero=float((da>0).mean())),'raw_diffusion_residual_norm':c.stats(catd('raw_residual_norm')),'near_zero_residual_fraction':float(catd('near_zero_residual').mean()),'reconstruction_mse':c.stats(catd('reconstruction_mse'))})
    if diff_losses:
        out['L_total']=c.stats([x['total'] for x in diff_losses]); out['L_rec']=c.stats([x['rec'] for x in diff_losses]); out['L_field']=c.stats([x['field'] for x in diff_losses])
    return out
