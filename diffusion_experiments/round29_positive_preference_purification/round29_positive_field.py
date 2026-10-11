from __future__ import annotations
import gc, hashlib, math
from collections import defaultdict
import numpy as np
import torch
from sklearn.preprocessing import normalize

from diffusion_experiments.round29_positive_preference_purification import round29_core as c
from diffusion_experiments.round29_positive_preference_purification.round29_colift_reference import FrozenCoLiftReference
from modules.attribute import build_item_matrices
from modules.ranking import histories_csr, l2_normalize_rows
from pipelines.coliftrec import _params
from pipelines.dataset_config import load_dataset_config

def _norm_rows(x): return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
def _ref_z(ref,sup):
    mu=ref.mean(1,keepdims=True); sd=np.maximum(ref.std(1,keepdims=True),1e-8)
    return ((ref-mu)/sd).astype(np.float32),((sup-mu)/sd).astype(np.float32)
def _sparse_scores(profile,item_matrix,items):
    B,K=items.shape; rr=np.repeat(np.arange(B),K); cand=item_matrix[items.reshape(-1)]
    return np.asarray(profile[rr].multiply(cand).sum(axis=1)).reshape(B,K).astype(np.float32)
def _masked_softmax_z(values,mask):
    out=np.zeros_like(values,np.float32)
    for r in range(len(values)):
        ix=np.flatnonzero(mask[r])
        if len(ix):
            v=values[r,ix]; z=(v-v.mean())/max(float(v.std()),1e-8); e=np.exp(z-z.max()); out[r,ix]=e/e.sum()
    return out

class PositiveFieldBank:
    """Event-level positive-conditioned local supports and canonical-context CoLift gate."""
    def __init__(self,seed,model,events,reference=None,force_rebuild=False,path_override=None):
        self.seed=int(seed); self.model=model; self.events=events
        self.ref=reference or FrozenCoLiftReference(seed,model,events)
        self.path=path_override or (c.ASSETS/f'ROUND29_POSITIVE_FIELD_BANK_SEED{seed}.npz')
        self.cfg=load_dataset_config('baby'); self.paths=self.cfg['resolved_paths']; self.ccfg=self.cfg['coliftrec']; self.params=_params(self.ccfg)
        if force_rebuild or not self.path.exists(): self._build()
        z=np.load(self.path)
        self.event_users=z['event_users'].astype(np.int64); self.event_pos=z['event_pos'].astype(np.int64)
        self.support=z['support_ids'].astype(np.int64); self.locality=z['locality'].astype(np.float32)
        self.local_mask=z['local_mask'].astype(bool); self.q=z['q'].astype(np.float32); self.pref_mask=z['pref_mask'].astype(bool)
        self.history_size=z['history_size'].astype(np.int64)
        self.event_groups=defaultdict(list)
        for i,(u,p) in enumerate(zip(self.event_users.tolist(),self.event_pos.tolist())): self.event_groups[(int(u),int(p))].append(i)
        self._counts=None
    def reset_epoch(self): self._counts=defaultdict(int)
    def resolve_event_ids(self,users_t,pos_t):
        if self._counts is None:self.reset_epoch()
        us=users_t.detach().cpu().numpy().astype(np.int64); ps=pos_t.detach().cpu().numpy().astype(np.int64); out=np.empty(len(us),np.int64)
        for r,(u,p) in enumerate(zip(us,ps)):
            ids=self.event_groups[(int(u),int(p))]; k=self._counts[(int(u),int(p))]
            if k>=len(ids): raise RuntimeError('event duplicate count exceeded')
            out[r]=ids[k]; self._counts[(int(u),int(p))]=k+1
        return out
    def _build(self):
        p=self.paths; histories=self.events.histories; N=len(self.events.u); n_items=self.model.n_items
        text=l2_normalize_rows(np.asarray(np.load(p['text_feature'],mmap_mode='r'),dtype=np.float32))
        visual=l2_normalize_rows(np.asarray(np.load(p['visual_feature'],mmap_mode='r'),dtype=np.float32))
        ac=self.ccfg['attribute']; weights=ac.get('weights') or {'title':.45,'brand':.20,'description':.35}
        mats,_=build_item_matrices(p['metadata'],n_items,min_df=int(ac.get('tfidf_min_df',2)),max_df=float(ac.get('tfidf_max_df',.8)),description_len=int(ac.get('description_len',128)),weights=weights)
        counts=np.asarray([len(h) for h in histories],np.int64); H=histories_csr(histories,n_items,mean=False)
        tsum=np.asarray(H@text,dtype=np.float32); vsum=np.asarray(H@visual,dtype=np.float32); asum={f:(H@mats[f]).tocsr() for f in ('title','brand','description')}
        support=np.zeros((N,c.K_HISTORY),np.int32); locality=np.zeros((N,c.K_HISTORY),np.float32); local_mask=np.zeros((N,c.K_HISTORY),bool)
        qout=np.zeros((N,c.K_HISTORY),np.float32); pref=np.zeros((N,c.K_HISTORY),bool); hsize=np.zeros(N,np.int16)
        refids=self.ref.candidates_for_users(self.events.u)
        bs=256
        for st in range(0,N,bs):
            en=min(st+bs,N); users=self.events.u[st:en].astype(np.int64); pos=self.events.p[st:en].astype(np.int64); B=len(users)
            hm=[]; occ=np.zeros(B,np.int64)
            for r,(u,pp) in enumerate(zip(users,pos)):
                h=histories[int(u)]; qq=[int(x) for x in h if int(x)!=int(pp)]; hm.append(qq); occ[r]=len(h)-len(qq); hsize[st+r]=len(qq)
            # positive-conditioned semantic TopK
            lengths=np.asarray([len(x) for x in hm],np.int64); offsets=np.zeros(B+1,np.int64); np.cumsum(lengths,out=offsets[1:])
            if offsets[-1]:
                flat=np.concatenate([np.asarray(x,np.int64) for x in hm if x]); rows=np.repeat(np.arange(B,dtype=np.int64),lengths); ar=pos[rows]
                ct=np.einsum('ij,ij->i',text[flat],text[ar],optimize=True).astype(np.float32)
                cv=np.einsum('ij,ij->i',visual[flat],visual[ar],optimize=True).astype(np.float32)
                ca=np.zeros(len(flat),np.float32)
                for f,w in weights.items(): ca+=float(w)*np.asarray(mats[f][flat].multiply(mats[f][ar]).sum(axis=1)).reshape(-1).astype(np.float32)
                loc=(ct+cv+ca)/3.0
                for r in range(B):
                    a,b=offsets[r],offsets[r+1]
                    if a==b:continue
                    ids=flat[a:b]; vv=loc[a:b]; order=np.lexsort((ids,-vv))[:c.K_HISTORY]; nn=len(order)
                    support[st+r,:nn]=ids[order]; locality[st+r,:nn]=vv[order]; local_mask[st+r,:nn]=True
            sup=support[st:en].astype(np.int64); mask=local_mask[st:en]; refs=refids[st:en].astype(np.int64)
            cnt=(counts[users]-occ).astype(np.int64)
            tp=_norm_rows(tsum[users]-occ[:,None].astype(np.float32)*text[pos]); vp=_norm_rows(vsum[users]-occ[:,None].astype(np.float32)*visual[pos])
            rt=np.einsum('bkd,bd->bk',text[refs],tp,optimize=True).astype(np.float32); stt=np.einsum('bkd,bd->bk',text[sup],tp,optimize=True).astype(np.float32)
            rv=np.einsum('bkd,bd->bk',visual[refs],vp,optimize=True).astype(np.float32); sv=np.einsum('bkd,bd->bk',visual[sup],vp,optimize=True).astype(np.float32)
            ztr,zts=_ref_z(rt,stt); zvr,zvs=_ref_z(rv,sv)
            field_ref={}; field_sup={}
            for f in ('title','brand','description'):
                prof=(asum[f][users]-mats[f][pos].multiply(occ[:,None])).tocsr()
                if f in ('title','description'): prof=normalize(prof,norm='l2',axis=1,copy=False)
                else: prof=prof.multiply((1.0/np.maximum(cnt,1)).astype(np.float32)[:,None]).tocsr()
                rr=_sparse_scores(prof,mats[f],refs); ss=_sparse_scores(prof,mats[f],sup); zr,zs=_ref_z(rr,ss); field_ref[f]=zr; field_sup[f]=zs
            ar=sum(float(weights[f])*field_ref[f] for f in field_ref); ass=sum(float(weights[f])*field_sup[f] for f in field_sup); zar,zas=_ref_z(ar,ass)
            pt=self.params; lm={'text':float(pt.lambda_text),'attribute':float(pt.lambda_attribute),'visual':float(pt.lambda_visual)}
            al={'text':float(pt.alpha_text),'attribute':float(pt.alpha_attribute),'visual':float(pt.alpha_visual)}
            q=np.zeros((B,c.K_HISTORY),np.float32)
            for name,zr,zs in (('text',ztr,zts),('attribute',zar,zas),('visual',zvr,zvs)):
                rawr=zr-lm[name]*self.ref.bg[name]['shrunk_mean'][refs]; raws=zs-lm[name]*self.ref.bg[name]['shrunk_mean'][sup]; _,Ls=_ref_z(rawr,raws); q+=al[name]*Ls
            q[~mask]=0; qout[st:en]=q
            for r in range(B):
                ix=np.flatnonzero(mask[r])
                if len(ix)<c.MIN_HISTORY:continue
                nkeep=max(3,int(math.ceil(len(ix)/2))); order=np.lexsort((sup[r,ix],-q[r,ix]))[:nkeep]; pref[st+r,ix[order]]=True
            if (st//bs)%50==0: print({'phase':'field_bank','seed':self.seed,'done':en,'total':N},flush=True)
        np.savez_compressed(self.path,event_users=self.events.u.astype(np.int32),event_pos=self.events.p.astype(np.int32),
                            support_ids=support,locality=locality,local_mask=local_mask,q=qout,pref_mask=pref,history_size=hsize)
        del text,visual,mats,H,tsum,vsum,asum; gc.collect()
    def bank_hash(self):
        h=hashlib.sha256()
        for x in (self.event_users,self.event_pos,self.support,self.locality,self.local_mask,self.q,self.pref_mask,self.history_size):
            h.update(np.ascontiguousarray(x).tobytes())
        return h.hexdigest()
    def batch_static(self,event_ids,device):
        e=np.asarray(event_ids,np.int64)
        return {'support_ids':torch.as_tensor(self.support[e],device=device,dtype=torch.long),
                'locality':self.locality[e],'local_mask':torch.as_tensor(self.local_mask[e],device=device),
                'q':self.q[e],'pref_mask':torch.as_tensor(self.pref_mask[e],device=device),
                'history_size':self.history_size[e]}

def dynamic_field(b,pos_t,static):
    sid=static['support_ids']; pm=static['pref_mask']; anchor=b['fi'][pos_t].detach(); sh=b['fi'][sid].detach()
    v=sh-anchor[:,None,:]; den=(anchor*anchor).sum(1,keepdim=True).unsqueeze(2)+c.EPS; tan=v-((v*anchor[:,None,:]).sum(2,keepdim=True)/den)*anchor[:,None,:]
    dirs=tan/(tan.norm(dim=2,keepdim=True)+c.EPS); dirs=dirs*pm.unsqueeze(2)
    pref=pm.detach().cpu().numpy(); dirs_np=dirs.detach().cpu().numpy(); coherent=np.zeros_like(pref); coherence=np.zeros_like(static['locality'],np.float32)
    for r in range(len(dirs_np)):
        ix=np.flatnonzero(pref[r])
        if len(ix)<c.MIN_HISTORY:continue
        dd=dirs_np[r,ix]; sim=dd@dd.T; score=np.maximum(sim,0).mean(1); med=int(np.argmax(score)); coh=np.maximum(sim[:,med],0).astype(np.float32); keep=ix[coh>0]; coherent[r,keep]=True; coherence[r,ix]=coh
    sem=_masked_softmax_z(static['locality'],coherent); weights=sem*coherence; dw=weights.sum(1,keepdims=True); weights=np.divide(weights,np.maximum(dw,1e-12),out=np.zeros_like(weights),where=dw>0)
    wt=torch.as_tensor(weights,device=anchor.device); cm=torch.as_tensor(coherent,device=anchor.device); vf=(dirs*wt.unsqueeze(2)).sum(1); rho=vf.norm(dim=1); dfield=vf/(rho[:,None]+c.EPS)
    ncoh=cm.sum(1); eligible=(torch.as_tensor(static['history_size'],device=anchor.device)>=c.MIN_HISTORY)&(ncoh>=c.MIN_COHERENT)&(rho>c.EPS)
    conf=(rho*torch.clamp(ncoh.float()/4.0,max=1.0)).clamp(0,1); conf=torch.where(eligible,conf,torch.zeros_like(conf)); dfield=torch.where(eligible[:,None],dfield,torch.zeros_like(dfield))
    return {'direction':dfield.detach(),'confidence':conf.detach(),'support_ids':sid,'support_mask':cm,'support_weights':wt,'support_dirs':dirs.detach(),
            'diag':{'history_size':static['history_size'],'local_support_count':static['local_mask'].detach().cpu().numpy().sum(1),'pref_support_count':pref.sum(1),'coherent_support_count':coherent.sum(1),
                    'rho':rho.detach().cpu().numpy(),'confidence':conf.detach().cpu().numpy(),'weights_sum':weights.sum(1),'q':static['q'][static['local_mask'].detach().cpu().numpy()],
                    'locality':static['locality'][static['local_mask'].detach().cpu().numpy()],'invalid':(~eligible).detach().cpu().numpy()}}
