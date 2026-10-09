from __future__ import annotations
import hashlib, json, math, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[2]
RDIR=ROOT/'diffusion_experiments/round18_diffusion_hard_negative'
PROTOCOL='ROUND18_HCD_HNC_V1'
SEEDS=(999,1000); HIST_LEN=20; T=32; AUDIT_T=(8,16,24,31)
REV_SEEDS=(20261801,20261802); UNIFORM_SEEDS=(20261811,20261812,20261813,20261814)
BANDS={1:(26,30),2:(21,25),3:(16,20),4:(11,15)}
GEN_EPOCHS=10; GEN_LR=1e-3; FT_EPOCHS=4; FT_LR_SCALE=.1
ALL=('R10','N10','R20','N20','R50','N50'); PRIMARY=('R10','N10','R20','N20')

from diffusion_experiments.round16_fcbrd import run_round16 as r16
from pipelines.msca_assets import load_msca_checkpoint, build_train_histories_and_validation
from pipelines.dataset_config import load_dataset_config
from modules.ranking import (topk_from_embeddings, metrics_at, rank_by_score, row_zscore,
                             histories_csr, l2_normalize_rows)
from modules.attribute import attribute_z, build_item_matrices, build_profiles
from modules.coliftrec import fit_backgrounds, score_coliftrec
from pipelines.coliftrec import _params


def seed_all(s):
 random.seed(int(s)); np.random.seed(int(s)); torch.manual_seed(int(s)); torch.cuda.manual_seed_all(int(s))
def sha256_file(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def utility(m,b): return float(np.mean([(m[k]-b[k])/b[k] for k in PRIMARY]))
def delta_pack(m,b):
 return {'absolute_delta':{k:float(m[k]-b[k]) for k in ALL},'relative_delta':{k:float((m[k]-b[k])/b[k]) for k in ALL},
         'U':utility(m,b),'primary_positive_count':int(sum(m[k]>b[k] for k in PRIMARY)),'overall_positive_count':int(sum(m[k]>b[k] for k in ALL))}
def stats(x):
 x=np.asarray(x,np.float64); return {'mean':float(x.mean()),'median':float(np.median(x)),'p10':float(np.quantile(x,.1)),'p50':float(np.quantile(x,.5)),'p90':float(np.quantile(x,.9)),'min':float(x.min()),'max':float(x.max())}

def checkpoint_info(seed): return r16.checkpoint_audit(seed)
def load_start(seed,freeze=False):
 a=checkpoint_info(seed); model,ck,ds,tr=load_msca_checkpoint(Path(a['checkpoint']),0)
 if freeze:
  for p in model.parameters(): p.requires_grad=False
  model.eval()
 return model,ck,ds,tr,a

@torch.no_grad()
def frozen_representations(model):
 fu,fi,collab,struct,image,text=model.forward(test=False)
 _,ci=torch.split(collab,[model.n_users,model.n_items],0)
 _,ii=torch.split(image,[model.n_users,model.n_items],0)
 _,ti=torch.split(text,[model.n_users,model.n_items],0)
 # SDPD-style history tokens use starting-model ID and projected modality features.
 id_i=model.item_id_embedding.weight.detach()
 vis_i=model.image_trs(model.image_embedding.weight).detach()
 txt_i=model.text_trs(model.text_embedding.weight).detach()
 return {'final_item':fi.detach(),'collab_item':ci.detach(),'graph_visual_item':ii.detach(),'graph_text_item':ti.detach(),
         'id_item':id_i,'visual_item':vis_i,'text_item':txt_i}

def cosine_betas(timesteps=T,s=.008,device='cpu'):
 steps=timesteps+1; x=torch.linspace(0,timesteps,steps,device=device); ac=torch.cos(((x/timesteps)+s)/(1+s)*torch.pi*.5)**2; ac=ac/ac[0]
 return torch.clamp(1-ac[1:]/ac[:-1],.0001,.9999)

class HCDDiffuser(nn.Module):
 def __init__(self,dim=64):
  super().__init__(); self.dim=dim; self.ln=nn.LayerNorm(dim,elementwise_affine=False)
  self.wq=nn.Linear(dim,dim,bias=False); self.wk=nn.Linear(dim,dim); self.wv=nn.Linear(dim,dim); self.out=nn.Linear(dim,dim)
  for m in (self.wq,self.wk,self.wv,self.out):
   nn.init.xavier_normal_(m.weight)
   if m.bias is not None: nn.init.zeros_(m.bias)
 def time_emb(self,t):
  half=self.dim//2; base=math.log(10000)/(half-1); f=torch.exp(torch.arange(half,device=t.device,dtype=torch.float32)*-base)
  z=t.float()[:,None]*f[None,:]; return torch.cat([z.sin(),z.cos()],1)
 def forward(self,xt,t,hids,hmask,id_item,vis_item,txt_item):
  # [time, history-ID, history-visual, history-text, noisy-x_t]
  te=self.time_emb(t)[:,None,:]; hi=id_item[hids]; hv=vis_item[hids]; ht=txt_item[hids]
  tok=torch.cat([te,hi,hv,ht,xt[:,None,:]],1); x=self.ln(tok)
  q=self.wq(x)*(self.dim**-.5); k=self.wk(x); v=self.wv(x); score=q@k.transpose(-1,-2)
  B,L=hmask.shape; valid=torch.cat([torch.ones(B,1,device=x.device,dtype=torch.bool),hmask,hmask,hmask,torch.ones(B,1,device=x.device,dtype=torch.bool)],1)
  score=score.masked_fill(~valid[:,None,:],-1e4); att=score.softmax(-1); y=att@v
  return self.out(y[:,-1,:])

class DiffusionSchedule:
 def __init__(self,device):
  self.betas=cosine_betas(device=device); self.alphas=1-self.betas; self.ac=torch.cumprod(self.alphas,0); self.acprev=F.pad(self.ac[:-1],(1,0),value=1.)
  self.sqrt_ac=torch.sqrt(self.ac); self.sqrt_om=torch.sqrt(1-self.ac)
  self.pm1=self.betas*torch.sqrt(self.acprev)/(1-self.ac); self.pm2=(1-self.acprev)*torch.sqrt(self.alphas)/(1-self.ac)
  self.pvar=self.betas*(1-self.acprev)/(1-self.ac)
 def extract(self,a,t,x): return a[t].reshape(len(t),1).to(x.device)
 def q(self,x0,t,noise): return self.extract(self.sqrt_ac,t,x0)*x0+self.extract(self.sqrt_om,t,x0)*noise

class HistoryData:
 def __init__(self,seed):
  cfg=load_dataset_config('baby'); a=checkpoint_info(seed); self.n_users=int(a['n_users']); self.n_items=int(a['n_items'])
  h,p,pu,vu,vs=build_train_histories_and_validation(cfg['resolved_paths']['interaction'],self.n_users)
  self.histories=h; self.pseudo=p; self.pseudo_users=pu; self.valid_users=vu; self.valid_sets=vs
  self.clean=np.load(RDIR.parent/'round16r_cleantrain'/f'assets/seed{seed}_train_clean.npz')
  self.clean_users=self.clean['users'].astype(np.int64); self.targets=self.clean['target_item'].astype(np.int64)
  if not np.array_equal(self.clean_users,self.pseudo_users[np.isin(self.pseudo_users,self.clean_users)]):
   # clean set may drop users; order still must be ascending canonical user order
   if np.any(np.diff(self.clean_users)<0): raise RuntimeError('clean users not canonical')
  self.row={int(u):i for i,u in enumerate(self.clean_users)}
 def pack_prefixes(self,users,target_positions=None,source_users=None,device='cuda'):
  users=np.asarray(users,np.int64); src=users if source_users is None else np.asarray(source_users,np.int64)
  arr=np.zeros((len(users),HIST_LEN),np.int64); mask=np.zeros((len(users),HIST_LEN),bool)
  for r,(u,su) in enumerate(zip(users,src)):
   if target_positions is None: pref=self.histories[int(su)]
   else:
    # target_positions correspond to source user's chronology only in true-history generator events
    pref=self.histories[int(su)][:int(target_positions[r])]
   pref=list(pref[-HIST_LEN:]); arr[r,:len(pref)]=pref; mask[r,:len(pref)]=True
  return torch.as_tensor(arr,device=device,dtype=torch.long),torch.as_tensor(mask,device=device)
 def clean_prefix_pack(self,users,source_users=None,device='cuda'):
  # clean pseudo pair target is final TRAIN event; prefix is h[:-1]
  users=np.asarray(users,np.int64); src=users if source_users is None else np.asarray(source_users,np.int64)
  arr=np.zeros((len(users),HIST_LEN),np.int64); mask=np.zeros((len(users),HIST_LEN),bool)
  for r,su in enumerate(src):
   pref=list(self.histories[int(su)][:-1][-HIST_LEN:]); arr[r,:len(pref)]=pref; mask[r,:len(pref)]=True
  return torch.as_tensor(arr,device=device,dtype=torch.long),torch.as_tensor(mask,device=device)

class ChronoEvents:
 def __init__(self,hdata:HistoryData,split_seed=20261800):
  eligible=np.asarray([u for u,h in enumerate(hdata.histories) if len(h)>=3],np.int64); rng=np.random.default_rng(split_seed); perm=rng.permutation(eligible)
  cut=int(round(.9*len(perm))); train_users=set(map(int,perm[:cut])); audit_users=set(map(int,perm[cut:])); self.train_users=train_users; self.audit_users=audit_users
  def build(us):
   U=[];P=[];Tg=[]
   for u in sorted(us):
    h=hdata.histories[u]
    for k in range(2,len(h)): U.append(u);P.append(k);Tg.append(h[k])
   return np.asarray(U,np.int64),np.asarray(P,np.int16),np.asarray(Tg,np.int64)
  self.train=build(train_users); self.audit=build(audit_users)


def save_frozen_repr(seed,model,reps):
 p=RDIR/'assets'/f'seed{seed}_frozen_repr.npz'; np.savez_compressed(p,**{k:v.detach().cpu().numpy().astype(np.float32) for k,v in reps.items()}); return p

def load_repr(seed,device):
 z=np.load(RDIR/'assets'/f'seed{seed}_frozen_repr.npz'); return {k:torch.as_tensor(z[k],device=device) for k in z.files}

def batch_events(events,order,batch=512):
 U,P,Tg=events
 for s in range(0,len(order),batch):
  ix=order[s:s+batch]; yield U[ix],P[ix],Tg[ix]

def generator_loss(net,sched,reps,hdata,u,pos,target,device):
 hids,hmask=hdata.pack_prefixes(u,pos,device=device); x0=reps['final_item'][torch.as_tensor(target,device=device)]
 t=torch.randint(0,T,(len(u),),device=device); eps=torch.randn_like(x0); xt=sched.q(x0,t,eps)
 pred=net(xt,t,hids,hmask,reps['id_item'],reps['visual_item'],reps['text_item']); return F.mse_loss(pred,x0),pred,x0,xt,t

@torch.no_grad()
def generator_quality(net,sched,reps,hdata,events,device,max_events=None):
 U,P,Tg=events
 if max_events is not None: U,P,Tg=U[:max_events],P[:max_events],Tg[:max_events]
 out={}; pred_all=[]
 for tt in AUDIT_T:
  noisy=[]; den=[]; norms=[]
  for s in range(0,len(U),512):
   u=U[s:s+512];p=P[s:s+512];tg=Tg[s:s+512];hids,hmask=hdata.pack_prefixes(u,p,device=device);x0=reps['final_item'][torch.as_tensor(tg,device=device)]
   g=torch.Generator(device=device); g.manual_seed(20261880+tt*100000+s); eps=torch.randn(x0.shape,generator=g,device=device); t=torch.full((len(u),),tt,device=device,dtype=torch.long);xt=sched.q(x0,t,eps);pred=net(xt,t,hids,hmask,reps['id_item'],reps['visual_item'],reps['text_item'])
   noisy.append(F.cosine_similarity(xt,x0).cpu().numpy());den.append(F.cosine_similarity(pred,x0).cpu().numpy());norms.append(pred.norm(dim=1).cpu().numpy());pred_all.append(pred.cpu())
  a=np.concatenate(noisy);b=np.concatenate(den);n=np.concatenate(norms);out[str(tt)]={'noisy_cos':stats(a),'denoised_cos':stats(b),'mean_gain':float(b.mean()-a.mean()),'PASS':bool(np.isfinite(b).all() and b.mean()>a.mean()),'pred_norm':stats(n)}
 PRED=torch.cat(pred_all); dimstd=PRED.std(dim=0).mean().item(); collapse=dimstd<1e-4 or not torch.isfinite(PRED).all()
 return {'timesteps':out,'all_t_pass':all(x['PASS'] for x in out.values()),'representation_dim_std_mean':dimstd,'representation_collapse':bool(collapse),'all_finite':bool(torch.isfinite(PRED).all()),'PASS':bool(all(x['PASS'] for x in out.values()) and not collapse and torch.isfinite(PRED).all())}

@torch.no_grad()
def reverse_stages(net,sched,reps,hdata,users,source_users=None,device='cuda',batch=256):
 users=np.asarray(users,np.int64); source_users=users if source_users is None else np.asarray(source_users,np.int64)
 captures={8:[],16:[],24:[],32:[]}
 for s in range(0,len(users),batch):
  u=users[s:s+batch]; su=source_users[s:s+batch]; hids,hmask=hdata.clean_prefix_pack(u,su,device=device); per_seed={k:[] for k in captures}
  for ns in REV_SEEDS:
   g=torch.Generator(device=device);g.manual_seed(int(ns)*100003+s);x=torch.randn((len(u),64),generator=g,device=device)
   got={}
   step=0
   for ti in reversed(range(T)):
    t=torch.full((len(u),),ti,device=device,dtype=torch.long); x0=net(x,t,hids,hmask,reps['id_item'],reps['visual_item'],reps['text_item']); mean=sched.extract(sched.pm1,t,x)*x0+sched.extract(sched.pm2,t,x)*x
    if ti==0: x=mean
    else:
     z=torch.randn(x.shape,generator=g,device=device);x=mean+torch.sqrt(sched.extract(sched.pvar,t,x))*z
    step+=1
    if step in captures: got[step]=F.normalize(x,p=2,dim=1)
   for k in captures: per_seed[k].append(got[k])
  for k in captures: captures[k].append(F.normalize(per_seed[k][0]+per_seed[k][1],p=2,dim=1).cpu().numpy().astype(np.float32))
 return {k:np.concatenate(v) for k,v in captures.items()}

def legal_band_rows(hdata:HistoryData,epoch:int):
 lo,hi=BANDS[epoch]; items=hdata.clean['ranked_items'].astype(np.int64); score=hdata.clean['ranked_full_scores'].astype(np.float32); tcol=hdata.clean['target_col'].astype(np.int64)
 n=len(hdata.clean_users); cand=np.full((n,hi-lo+1),-1,np.int64); ranks=np.full_like(cand,-1); scores=np.full(cand.shape,np.nan,np.float32); valid=np.zeros(cand.shape,bool)
 for r,u in enumerate(hdata.clean_users):
  observed=set(hdata.histories[int(u)][:-1]); target=int(hdata.targets[r]); kk=0
  for rank in range(lo,hi+1):
   j=int(items[r,rank-1])
   if j<0 or j>=hdata.n_items or j==target or j in observed: continue
   cand[r,kk]=j;ranks[r,kk]=rank;scores[r,kk]=score[r,rank-1];valid[r,kk]=True;kk+=1
 return cand,ranks,scores,valid

def deterministic_uniform(hdata:HistoryData,epoch:int,eligible):
 rng=np.random.default_rng(UNIFORM_SEEDS[epoch-1]); out=np.full(len(hdata.clean_users),-1,np.int64)
 for r,u in enumerate(hdata.clean_users):
  if not eligible[r]: continue
  bad=set(hdata.histories[int(u)][:-1]); bad.add(int(hdata.targets[r]))
  while True:
   j=int(rng.integers(0,hdata.n_items))
   if j not in bad: out[r]=j;break
 return out

def select_from_band(cand,valid,query,item_emb):
 item_norm=item_emb/np.maximum(np.linalg.norm(item_emb,axis=1,keepdims=True),1e-12); q=query/np.maximum(np.linalg.norm(query,axis=1,keepdims=True),1e-12)
 out=np.full(len(cand),-1,np.int64); sim=np.full(len(cand),np.nan,np.float32)
 for r in range(len(cand)):
  ids=cand[r,valid[r]]
  if len(ids)==0: continue
  s=item_norm[ids]@q[r]; k=int(np.argmax(s));out[r]=int(ids[k]);sim[r]=float(s[k])
 return out,sim

def build_negative_maps(seed,stages,shuf_stages):
 h=HistoryData(seed); z=np.load(RDIR/'assets'/f'seed{seed}_frozen_repr.npz'); item=z['final_item'].astype(np.float32); pos=item[h.targets]
 rows=[]; overlap={};histdiag={}; bandaudit={}
 maps={v:[] for v in ('F1','F2','F3','F4')}
 stage_for={1:8,2:16,3:24,4:32}
 for ep in range(1,5):
  cand,ranks,scores,valid=legal_band_rows(h,ep); eligible=valid.any(1); coverage=float(eligible.mean()); bandaudit[str(ep)]={'band':list(BANDS[ep]),'coverage':coverage,'eligible':int(eligible.sum()),'total':int(len(eligible)),'LOW_BAND_COVERAGE':coverage<.9}
  # F1: uniform but restricted to same users eligible for the current band for exact positive-user fairness.
  n1=deterministic_uniform(h,ep,eligible)
  # F2: frontmost legal frozen rank in band.
  n2=np.full(len(eligible),-1,np.int64);r2=np.full(len(eligible),-1,np.int64);s2=np.full(len(eligible),np.nan,np.float32)
  for r in np.where(eligible)[0]:
   k=np.where(valid[r])[0][0];n2[r]=cand[r,k];r2[r]=ranks[r,k];s2[r]=scores[r,k]
  # F3: raw positive representation query.
  n3,sim3=select_from_band(cand,valid,pos,item)
  # F4: history-conditioned reverse-stage query.
  q=stages[stage_for[ep]]; n4,sim4=select_from_band(cand,valid,q,item)
  qs=shuf_stages[stage_for[ep]]; n4s,sim4s=select_from_band(cand,valid,qs,item)
  for v,neg in [('F1',n1),('F2',n2),('F3',n3),('F4',n4)]:
   rr=np.full(len(neg),-1,np.int64);bs=np.full(len(neg),np.nan,np.float32)
   for r in np.where(eligible)[0]:
    if v=='F1': continue
    loc=np.where(cand[r]==neg[r])[0];
    if len(loc): rr[r]=ranks[r,loc[0]];bs[r]=scores[r,loc[0]]
   maps[v].append({'epoch':ep,'eligible':eligible,'negative':neg,'rank':rr,'base_score':bs})
  ov=float(np.mean(n3[eligible]==n4[eligible])); overlap[str(ep)]={'same_negative_fraction':ov,'DIFFUSION_QUERY_REDUNDANT':ov>.9}
  agree=float(np.mean(n4[eligible]==n4s[eligible])); qcos=np.sum(q*qs,axis=1)/(np.maximum(np.linalg.norm(q,axis=1),1e-12)*np.maximum(np.linalg.norm(qs,axis=1),1e-12));
  histdiag[str(ep)]={'selected_negative_agreement':agree,'query_cosine':stats(qcos[eligible]),'true_selected_query_cos':stats(sim4[eligible]),'shuffled_selected_query_cos':stats(sim4s[eligible]),'HISTORY_CONDITION_EFFECT_WEAK':agree>.9}
  # audit rows: positive score from frozen clean row target col
  pscore=h.clean['ranked_full_scores'][np.arange(len(eligible)),h.clean['target_col'].astype(np.int64)]
  for v in ('F2','F3','F4'):
   m=maps[v][-1];sel=m['eligible'];neg=m['negative']; negcos=np.sum((item[neg[sel]]/np.maximum(np.linalg.norm(item[neg[sel]],axis=1,keepdims=True),1e-12))*(pos[sel]/np.maximum(np.linalg.norm(pos[sel],axis=1,keepdims=True),1e-12)),axis=1)
   margin=pscore[sel]-m['base_score'][sel]; m['audit']={'frozen_rank':stats(m['rank'][sel]),'positive_negative_cosine':stats(negcos),'base_score':stats(m['base_score'][sel]),'positive_base_margin':stats(margin)}
  qpos=np.sum((q/np.maximum(np.linalg.norm(q,axis=1,keepdims=True),1e-12))*(pos/np.maximum(np.linalg.norm(pos,axis=1,keepdims=True),1e-12)),axis=1)
  maps['F4'][-1]['audit']['query_positive_cosine']=stats(qpos[eligible]);maps['F4'][-1]['audit']['query_negative_cosine']=stats(sim4[eligible])
 # save maps with explicit epoch rows
 for v in maps:
  arr=[]
  for rec in maps[v]:
   ep=rec['epoch'];sel=np.where(rec['eligible'])[0]
   arr.append((h.clean_users[sel],h.targets[sel],np.full(len(sel),ep,np.int16),rec['negative'][sel],rec['rank'][sel]))
  U=np.concatenate([x[0] for x in arr]);P=np.concatenate([x[1] for x in arr]);E=np.concatenate([x[2] for x in arr]);N=np.concatenate([x[3] for x in arr]);R=np.concatenate([x[4] for x in arr])
  path=RDIR/'assets'/f'seed{seed}_{v}_negative_map.npz';np.savez_compressed(path,user=U,positive=P,epoch=E,negative=N,frozen_rank=R)
 return maps,bandaudit,overlap,histdiag

class ValidationEvaluator:
 def __init__(self,seed):
  self.seed=seed; self.h=HistoryData(seed); cfg=load_dataset_config('baby');self.cfg=cfg;self.paths=cfg['resolved_paths'];self.ccfg=cfg['coliftrec'];self.params=_params(self.ccfg);self.enabled={m:bool(self.ccfg[m]['enabled']) for m in ('text','attribute','visual')}
  self.text=l2_normalize_rows(np.asarray(np.load(self.paths['text_feature'],mmap_mode='r'),dtype=np.float32));self.visual=l2_normalize_rows(np.asarray(np.load(self.paths['visual_feature'],mmap_mode='r'),dtype=np.float32))
  Hf=histories_csr(self.h.histories,self.h.n_items,users=self.h.valid_users,mean=True);Hp=histories_csr(self.h.pseudo,self.h.n_items,users=self.h.pseudo_users,mean=True)
  self.text_full=l2_normalize_rows(np.asarray(Hf@self.text,dtype=np.float32));self.text_pseudo=l2_normalize_rows(np.asarray(Hp@self.text,dtype=np.float32));self.vis_full=l2_normalize_rows(np.asarray(Hf@self.visual,dtype=np.float32));self.vis_pseudo=l2_normalize_rows(np.asarray(Hp@self.visual,dtype=np.float32))
  ac=self.ccfg['attribute'];self.mats,_=build_item_matrices(self.paths['metadata'],self.h.n_items,min_df=int(ac.get('tfidf_min_df',2)),max_df=float(ac.get('tfidf_max_df',.8)),description_len=int(ac.get('description_len',128)),weights=ac.get('weights'))
  self.full_profiles=build_profiles(self.mats,self.h.histories,self.h.n_items);self.pseudo_profiles=build_profiles(self.mats,self.h.pseudo,self.h.n_items);self.aw=ac.get('weights')
 def sem(self,feat,prof,items): return row_zscore(np.einsum('bld,bd->bl',feat[items],prof,optimize=True).astype(np.float32))
 @torch.no_grad()
 def evaluate(self,model):
  model.eval();fu,fi=model.forward(test=True);vi,vs=topk_from_embeddings(fu,fi,self.h.valid_users,self.h.histories,100,1024);pi,ps=topk_from_embeddings(fu,fi,self.h.pseudo_users,self.h.pseudo,100,1024)
  back=metrics_at(vi,self.h.valid_users,self.h.valid_sets)
  ztp=self.sem(self.text,self.text_pseudo,pi);ztv=self.sem(self.text,self.text_full,vi);zvp=self.sem(self.visual,self.vis_pseudo,pi);zvv=self.sem(self.visual,self.vis_full,vi)
  zap,_=attribute_z(self.mats,self.pseudo_profiles,self.h.pseudo_users,pi,256,self.aw);zav,_=attribute_z(self.mats,self.full_profiles,self.h.valid_users,vi,256,self.aw);bg=fit_backgrounds(pi,ztp,zap,zvp,self.h.n_items)
  full_scores,_=score_coliftrec(vs,vi,ztv,zav,zvv,bg,self.params,self.enabled);rank=rank_by_score(vi,full_scores);full=metrics_at(rank,self.h.valid_users,self.h.valid_sets)
  return {'backbone':back,'colift':full,'users':self.h.valid_users,'items':vi,'backbone_scores':vs,'full_scores':full_scores,'diffusion_inference_calls':0}
