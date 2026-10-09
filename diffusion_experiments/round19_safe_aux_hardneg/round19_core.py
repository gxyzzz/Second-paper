from __future__ import annotations
import hashlib,json,random,gc
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[2]
RDIR=ROOT/'diffusion_experiments/round19_safe_aux_hardneg'
R18=ROOT/'diffusion_experiments/round18_diffusion_hard_negative'
PROTOCOL='ROUND19_SA_DHN_V1'
SEEDS=(999,1000); EPOCHS=3; LAMBDA_HARD=.20; Q_STEP=24; TOPK=5
MAP_SEEDS=(20261901,20261902,20261903); PLAN_SEEDS=(20261921,20261922,20261923); AUX_ORDER_SEEDS=(20261931,20261932,20261933)
GEN_HASH={999:'a4b6aba18442fe60e9c6648faa8d28df491ca919b4b0d7fb98b86aa7a6f805cf',1000:'a6fadc12432ffe4f6d5ba33b28be75250070e2d4c5a4a9c484ce3614f3f0d922'}
ALL=('R10','N10','R20','N20','R50','N50'); PRIMARY=('R10','N10','R20','N20')

from diffusion_experiments.round18_diffusion_hard_negative.round18_core import (
 HCDDiffuser,DiffusionSchedule,HistoryData,ValidationEvaluator,load_start,frozen_representations,sha256_file,
 rank_by_score,topk_from_embeddings,metrics_at,row_zscore,l2_normalize_rows,histories_csr,
 attribute_z,build_item_matrices,build_profiles,fit_backgrounds,score_coliftrec,_params)
from utils.dataloader import TrainDataLoader
from pipelines.dataset_config import load_dataset_config
from pipelines.msca_assets import build_train_histories_and_validation


def seed_all(s):
 random.seed(int(s));np.random.seed(int(s));torch.manual_seed(int(s));torch.cuda.manual_seed_all(int(s))
def stats(x):
 x=np.asarray(x,np.float64);return {'mean':float(x.mean()),'median':float(np.median(x)),'p10':float(np.quantile(x,.1)),'p50':float(np.quantile(x,.5)),'p90':float(np.quantile(x,.9)),'min':float(x.min()),'max':float(x.max())}
def utility(m,b):return float(np.mean([(m[k]-b[k])/b[k] for k in PRIMARY]))
def delta_pack(m,b):
 rel={k:float((m[k]-b[k])/b[k]) for k in ALL};return {'absolute_delta':{k:float(m[k]-b[k]) for k in ALL},'relative_delta':rel,'U':float(np.mean([rel[k] for k in PRIMARY])),'primary_positive_count':int(sum(m[k]>b[k] for k in PRIMARY)),'overall_positive_count':int(sum(m[k]>b[k] for k in ALL))}

def load_generator(seed,device):
 p=R18/'assets'/f'seed{seed}_generator_epoch10.pt';h=sha256_file(p)
 if h!=GEN_HASH[seed]:raise RuntimeError(f'generator hash mismatch {seed}: {h}')
 x=torch.load(p,map_location='cpu',weights_only=False);net=HCDDiffuser().to(device);net.load_state_dict(x['state'],strict=True);net.eval()
 for q in net.parameters():q.requires_grad=False
 return net,{'path':str(p),'sha256':h,'epoch':int(x['epoch']),'trainable_params':sum(q.numel() for q in net.parameters() if q.requires_grad)}

class EventData:
 def __init__(self,seed):
  self.h=HistoryData(seed);U=[];P=[];T=[]
  for u,h in enumerate(self.h.histories):
   for k in range(2,len(h)):U.append(u);P.append(k);T.append(h[k])
  self.user=np.asarray(U,np.int64);self.pos=np.asarray(P,np.int16);self.target=np.asarray(T,np.int64)
  self.n=len(U);self.key=np.arange(self.n,dtype=np.int64)
 def pack_prefix(self,rows,source_users=None,device='cuda'):
  rows=np.asarray(rows,np.int64);users=self.user[rows];positions=self.pos[rows];src=users if source_users is None else np.asarray(source_users,np.int64)
  arr=np.zeros((len(rows),20),np.int64);mask=np.zeros((len(rows),20),bool)
  for r,(u,k,su) in enumerate(zip(users,positions,src)):
   if source_users is None: pref=self.h.histories[int(u)][:int(k)]
   else:
    sh=self.h.histories[int(su)];pref=sh[:min(int(k),len(sh))]
   pref=list(pref[-20:]);arr[r,:len(pref)]=pref;mask[r,:len(pref)]=True
  return torch.as_tensor(arr,device=device),torch.as_tensor(mask,device=device)

def load_round18_repr(seed,device):
 p=R18/'assets'/f'seed{seed}_frozen_repr.npz'
 z=np.load(p)
 return {k:torch.as_tensor(z[k],device=device) for k in z.files},sha256_file(p)

@torch.no_grad()
def event_q75(seed,shuffled=False,batch=256):
 ed=EventData(seed);device='cuda:0'
 reps,rhash=load_round18_repr(seed,device)
 net,ga=load_generator(seed,device)
 sched=DiffusionSchedule(device)
 outs=[];rows=np.arange(ed.n,dtype=np.int64)
 source=None if not shuffled else ((ed.user+1)%ed.h.n_users).astype(np.int64)
 for st in range(0,ed.n,batch):
  rr=rows[st:st+batch]
  src=None if source is None else source[st:st+batch]
  hids,hmask=ed.pack_prefix(rr,src,device=device)
  per=[]
  for ns in (20261801,20261802):
   g=torch.Generator(device=device);g.manual_seed(int(ns)*100003+st)
   x=torch.randn((len(rr),64),generator=g,device=device)
   cap=None;step=0
   for ti in reversed(range(32)):
    t=torch.full((len(rr),),ti,device=device,dtype=torch.long)
    x0=net(x,t,hids,hmask,reps['id_item'],reps['visual_item'],reps['text_item'])
    mean=sched.extract(sched.pm1,t,x)*x0+sched.extract(sched.pm2,t,x)*x
    if ti==0:x=mean
    else:x=mean+torch.sqrt(sched.extract(sched.pvar,t,x))*torch.randn(x.shape,generator=g,device=device)
    step+=1
    if step==Q_STEP:cap=F.normalize(x,p=2,dim=1).clone()
   per.append(cap)
  outs.append(F.normalize(per[0]+per[1],p=2,dim=1).cpu().numpy().astype(np.float32))
 q=np.concatenate(outs)
 p=RDIR/'assets'/f'seed{seed}_{"shuffled_" if shuffled else ""}q75.npz'
 np.savez_compressed(p,user=ed.user,event_position=ed.pos,target=ed.target,q75=q)
 del net;torch.cuda.empty_cache();gc.collect()
 return {'seed':seed,'shuffled':shuffled,'events':ed.n,'shape':list(q.shape),'q75_reverse_steps':Q_STEP,'generator':ga,'frozen_repr_sha256':rhash,'sha256':sha256_file(p),'TEST_ACCESSED':False}

def top5_for_query(pool_items,q,item_emb):
 emb=item_emb/np.maximum(np.linalg.norm(item_emb,axis=1,keepdims=True),1e-12)
 q=q/np.maximum(np.linalg.norm(q,axis=1,keepdims=True),1e-12)
 s=np.einsum('bkd,bd->bk',emb[pool_items],q,optimize=True)
 ix=np.argsort(-s,axis=1,kind='stable')[:,:TOPK]
 return np.take_along_axis(pool_items,ix,axis=1),np.take_along_axis(s,ix,axis=1)

def build_aux_maps(seed):
 ed=EventData(seed)
 pool=np.load(RDIR/'assets'/f'seed{seed}_safe_pool.npz')
 qz=np.load(RDIR/'assets'/f'seed{seed}_q75.npz')
 sqz=np.load(RDIR/'assets'/f'seed{seed}_shuffled_q75.npz')
 rz=np.load(R18/'assets'/f'seed{seed}_frozen_repr.npz')
 item=rz['final_item'].astype(np.float32)
 row={int(u):i for i,u in enumerate(pool['users'])}
 pr=np.asarray([row[int(u)] for u in ed.user],np.int64)
 safe=pool['items'][pr,20:100].astype(np.int32)
 bad=0
 for r,u in enumerate(ed.user):
  obs=set(ed.h.histories[int(u)])
  bad+=sum((int(j) in obs) or (int(j)==int(ed.target[r])) or (int(j)<0) for j in safe[r])
 posq=item[ed.target]
 a3top,a3sim=top5_for_query(safe,posq,item)
 a4top,a4sim=top5_for_query(safe,qz['q75'].astype(np.float32),item)
 shtop,shsim=top5_for_query(safe,sqz['q75'].astype(np.float32),item)
 maps={}
 for v in ('A2','A3','A4'):
  rows=[]
  for ep,sd in enumerate(MAP_SEEDS,1):
   rng=np.random.default_rng(sd)
   if v=='A2':
    pick=rng.integers(0,20,size=ed.n)
    neg=safe[np.arange(ed.n),pick]
    orig_rank=21+pick
    top=np.full((ed.n,TOPK),-1,np.int32)
   else:
    top=a3top if v=='A3' else a4top
    pick=rng.integers(0,TOPK,size=ed.n)
    neg=top[np.arange(ed.n),pick]
    orig_rank=np.empty(ed.n,np.int16)
    for r in range(ed.n):
     orig_rank[r]=21+int(np.where(safe[r]==neg[r])[0][0])
   rows.append((np.full(ed.n,ep,np.int8),neg.astype(np.int32),orig_rank.astype(np.int16),top.astype(np.int32)))
  E=np.concatenate([x[0] for x in rows]);N=np.concatenate([x[1] for x in rows]);R=np.concatenate([x[2] for x in rows]);T5=np.concatenate([x[3] for x in rows])
  U=np.tile(ed.user,EPOCHS);P=np.tile(ed.target,EPOCHS);POS=np.tile(ed.pos,EPOCHS);KEY=np.tile(ed.key,EPOCHS)
  path=RDIR/'assets'/f'seed{seed}_{v}_aux_map.npz'
  np.savez_compressed(path,user=U,positive=P,event_position=POS,event_key=KEY,epoch=E,negative=N,original_rank=R,top5_candidates=T5)
  maps[v]={'path':str(path),'sha256':sha256_file(path),'rows':int(len(E)),'events_per_epoch':ed.n}
 top_overlap=float(np.mean([len(set(a3top[r])&set(a4top[r]))/TOPK for r in range(ed.n)]))
 qcos=np.sum(qz['q75']*sqz['q75'],axis=1)/(np.maximum(np.linalg.norm(qz['q75'],axis=1),1e-12)*np.maximum(np.linalg.norm(sqz['q75'],axis=1),1e-12))
 sh_overlap=float(np.mean([len(set(a4top[r])&set(shtop[r]))/TOPK for r in range(ed.n)]))
 rng=np.random.default_rng(MAP_SEEDS[0]);pick=rng.integers(0,TOPK,size=ed.n)
 agree=float(np.mean(a4top[np.arange(ed.n),pick]==shtop[np.arange(ed.n),pick]))
 collision={}
 for v in ('A2','A3','A4'):
  z=np.load(maps[v]['path'])
  hits=np.fromiter((int(n) in ed.h.valid_sets.get(int(u),set()) for u,n in zip(z['user'],z['negative'])),dtype=bool,count=len(z['user']))
  collision[v]=float(hits.mean())
 z3=np.load(maps['A3']['path']);z4=np.load(maps['A4']['path'])
 same_neg=float(np.mean(z3['negative']==z4['negative']))
 emb=item/np.maximum(np.linalg.norm(item,axis=1,keepdims=True),1e-12)
 model,ck,ds,tr,a=load_start(seed,False);model.eval()
 with torch.no_grad():fu,fi=model.forward(test=True);fu_np=fu.cpu().numpy().astype(np.float32)
 hard={}
 for v in ('A2','A3','A4'):
  z=np.load(maps[v]['path']);ep1=np.flatnonzero(z['epoch']==1)
  neg=z['negative'][ep1];u=z['user'][ep1];pos=z['positive'][ep1];ranks=z['original_rank'][ep1].astype(np.int64)
  rr=np.asarray([row[int(x)] for x in u],np.int64);cols=ranks-1
  bneg=pool['backbone_scores'][rr,cols];fneg=pool['full_scores'][rr,cols]
  pscore=np.sum(fu_np[u]*item[pos],axis=1);cos=np.sum(emb[pos]*emb[neg],axis=1)
  hard[v]={'rank':stats(ranks),'backbone_negative_score':stats(bneg),'fullcolift_negative_score':stats(fneg),'backbone_positive_negative_margin':stats(pscore-bneg),'positive_negative_embedding_cosine':stats(cos)}
 del model;torch.cuda.empty_cache();gc.collect()
 audit={'seed':seed,'events':ed.n,'safe_pool_all_train_observed_violation_count':int(bad),'protected_rank1_20_selected_count':0,'A3_A4_same_negative_fraction':same_neg,'A3_A4_top5_candidate_overlap':top_overlap,'DIFFUSION_QUERY_REDUNDANT':bool(top_overlap>.9),'history_specificity':{'true_shuffled_top5_overlap':sh_overlap,'selected_negative_agreement_epoch1':agree,'query_cosine':stats(qcos)},'VALIDATION_FUTURE_POSITIVE_COLLISION':collision,'hardness':hard,'maps':maps,'TEST_ACCESSED':False}
 return audit

def build_normal_plan(seed):
 model,ck,ds,tr,a=load_start(seed,False);c=ck['config']
 dl=TrainDataLoader(c,tr,batch_size=c['train_batch_size'],shuffle=True);dl.pretrain_setup()
 rows=[]
 for ep,sd in enumerate(PLAN_SEEDS,1):
  random.seed(sd);np.random.seed(sd);torch.manual_seed(sd)
  for bi,b in enumerate(dl):
   x=b.detach().cpu().numpy();n=x.shape[1]
   rows.append((np.full(n,ep,np.int8),np.full(n,bi,np.int16),x[0].astype(np.int32),x[1].astype(np.int32),x[2].astype(np.int32)))
 E=np.concatenate([x[0] for x in rows]);B=np.concatenate([x[1] for x in rows]);U=np.concatenate([x[2] for x in rows]);P=np.concatenate([x[3] for x in rows]);N=np.concatenate([x[4] for x in rows])
 p=RDIR/'assets'/f'seed{seed}_normal_train_plan.npz';np.savez_compressed(p,epoch=E,batch=B,user=U,positive=P,normal_negative=N)
 out={'seed':seed,'train_interactions_per_epoch':int(len(tr)),'batch_size':int(c['train_batch_size']),'batches_per_epoch':int(len(dl)),'epochs':EPOCHS,'rows':int(len(E)),'source':'original TrainDataLoader + original _sample_neg_ids','plan_seeds':list(PLAN_SEEDS),'sha256':sha256_file(p),'TEST_ACCESSED':False}
 del model;torch.cuda.empty_cache();gc.collect();return out

def round19_loss(model,normal,aux=None,lambda_hard=0.0):
 users,pos_items,neg_items=normal
 fu,fi,collab,struct,image,text=model.forward(test=False)
 bpr=model.cal_bpr_loss(fu[users],fi[pos_items],fi[neg_items]);reg=model.cal_reg_loss()
 cu,ci=torch.split(collab,[model.n_users,model.n_items],0);su,si=torch.split(struct,[model.n_users,model.n_items],0);iu,ii=torch.split(image,[model.n_users,model.n_items],0);tu,ti=torch.split(text,[model.n_users,model.n_items],0)
 Mu=model.cal_cl_loss(cu[users],iu[users],model.tau)+model.cal_cl_loss(cu[users],tu[users],model.tau)
 Mi=model.cal_cl_loss(ci[pos_items],ii[pos_items],model.tau)+model.cal_cl_loss(ci[pos_items],ti[pos_items],model.tau)
 Cu=model.cal_cl_loss(cu[users],su[users],model.tau);Ci=model.cal_cl_loss(ci[pos_items],si[pos_items],model.tau)
 cl=Cu+Ci+Mu+Mi;msca=bpr+model.cl_weight*cl+model.reg_weight*reg
 auxloss=torch.zeros((),device=msca.device)
 if aux is not None and lambda_hard!=0:
  au,ap,an=aux;auxloss=model.cal_bpr_loss(fu[au],fi[ap],fi[an])
 total=msca+float(lambda_hard)*auxloss
 return total,{'L_MSCA':msca,'L_normal_BPR':bpr,'L_CL':cl,'L_reg':reg,'L_aux':auxloss,'L_total':total}

class CachedEvaluator:
 def __init__(self,seed,model):
  self.seed=seed;self.device=model.device;self.cfg=load_dataset_config('baby');p=self.cfg['resolved_paths'];self.ccfg=self.cfg['coliftrec'];self.params=_params(self.ccfg);self.enabled={m:bool(self.ccfg[m]['enabled']) for m in ('text','attribute','visual')}
  self.hist,self.pseudo,self.pusers,self.vusers,self.vsets=build_train_histories_and_validation(p['interaction'],model.n_users);self.n_items=model.n_items
  text=l2_normalize_rows(np.asarray(np.load(p['text_feature'],mmap_mode='r'),dtype=np.float32));visual=l2_normalize_rows(np.asarray(np.load(p['visual_feature'],mmap_mode='r'),dtype=np.float32))
  hf=histories_csr(self.hist,self.n_items,users=self.vusers,mean=True);hp=histories_csr(self.pseudo,self.n_items,users=self.pusers,mean=True)
  tf=l2_normalize_rows(np.asarray(hf@text,dtype=np.float32));tp=l2_normalize_rows(np.asarray(hp@text,dtype=np.float32));vf=l2_normalize_rows(np.asarray(hf@visual,dtype=np.float32));vp=l2_normalize_rows(np.asarray(hp@visual,dtype=np.float32))
  self.text=torch.as_tensor(text,device=self.device);self.visual=torch.as_tensor(visual,device=self.device);self.tf=torch.as_tensor(tf,device=self.device);self.tp=torch.as_tensor(tp,device=self.device);self.vf=torch.as_tensor(vf,device=self.device);self.vp=torch.as_tensor(vp,device=self.device)
  ac=self.ccfg['attribute'];self.mats,_=build_item_matrices(p['metadata'],self.n_items,min_df=int(ac.get('tfidf_min_df',2)),max_df=float(ac.get('tfidf_max_df',.8)),description_len=int(ac.get('description_len',128)),weights=ac.get('weights'))
  self.af=build_profiles(self.mats,self.hist,self.n_items);self.ap=build_profiles(self.mats,self.pseudo,self.n_items);self.aw=ac.get('weights')
 def sem(self,feat,prof,items,batch):
  outs=[]
  for st in range(0,len(items),batch):
   en=min(st+batch,len(items));ids=torch.as_tensor(items[st:en],device=self.device);outs.append(torch.einsum('bld,bd->bl',feat[ids],prof[st:en]).cpu())
  return row_zscore(torch.cat(outs).numpy())
 @torch.no_grad()
 def evaluate(self,model):
  model.eval();fu,fi=model.forward(test=True);vi,vs=topk_from_embeddings(fu,fi,self.vusers,self.hist,100,1024);pi,ps=topk_from_embeddings(fu,fi,self.pusers,self.pseudo,100,1024);back=metrics_at(vi,self.vusers,self.vsets)
  ztp=self.sem(self.text,self.tp,pi,512);ztv=self.sem(self.text,self.tf,vi,512);zvp=self.sem(self.visual,self.vp,pi,128);zvv=self.sem(self.visual,self.vf,vi,128)
  zap,_=attribute_z(self.mats,self.ap,self.pusers,pi,256,self.aw);zav,_=attribute_z(self.mats,self.af,self.vusers,vi,256,self.aw);bg=fit_backgrounds(pi,ztp,zap,zvp,self.n_items);full_scores,_=score_coliftrec(vs,vi,ztv,zav,zvv,bg,self.params,self.enabled);full=metrics_at(rank_by_score(vi,full_scores),self.vusers,self.vsets)
  return {'backbone':back,'colift':full,'diffusion_inference_calls':0}

def _tensor3(model,u,p,n):
 d=model.device
 return (torch.as_tensor(u,device=d,dtype=torch.long),torch.as_tensor(p,device=d,dtype=torch.long),torch.as_tensor(n,device=d,dtype=torch.long))

def audit_losses(seed):
 plan=np.load(RDIR/'assets'/f'seed{seed}_normal_train_plan.npz');ix=np.flatnonzero((plan['epoch']==1)&(plan['batch']==0));normal_np=(plan['user'][ix],plan['positive'][ix],plan['normal_negative'][ix])
 model,ck,ds,tr,a=load_start(seed,False);model.train();normal=_tensor3(model,*normal_np)
 seed_all(20261999);direct=model.calculate_loss(normal)
 seed_all(20261999);wrapped,parts0=round19_loss(model,normal,None,0.0)
 loss_diff=float(abs(direct.detach()-wrapped.detach()))
 z=np.load(RDIR/'assets'/f'seed{seed}_A4_aux_map.npz');ze=np.flatnonzero(z['epoch']==1)[:len(ix)];aux=_tensor3(model,z['user'][ze],z['positive'][ze],z['negative'][ze])
 seed_all(20261998);tot,parts=round19_loss(model,normal,aux,LAMBDA_HARD)
 vals={k:float(v.detach()) for k,v in parts.items()};aux_err=abs((vals['L_total']-vals['L_MSCA'])-LAMBDA_HARD*vals['L_aux'])
 out={'seed':seed,'direct_calculate_loss':float(direct.detach()),'round19_wrapper_lambda0':float(wrapped.detach()),'abs_loss_diff':loss_diff,'LOSS_PARITY_PASS':loss_diff<1e-6,'aux_parts':vals,'aux_identity_error':float(aux_err),'AUX_LOSS_PASS':aux_err<1e-6,'normal_negative_present':True,'auxiliary_batch_independent':True,'lambda_hard':LAMBDA_HARD,'TEST_ACCESSED':False}
 del model;torch.cuda.empty_cache();gc.collect();return out

def _make_aux_order(n_events,total,seed):
 rng=np.random.default_rng(seed);p=rng.permutation(n_events);reps=int(np.ceil(total/len(p)));return np.tile(p,reps)[:total]

def save_model(model,seed,variant,epoch):
 p=RDIR/'outputs'/f'seed{seed}/{variant}/epoch{epoch}.pt';p.parent.mkdir(parents=True,exist_ok=True);torch.save({'protocol':PROTOCOL,'seed':seed,'variant':variant,'epoch':epoch,'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'TEST_ACCESSED':False},p);return p

def train_variant(seed,variant,a0,evaluator,smoke=False):
 model,ck,ds,tr,a=load_start(seed,False);c=ck['config'];lr=float(c['learning_rate'])*.1;opt=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=float(c['weight_decay']))
 plan=np.load(RDIR/'assets'/f'seed{seed}_normal_train_plan.npz');ed=EventData(seed);amap=None if variant=='A1' else np.load(RDIR/'assets'/f'seed{seed}_{variant}_aux_map.npz')
 logs=[];epochs=1 if smoke else EPOCHS
 for ep in range(1,epochs+1):
  eidx=np.flatnonzero(plan['epoch']==ep);batch_ids=np.unique(plan['batch'][eidx]);order=_make_aux_order(ed.n,len(eidx),AUX_ORDER_SEEDS[ep-1]);cursor=0;sumloss=0.;nsteps=0
  model.train()
  for bi in batch_ids:
   ix=np.flatnonzero((plan['epoch']==ep)&(plan['batch']==bi));normal=_tensor3(model,plan['user'][ix],plan['positive'][ix],plan['normal_negative'][ix]);aux=None;lam=0.0
   if variant!='A1':
    ai=order[cursor:cursor+len(ix)];cursor+=len(ix);zix=ai+(ep-1)*ed.n;aux=_tensor3(model,amap['user'][zix],amap['positive'][zix],amap['negative'][zix]);lam=LAMBDA_HARD
   opt.zero_grad(set_to_none=True);loss,parts=round19_loss(model,normal,aux,lam)
   if not torch.isfinite(loss):raise RuntimeError(f'nonfinite loss seed{seed} {variant} ep{ep} batch{bi}')
   loss.backward();opt.step();sumloss+=float(loss.detach());nsteps+=1
   if smoke:break
  ev=evaluator.evaluate(model);rec={'epoch':ep,'steps':nsteps,'mean_total_loss':sumloss/max(nsteps,1),'evaluation':ev,'backbone_vs_A0':delta_pack(ev['backbone'],a0['backbone']),'colift_vs_A0':delta_pack(ev['colift'],a0['colift']),'diffusion_inference_calls':0};logs.append(rec);print(json.dumps({'seed':seed,'variant':variant,'epoch':ep,'steps':nsteps,'U_backbone':rec['backbone_vs_A0']['U'],'U_colift':rec['colift_vs_A0']['U']}),flush=True)
 if not smoke:p=save_model(model,seed,variant,EPOCHS)
 out={'seed':seed,'variant':variant,'starting_checkpoint':a['checkpoint'],'starting_checkpoint_sha256':a['checkpoint_sha256'],'optimizer':'Adam','lr':lr,'weight_decay':float(c['weight_decay']),'lambda_hard':0.0 if variant=='A1' else LAMBDA_HARD,'epochs':epochs,'fixed_final_epoch':epochs,'validation_used_for_epoch_selection':False,'trajectory':logs,'final':logs[-1],'checkpoint':None if smoke else str(p),'TEST_ACCESSED':False}
 if not smoke:
  q=RDIR/'evidence'/f'ROUND19_{variant}_SEED{seed}.json';q.write_text(json.dumps(out,indent=2)+'\n')
 del model,opt;torch.cuda.empty_cache();gc.collect();return out

# Strict TRAIN-only override for safe-pool construction. This definition intentionally
# replaces the earlier convenience implementation that routed through ValidationEvaluator.
def build_frozen_safe_pool(seed):
 import pandas as pd
 from modules.ranking import semantic_z_for_candidates
 model,ck,ds,tr,a=load_start(seed,False);cfg=load_dataset_config('baby');p=cfg['resolved_paths'];n_users=model.n_users;n_items=model.n_items
 df=pd.read_csv(p['interaction'],sep='\t',usecols=['userID','itemID','timestamp','x_label']);df['_row']=np.arange(len(df));tdf=df[df.x_label==0].sort_values(['userID','timestamp','_row'],kind='stable')
 histories=[[] for _ in range(n_users)]
 for u,g in tdf.groupby('userID',sort=False):histories[int(u)]=g.itemID.astype(np.int64).tolist()
 users=np.asarray([u for u,h in enumerate(histories) if h],np.int64);pseudo=[list(h[:-1]) if len(h)>=2 else list(h) for h in histories];pusers=np.asarray([u for u,h in enumerate(histories) if len(h)>=2],np.int64)
 model.eval()
 with torch.no_grad():fu,fi=model.forward(test=True);ci,cs=topk_from_embeddings(fu,fi,users,histories,100,1024);pi,ps=topk_from_embeddings(fu,fi,pusers,pseudo,100,1024)
 ztp,_=semantic_z_for_candidates(p['text_feature'],pseudo,pusers,pi,128);ztf,_=semantic_z_for_candidates(p['text_feature'],histories,users,ci,128);zvp,_=semantic_z_for_candidates(p['visual_feature'],pseudo,pusers,pi,64);zvf,_=semantic_z_for_candidates(p['visual_feature'],histories,users,ci,64)
 ac=cfg['coliftrec']['attribute'];mats,_=build_item_matrices(p['metadata'],n_items,min_df=int(ac.get('tfidf_min_df',2)),max_df=float(ac.get('tfidf_max_df',.8)),description_len=int(ac.get('description_len',128)),weights=ac.get('weights'));pp=build_profiles(mats,pseudo,n_items);fp=build_profiles(mats,histories,n_items);zap,_=attribute_z(mats,pp,pusers,pi,256,ac.get('weights'));zaf,_=attribute_z(mats,fp,users,ci,256,ac.get('weights'))
 bg=fit_backgrounds(pi,ztp,zap,zvp,n_items);cc=cfg['coliftrec'];params=_params(cc);enabled={m:bool(cc[m]['enabled']) for m in ('text','attribute','visual')};full,_=score_coliftrec(cs,ci,ztf,zaf,zvf,bg,params,enabled);order=np.argsort(-full,axis=1,kind='stable');ranked=np.take_along_axis(ci,order,axis=1).astype(np.int32);fs=np.take_along_axis(full,order,axis=1).astype(np.float32);bs=np.take_along_axis(cs,order,axis=1).astype(np.float32)
 viol=0
 for r,u in enumerate(users):
  obs=set(histories[int(u)]);viol+=sum(int(j) in obs for j in ranked[r,20:100])
 path=RDIR/'assets'/f'seed{seed}_safe_pool.npz';np.savez_compressed(path,users=users,items=ranked,full_scores=fs,backbone_scores=bs)
 out={'seed':seed,'source':'TRAIN-only starting backbone Top100 reranked by frozen Full CoLiftRec','background_fit_source':'TRAIN pseudo history only','validation_labels_used':False,'users':int(len(users)),'shape':list(ranked.shape),'safe_pool_rank':[21,100],'protected_rank':[1,20],'train_observed_collisions_rank21_100':int(viol),'coverage':1.0,'sha256':sha256_file(path),'TEST_ACCESSED':False}
 del model;torch.cuda.empty_cache();gc.collect();return out
