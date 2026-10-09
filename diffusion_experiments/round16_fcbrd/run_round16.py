from __future__ import annotations
import argparse, gc, json, random, shutil, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,str(ROOT))
from utils.dataloader import TrainDataLoader
from pipelines.msca_assets import load_msca_checkpoint
from modules.ranking import metrics_at, rank_by_score
from pipelines.coliftrec import PRIMARY,ALL
from diffusion_experiments.round9_cabrp import run_round9 as r9
from diffusion_experiments.round14_aihu import run_round14 as r14
from diffusion_experiments.round15_baurp import run_round15 as r15run
from diffusion_experiments.round15_baurp import base_anchored_residual as r15
from diffusion_experiments.round16_fcbrd import frozen_reference as fr
from diffusion_experiments.round16_fcbrd import colift_context as cc

PROTOCOL='ROUND16_FCBRD_V1'; SOURCE='cf9638237cae2a7ff6372b85131920ba8fd33630'
PREFLIGHT=(999,1000); EXPANSION=(1001,1002); VARIANTS=('A1','A2','A3')
EPOCHS=5; LR_SCALE=.1; PAIR_BATCH_USERS=256; DIAG_USERS=1024
RANDOM_SEEDS=tuple(range(20261601,20261609))
BASE_SEED_OFFSET=202614000


def seed_all(s):
 random.seed(int(s)); np.random.seed(int(s)); torch.manual_seed(int(s)); torch.cuda.manual_seed_all(int(s))
def stat(x):
 x=np.asarray(x,np.float64); return {'mean':float(x.mean()),'median':float(np.median(x)),'fraction_positive':float(np.mean(x>0)),'p10':float(np.quantile(x,.1)),'p90':float(np.quantile(x,.9))}
def stat_ext(x):
 x=np.asarray(x,np.float64); return {'mean':float(x.mean()),'median':float(np.median(x)),'p90':float(np.quantile(x,.9)),'p95':float(np.quantile(x,.95)),'p99':float(np.quantile(x,.99)),'max':float(x.max())}
def utility(m,b): return float(np.mean([(m[k]-b[k])/b[k] for k in PRIMARY]))
def delta_pack(m,b):
 return {'absolute_delta':{k:float(m[k]-b[k]) for k in ALL},'relative_delta':{k:float((m[k]-b[k])/b[k]) for k in ALL},
         'U':utility(m,b),'primary_positive_count':int(sum(m[k]>b[k] for k in PRIMARY)),'overall_positive_count':int(sum(m[k]>b[k] for k in ALL))}
def sync(d):
 if str(d).startswith('cuda'): torch.cuda.synchronize(d)
def checkpoint_audit(seed):
 a=json.loads((r9.paths(seed)['msca']/'audit.json').read_text())
 if a.get('TEST_ACCESSED') is not False: raise RuntimeError('backbone audit contaminated')
 return a


def load_backbone(seed):
 a=checkpoint_audit(seed); model,ck,_,train_dataset=load_msca_checkpoint(Path(a['checkpoint']),0); fr.freeze_recommender(model)
 cfg=ck['config']; train=TrainDataLoader(cfg,train_dataset,batch_size=cfg['train_batch_size'],shuffle=True)
 seed_all(202614700+seed); train.pretrain_setup()
 with torch.no_grad(): c={k:v.detach() for k,v in fr.forward_components(model).items()}
 ab=fr.cosine_alpha_bar(device=model.device)
 return model,train,cfg,a,c,ab


def base_dir(root,seed): return root/f'seed{seed}'/'BASE'
def base_probe(base,c):
 ids=torch.arange(min(128,c['collab_item'].shape[0]),device=c['collab_item'].device); t=torch.full((len(ids),),3,device=ids.device,dtype=torch.long)
 ucf=torch.zeros_like(c['collab_item'][ids]); icf=c['collab_item'][ids]; g=torch.Generator(device=ids.device); g.manual_seed(20261699)
 out=[]
 with torch.no_grad():
  for den,h in ((base.text,c['text_item'][ids]),(base.visual,c['image_item'][ids])):
   z=torch.randn((len(ids),fr.LATENT_DIM),generator=g,device=ids.device); _,ht=fr.noisy_state(h,t,fr.cosine_alpha_bar(device=ids.device),z)
   out.append(den(ht,t,fr.base_condition(base,ucf,icf)).detach().cpu())
 return torch.cat(out).numpy()


def train_base(seed,root,max_batches=None,force=False):
 out=base_dir(root,seed); cp=out/'BASE_REFERENCE.pt'; rp=out/'result.json'
 if cp.exists() and rp.exists() and not force: return json.loads(rp.read_text())
 shutil.rmtree(out,ignore_errors=True); out.mkdir(parents=True,exist_ok=True)
 model,train,cfg,a,c,ab=load_backbone(seed); seed_all(BASE_SEED_OFFSET+seed); base=fr.BaseModules().to(model.device)
 opt=torch.optim.Adam(base.parameters(),lr=float(cfg['learning_rate'])*LR_SCALE,weight_decay=float(cfg['weight_decay'] or 0.0)); logs=[]
 for ep in range(1,EPOCHS+1):
  seed_all(seed*10000+ep); base.train(); sums={'L_base':0.,'L_base_text':0.,'L_base_visual':0.}; n=0; tmin=99;tmax=-1
  for bi,it in enumerate(train):
   if max_batches is not None and bi>=max_batches: break
   opt.zero_grad(set_to_none=True); loss,p=fr.base_reconstruction_loss(base,c,it,ab); loss.backward(); opt.step(); n+=1
   for k in sums:sums[k]+=float(p[k].detach())
   tmin=min(tmin,int(p['t_min']));tmax=max(tmax,int(p['t_max']))
  rec={k:v/max(n,1) for k,v in sums.items()}; rec.update({'epoch':ep,'batches':n,'t_range':[tmin,tmax]}); logs.append(rec)
  print(json.dumps({'stage':'R','seed':seed,**rec},sort_keys=True),flush=True)
 for p in base.parameters(): p.requires_grad=False
 h=fr.state_sha256(base); probe=base_probe(base,c)
 torch.save({'protocol':PROTOCOL,'seed':seed,'epoch':5,'state':{k:v.detach().cpu() for k,v in base.state_dict().items()},'sha256':h,'TEST_ACCESSED':False},cp)
 np.save(out/'BASE_REFERENCE_PROBE.npy',probe)
 result={'status':'PASS','stage':'R','seed':seed,'epochs':5,'selected_epoch':5,'selection_used_validation':False,'logs':logs,
         'parameter_count':sum(p.numel() for p in base.parameters()),'trainable_parameter_count':sum(p.numel() for p in base.parameters() if p.requires_grad),
         'state_dict_sha256':h,'probe_path':str((out/'BASE_REFERENCE_PROBE.npy').resolve()),'TEST_ACCESSED':False}
 rp.write_text(json.dumps(result,indent=2)+'\n'); del model,base,opt; torch.cuda.empty_cache(); gc.collect(); return result


def load_base(seed,root,device,c=None):
 p=torch.load(base_dir(root,seed)/'BASE_REFERENCE.pt',map_location='cpu',weights_only=False); b=fr.BaseModules().to(device); b.load_state_dict(p['state'],strict=True)
 for x in b.parameters(): x.requires_grad=False
 b.eval(); return b,p

class TrainContext:
 def __init__(self,path,n_users,n_items,device):
  z=np.load(path); self.n_items=n_items; self.device=device
  users=z['users'].astype(np.int64); self.row=np.full(n_users,-1,np.int64); self.row[users]=np.arange(len(users))
  self.items=z['ranked_items'].astype(np.int64); self.ctx=z['ranked_context'].astype(np.float32); self.score=z['ranked_full_scores'].astype(np.float32)
  self.pkeys=z['positive_keys'].astype(np.int64); self.pctx=z['positive_context'].astype(np.float32); self.pscore=z['positive_full_scores'].astype(np.float32)
 def sample(self,users,pos):
  un=users.detach().cpu().numpy().astype(np.int64); pn=pos.detach().cpu().numpy().astype(np.int64); rr=self.row[un]
  if np.any(rr<0): raise RuntimeError('TRAIN context missing user')
  col=np.random.randint(5,30,size=len(un)); neg=self.items[rr,col]; ns=self.score[rr,col]; nc=self.ctx[rr,col]
  keys=un*self.n_items+pn; ix=np.searchsorted(self.pkeys,keys)
  ok=(ix<len(self.pkeys)) & (self.pkeys[np.minimum(ix,len(self.pkeys)-1)]==keys)
  if not np.all(ok): raise RuntimeError(f'positive context missing {int((~ok).sum())}')
  ps=self.pscore[ix]; pc=self.pctx[ix]
  return (torch.as_tensor(neg,device=self.device,dtype=torch.long),torch.as_tensor(np.concatenate([ps,ns]),device=self.device),
          torch.as_tensor(pc,device=self.device),torch.as_tensor(nc,device=self.device),torch.as_tensor(col+1,device=self.device))

class ValidationEvaluator(r15run.ValidationEvaluator):
 def __init__(self,seed,ctx_path):
  super().__init__(seed); z=np.load(ctx_path); u=z['users'].astype(np.int64); it=z['items'].astype(np.int32)
  if not np.array_equal(u,self.users) or not np.array_equal(it,self.items): raise RuntimeError('Validation context identity mismatch')
  self.ctx=z['context'].astype(np.float32)
 @torch.no_grad()
 def evaluate(self,model,base,user,c,variant,mode='true',control_seed=None,batch_users=PAIR_BATCH_USERS):
  model.eval(); base.eval(); user.eval(); sync(model.device); t0=time.time(); ab=fr.cosine_alpha_bar(device=model.device); noise=fr.build_noise_tables(model.n_items,model.device)
  width=self.items.shape[1]; delta=np.empty_like(self.full,dtype=np.float32); rn=0.;rc=0
  for r0 in range(0,len(self.users),batch_users):
   r1=min(r0+batch_users,len(self.users)); uu=torch.as_tensor(self.users[r0:r1],device=model.device); ids=torch.as_tensor(self.items[r0:r1].reshape(-1),device=model.device); ur=uu.repeat_interleave(width)
   ctx=None if variant!='A3' else torch.as_tensor(self.ctx[r0:r1].reshape(-1,8),device=model.device)
   bb,aa,rt,rv=fr.pair_scores(model,base,user,c,ur,ids,ctx,ab,noise,mode,control_seed)
   delta[r0:r1]=(aa-bb).float().cpu().numpy().reshape(r1-r0,width); rn+=float(rt.norm(dim=1).sum()+rv.norm(dim=1).sum());rc+=2*len(ids)
  final=self.full+delta; rank=rank_by_score(self.items,final); met=metrics_at(rank,self.users,self.eval_sets); sync(model.device)
  return {'metrics':met,'rank':rank,'final_scores':final,'score_delta':delta,'mean_residual_norm':rn/max(rc,1),'inference_seconds':time.time()-t0,'mode':mode,'control_seed':control_seed}


def context_paths(seed):
 d=ROOT/'diffusion_experiments/round16_fcbrd/assets'; return d/f'seed{seed}_train.npz',d/f'seed{seed}_train_audit.json',d/f'seed{seed}_validation.npz',d/f'seed{seed}_validation_audit.json'
def ensure_context(seed):
 tp,ta,vp,va=context_paths(seed)
 if not tp.exists(): cc.build_train_context(seed,tp,ta,None)
 if not vp.exists(): cc.build_validation_context(seed,vp,va,None)
 return tp,json.loads(ta.read_text()),vp,json.loads(va.read_text())


def a0_parity(seed):
 old=json.loads((ROOT/f'diffusion_experiments/round15_baurp/outputs/seed{seed}/A2/result.json').read_text())
 model,_,_,_,c,_=load_backbone(seed); mods=r15.BAURPModules().to(model.device); r15run.restore(ROOT/f'diffusion_experiments/round15_baurp/outputs/seed{seed}/A2/best_checkpoint.pt',mods); ev=r15run.ValidationEvaluator(seed)
 now=ev.evaluate_variant(model,mods,c,'A2','true'); diff=max(abs(now['metrics'][k]-old['best_metrics'][k]) for k in ALL); out={'seed':seed,'max_abs_diff':diff,'metrics':now['metrics'],'round15':old['best_metrics'],'PASS':diff<=1e-8,'TEST_ACCESSED':False}
 del model,mods;torch.cuda.empty_cache();return out

def save_user(path,user,epoch,variant,seed):
 torch.save({'protocol':PROTOCOL,'seed':seed,'variant':variant,'epoch':epoch,'state':{k:v.detach().cpu() for k,v in user.state_dict().items()},'TEST_ACCESSED':False},path)
def restore_user(path,user):
 p=torch.load(path,map_location='cpu',weights_only=False); user.load_state_dict(p['state'],strict=True); return p


def hard_shell_diag(model,base,user,c,ev,variant,max_users=DIAG_USERS,batch=4096):
 us=[]; pi=[]; ni=[]; ps=[]; ns=[]; pc=[]; nc=[]; eligible=0
 for r,u in enumerate(ev.users[:min(max_users,len(ev.users))]):
  item_to_col={int(x):j for j,x in enumerate(ev.items[r])}; poscols=[item_to_col[int(x)] for x in ev.eval_sets[int(u)] if int(x) in item_to_col]
  if not poscols: continue
  eligible+=1; pcol=poscols[0]; pitem=int(ev.items[r,pcol])
  for x in ev.c0_rank[r,5:30]:
   x=int(x)
   if x in ev.eval_sets[int(u)]: continue
   j=item_to_col[x]; us.append(int(u));pi.append(pitem);ni.append(x);ps.append(float(ev.full[r,pcol]));ns.append(float(ev.full[r,j]))
   if variant=='A3': pc.append(ev.ctx[r,pcol]);nc.append(ev.ctx[r,j])
 if not us: raise RuntimeError('no Validation hard-shell diagnostic pairs')
 arr={k:[] for k in ('Dtrue','Dshuf','mtrue','mbase','mminus','lplus_base','lminus_base','lplus_minus','angle_t','angle_v','res_t','res_v','score_delta')}
 ab=fr.cosine_alpha_bar(device=model.device); noise=fr.build_noise_tables(model.n_items,model.device)
 for s in range(0,len(us),batch):
  e=min(s+batch,len(us)); uu=torch.as_tensor(us[s:e],device=model.device); p=torch.as_tensor(pi[s:e],device=model.device); n=torch.as_tensor(ni[s:e],device=model.device)
  ids=torch.cat([p,n]); ur=torch.cat([uu,uu]); ctx=None
  if variant=='A3': ctx=torch.as_tensor(np.concatenate([np.asarray(pc[s:e]),np.asarray(nc[s:e])]),device=model.device)
  full=torch.as_tensor(np.concatenate([ps[s:e],ns[s:e]]),device=model.device)
  with torch.no_grad():
   b0,a0,rt,rv,d=fr.pair_scores(model,base,user,c,ur,ids,ctx,ab,noise,'true',None,True)
   b1,a1,_,_=fr.pair_scores(model,base,user,c,ur,ids,ctx,ab,noise,'shuffled')
   bm,am,_,_=fr.pair_scores(model,base,user,c,ur,ids,ctx,ab,noise,'negated')
  q=e-s; mt=(full+a0-b0)[:q]-(full+a0-b0)[q:]; mb=full[:q]-full[q:]; ms=(full+a1-b1)[:q]-(full+a1-b1)[q:]; mm=(full+am-bm)[:q]-(full+am-bm)[q:]
  dt=mt-mb; ds=ms-mb; lp=F.softplus(-mt); lb=F.softplus(-mb); lm=F.softplus(-mm)
  for k,v in [('Dtrue',dt),('Dshuf',ds),('mtrue',mt),('mbase',mb),('mminus',mm),('lplus_base',lp-lb),('lminus_base',lm-lb),('lplus_minus',lp-lm)]: arr[k].append(v.cpu().numpy())
  arr['angle_t'].append(d['text']['angle_deg'].cpu().numpy());arr['angle_v'].append(d['visual']['angle_deg'].cpu().numpy());arr['res_t'].append(rt.norm(dim=1).cpu().numpy());arr['res_v'].append(rv.norm(dim=1).cpu().numpy());arr['score_delta'].append((a0-b0).cpu().numpy())
 z={k:np.concatenate(v) for k,v in arr.items()}; D=z['Dtrue']; I=D-z['Dshuf']; L=z['lplus_base']
 return {'sample_users':min(max_users,len(ev.users)),'eligible_users':eligible,'pair_count':len(us),'definition':'first 1024 Validation users; frozen C0 ranks 6-30; all non-positive candidates; positive must occur in candidate Top100',
   'Delta_true':stat(D),'Delta_identity':stat(I),'fraction_Delta_true_gt0':float(np.mean(D>0)),'fraction_Delta_true_gt_shuf':float(np.mean(I>0)),
   'L_plus_minus_base':stat(L),'L_minus_minus_base':stat(z['lminus_base']),'L_plus_minus_minus':stat(z['lplus_minus']),
   'fraction_Lplus_lt_base':float(np.mean(L<0)),'fraction_Lplus_lt_minus':float(np.mean(z['lplus_minus']<0)),
   'angle_text':stat_ext(z['angle_t']),'angle_visual':stat_ext(z['angle_v']),'residual_text':stat_ext(z['res_t']),'residual_visual':stat_ext(z['res_v']),
   'score_delta':stat_ext(np.abs(z['score_delta'])),'near_zero_action_fraction':float(np.mean(np.abs(z['score_delta'])<1e-6)),'TEST_ACCESSED':False}


def train_epoch(model,base,user,opt,c,ab,train,tc,variant,epoch,seed,max_batches=None):
 seed_all(seed*10000+epoch); model.eval();base.eval();user.train(); keys=['L_rec','L_gain','L_identity','L_zero','L_bound','total_loss','m_base','m_true','m_shuf','Delta_true_mean','Delta_true_median','fraction_Delta_true_gt0','Delta_identity_mean','fraction_Delta_true_gt_shuf','L_plus','L_pair_base','L_minus','L_plus_minus_base','L_minus_minus_base','L_plus_minus_minus','fraction_Lplus_lt_base','fraction_Lplus_lt_minus','near_zero_action_fraction','raw_residual_text_true','raw_residual_visual_true','cap_fraction_text','cap_fraction_visual','angle_text_mean','angle_text_max','angle_visual_mean','angle_visual_max']; sums={k:0. for k in keys};n=0;rmin=99;rmax=-1
 for bi,it in enumerate(train):
  if max_batches is not None and bi>=max_batches:break
  users,pos=it[0],it[1]; neg,full,pc,nc,ranks=tc.sample(users,pos); cp=pc if variant=='A3' else None; cn=nc if variant=='A3' else None
  opt.zero_grad(set_to_none=True); loss,p=fr.preference_terms(model,base,user,c,users,pos,neg,full,ab,variant,cp,cn); loss.backward()
  if any(x.grad is not None for x in base.parameters()) or any(x.grad is not None for x in model.parameters()): raise RuntimeError('frozen parameter received gradient')
  clip=model.config['clip_grad_norm'] if hasattr(model,'config') else None
  if clip: torch.nn.utils.clip_grad_norm_(list(user.parameters()),**clip)
  opt.step();n+=1
  for k in keys:sums[k]+=float(p[k].detach())
  rmin=min(rmin,int(ranks.min()));rmax=max(rmax,int(ranks.max()))
 out={k:v/max(n,1) for k,v in sums.items()};out.update({'batches':n,'hard_rank_min':rmin,'hard_rank_max':rmax,'preference_t_min':3,'preference_t_max':3,'base_hash':fr.state_sha256(base)})
 return out

def run_variant(seed,variant,root,max_batches=None):
 tp,ta,vp,va=ensure_context(seed); out=root/f'seed{seed}'/variant; shutil.rmtree(out,ignore_errors=True);out.mkdir(parents=True,exist_ok=True)
 model,train,cfg,a,c,ab=load_backbone(seed); base,bmeta=load_base(seed,root,model.device); init_hash=fr.state_sha256(base); init_probe=base_probe(base,c)
 user=fr.clone_user_from_base(base,8 if variant=='A3' else 0).to(model.device); opt=torch.optim.Adam(user.parameters(),lr=float(cfg['learning_rate'])*LR_SCALE,weight_decay=float(cfg['weight_decay'] or 0.0)); tc=TrainContext(tp,model.n_users,model.n_items,model.device); ev=ValidationEvaluator(seed,vp); logs=[]
 for ep in range(1,EPOCHS+1):
  tr=train_epoch(model,base,user,opt,c,ab,train,tc,variant,ep,seed,max_batches); val=ev.evaluate(model,base,user,c,variant,'true'); mech=hard_shell_diag(model,base,user,c,ev,variant)
  sane=mech['Delta_true']['mean']>0 and mech['Delta_identity']['mean']>0; cp=out/f'checkpoint_ep{ep}.pt'; save_user(cp,user,ep,variant,seed)
  rec={'epoch':ep,'train':tr,'validation_metrics':val['metrics'],'U_vs_C0':utility(val['metrics'],ev.c0_metrics),'mechanism':mech,'mechanism_sane':bool(sane),'base_hash':fr.state_sha256(base),'base_probe_max_abs_diff':float(np.max(np.abs(base_probe(base,c)-init_probe)))};logs.append(rec)
  print(json.dumps({'seed':seed,'variant':variant,'epoch':ep,'U':rec['U_vs_C0'],'R20':val['metrics']['R20'],'D':mech['Delta_true']['mean'],'Dfrac':mech['fraction_Delta_true_gt0'],'I':mech['Delta_identity']['mean'],'Ifrac':mech['fraction_Delta_true_gt_shuf'],'dirfrac':mech['fraction_Lplus_lt_base']},sort_keys=True),flush=True)
 sane=[x for x in logs if x['mechanism_sane']]; ineligible=not bool(sane); candidates=sane if sane else logs; chosen=max(candidates,key=lambda x:x['validation_metrics']['R20']); bep=chosen['epoch']; shutil.copy2(out/f'checkpoint_ep{bep}.pt',out/'best_checkpoint.pt');restore_user(out/'best_checkpoint.pt',user)
 true=ev.evaluate(model,base,user,c,variant,'true'); shuf=ev.evaluate(model,base,user,c,variant,'shuffled'); zero=ev.evaluate(model,base,user,c,variant,'zero'); neg=ev.evaluate(model,base,user,c,variant,'negated'); mech=hard_shell_diag(model,base,user,c,ev,variant)
 basehash=fr.state_sha256(base); probe_diff=float(np.max(np.abs(base_probe(base,c)-init_probe))); maxang=max(mech['angle_text']['max'],mech['angle_visual']['max'])
 result={'status':'PASS','protocol':PROTOCOL,'seed':seed,'variant':variant,'epochs':5,'best_epoch':bep,'selection_policy':'mechanism-sane: mean Delta_true>0 AND mean(Delta_true-Delta_shuf)>0; then max Validation R20; else max R20 report-only','PASS_INELIGIBLE':ineligible,
  'C0_metrics':ev.c0_metrics,'best_metrics':true['metrics'],'best_vs_C0':delta_pack(true['metrics'],ev.c0_metrics),'identity_ablation':{'true':delta_pack(true['metrics'],ev.c0_metrics),'shuffled':delta_pack(shuf['metrics'],ev.c0_metrics),'zero':delta_pack(zero['metrics'],ev.c0_metrics)},'negated':delta_pack(neg['metrics'],ev.c0_metrics),
  'hard_shell_mechanism':mech,'epoch_logs':logs,'reference':{'initial_hash':init_hash,'final_hash':basehash,'hash_unchanged':init_hash==basehash,'probe_max_abs_diff':probe_diff,'zero_ranking_exact':bool(np.array_equal(zero['rank'],ev.c0_rank))},
  'bound':{'max_angle':maxang,'cap_warning':bool(max(logs[bep-1]['train']['cap_fraction_text'],logs[bep-1]['train']['cap_fraction_visual'])>.8)},'context_audit':va if variant=='A3' else None,'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n'); np.savez_compressed(out/'best_validation_rankings.npz',users=ev.users,ranked_items=true['rank'],candidate_items=ev.items,final_scores=true['final_scores'])
 del model,base,user,opt;torch.cuda.empty_cache();gc.collect();return result


def random_controls(seed,variant,root):
 p=root/f'seed{seed}'/variant/'result.json';r=json.loads(p.read_text());model,_,_,_,c,_=load_backbone(seed);base,_=load_base(seed,root,model.device);user=fr.clone_user_from_base(base,8 if variant=='A3' else 0).to(model.device);restore_user(root/f'seed{seed}'/variant/'best_checkpoint.pt',user);_,_,vp,_=context_paths(seed);ev=ValidationEvaluator(seed,vp)
 vals=[]
 for rs in RANDOM_SEEDS:
  x=ev.evaluate(model,base,user,c,variant,'random',rs); vals.append({'seed':rs,'metrics':x['metrics'],'vs_C0':delta_pack(x['metrics'],ev.c0_metrics)})
 trueU=r['best_vs_C0']['U']; us=np.asarray([x['vs_C0']['U'] for x in vals]); out={'seed':seed,'variant':variant,'random_seeds':list(RANDOM_SEEDS),'controls':vals,'mean_U_random':float(us.mean()),'std_U_random':float(us.std()),'min_U_random':float(us.min()),'max_U_random':float(us.max()),'U_true':trueU,'U_negated':r['negated']['U'],'true_percentile_vs_random':float(100*np.mean(us<trueU)),'PASS':bool(trueU>us.mean() and trueU>r['negated']['U']),'TEST_ACCESSED':False}
 del model,base,user;torch.cuda.empty_cache();return out

def _smoke_batch(tc,n_items,device,n=16):
 users=np.flatnonzero(tc.row>=0)[:n].astype(np.int64); pos=[]
 for u in users:
  lo=np.searchsorted(tc.pkeys,int(u)*n_items); hi=np.searchsorted(tc.pkeys,(int(u)+1)*n_items)
  if lo>=hi: raise RuntimeError('smoke user has no positive context')
  pos.append(int(tc.pkeys[lo]%n_items))
 return torch.as_tensor(users,device=device),torch.as_tensor(pos,device=device)

def _grad_norm(mod):
 return float(sum(float(p.grad.detach().float().pow(2).sum()) for p in mod.parameters() if p.grad is not None)**.5)

def smoke(root,evid):
 sr=ROOT/'diffusion_experiments/round16_fcbrd/outputs_smoke'; shutil.rmtree(sr,ignore_errors=True);sr.mkdir(parents=True,exist_ok=True); audits={'protocol':PROTOCOL,'seeds':{},'TEST_ACCESSED':False}
 for seed in PREFLIGHT:
  train_sm=ROOT/f'diffusion_experiments/round16_fcbrd/assets/seed{seed}_train_smoke.npz';val_sm=ROOT/f'diffusion_experiments/round16_fcbrd/assets/seed{seed}_val_smoke.npz'
  if not train_sm.exists(): cc.build_train_context(seed,train_sm,train_sm.with_name(train_sm.stem+'_audit.json'),32)
  if not val_sm.exists(): cc.build_validation_context(seed,val_sm,val_sm.with_name(val_sm.stem+'_audit.json'),32)
  bres=train_base(seed,sr,max_batches=1,force=True); model,train,cfg,a,c,ab=load_backbone(seed); base,bm=load_base(seed,sr,model.device); h0=fr.state_sha256(base);p0=base_probe(base,c);tc=TrainContext(train_sm,model.n_users,model.n_items,model.device);uu,pp=_smoke_batch(tc,model.n_items,model.device); neg,full,pc,nc,ranks=tc.sample(uu,pp)
  seedrow={'base':bres,'base_hash_initial':h0,'variants':{},'A0_parity':a0_parity(seed)}
  if not seedrow['A0_parity']['PASS']: raise RuntimeError('A0 parity FAIL')
  for v in VARIANTS:
   user=fr.clone_user_from_base(base,8 if v=='A3' else 0).to(model.device); cp=pc if v=='A3' else None;cn=nc if v=='A3' else None
   user.zero_grad(set_to_none=True);loss,parts=fr.preference_terms(model,base,user,c,uu,pp,neg,full,ab,v,cp,cn);loss.backward(); gtot=_grad_norm(user)
   frozen_none=all(p.grad is None for p in base.parameters()) and all(p.grad is None for p in model.parameters()); finite=bool(torch.isfinite(loss)); maxang=max(float(parts['angle_text_max']),float(parts['angle_visual_max']))
   # One real optimization step makes ZERO-consistency active; smoke outputs are isolated from formal.
   optu=torch.optim.Adam(user.parameters(),lr=float(cfg['learning_rate'])*LR_SCALE,weight_decay=float(cfg['weight_decay'] or 0.0)); optu.step(); user.zero_grad(set_to_none=True)
   if not finite or gtot<=0 or not frozen_none or maxang>5.01: raise RuntimeError(f'{v} smoke failed')
   # Component gradient audit; bound may be inactive under cap, which is separately audited by hard-cap exactness.
   comp={}
   for lk in (('L_rec' if v=='A1' else 'L_gain'),'L_identity','L_zero','L_bound'):
    user.zero_grad(set_to_none=True); _,q=fr.preference_terms(model,base,user,c,uu,pp,neg,full,ab,v,cp,cn); q[lk].backward(); comp[lk]={'user_grad_norm':_grad_norm(user),'base_grad_none':all(x.grad is None for x in base.parameters()),'backbone_grad_none':all(x.grad is None for x in model.parameters()),'value':float(q[lk].detach())}
   # Bound penalty is deliberately inactive below cap. Audit its graph on a scaled real residual only.
   if comp['L_bound']['user_grad_norm']==0.0:
    user.zero_grad(set_to_none=True); ids=torch.cat([pp,neg]); ur=torch.cat([uu,uu]); icf=c['collab_item'][ids]; ucf=c['collab_user'][ur]; ctx=torch.cat([cp,cn]) if v=='A3' else None; tt=torch.full((len(ids),),3,device=ids.device,dtype=torch.long); zz=torch.randn((len(ids),fr.LATENT_DIM),device=ids.device); _,ht=fr.noisy_state(c['text_item'][ids],tt,ab,zz)
    with torch.no_grad(): pb=base.text(ht,tt,fr.base_condition(base,ucf,icf))
    pu=user.text(ht,tt,fr.user_condition(user,ucf,icf,ctx)); syn=r15.excess_penalty(c['text_item'][ids],1000.0*(pu-pb)); syn.backward(); comp['L_bound_synthetic_excess']={'value':float(syn.detach()),'user_grad_norm':_grad_norm(user),'base_grad_none':all(x.grad is None for x in base.parameters()),'backbone_grad_none':all(x.grad is None for x in model.parameters()),'training_used':False}
   seedrow['variants'][v]={'forward_backward':True,'loss':float(loss.detach()),'user_grad_norm':gtot,'frozen_grad_none':frozen_none,'max_angle':maxang,'component_gradients':comp,'context_dim':8 if v=='A3' else 0}
   del user,optu
  seedrow['base_hash_final']=fr.state_sha256(base);seedrow['base_probe_max_abs_diff']=float(np.max(np.abs(base_probe(base,c)-p0)));seedrow['base_stable']=seedrow['base_hash_final']==h0 and seedrow['base_probe_max_abs_diff']<=1e-7
  if not seedrow['base_stable']: raise RuntimeError('smoke frozen reference changed')
  audits['seeds'][str(seed)]=seedrow;del model,base;torch.cuda.empty_cache();gc.collect()
 audits['PASS']=all(x['base_stable'] and x['A0_parity']['PASS'] and all(y['forward_backward'] for y in x['variants'].values()) for x in audits['seeds'].values());evid.mkdir(parents=True,exist_ok=True);(evid/'ROUND16_PREFLIGHT_AUDIT.json').write_text(json.dumps(audits,indent=2)+'\n');print(json.dumps({'PASS':audits['PASS'],'seeds':list(audits['seeds'])},sort_keys=True));return audits

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['smoke','base','variant','random'],required=True);ap.add_argument('--seed',type=int);ap.add_argument('--variant',choices=VARIANTS);ap.add_argument('--root',default=str(ROOT/'diffusion_experiments/round16_fcbrd/outputs'));ap.add_argument('--evidence',default=str(ROOT/'diffusion_experiments/round16_fcbrd/evidence'));a=ap.parse_args();root=Path(a.root);evid=Path(a.evidence)
 if a.mode=='smoke':smoke(root,evid)
 elif a.mode=='base':print(json.dumps(train_base(a.seed,root),sort_keys=True))
 elif a.mode=='variant':print(json.dumps({'seed':a.seed,'variant':a.variant,'status':run_variant(a.seed,a.variant,root)['status']},sort_keys=True))
 else: print(json.dumps(random_controls(a.seed,a.variant,root),sort_keys=True))
if __name__=='__main__':main()
