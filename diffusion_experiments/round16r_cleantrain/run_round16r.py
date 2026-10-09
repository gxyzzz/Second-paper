from __future__ import annotations
import argparse,gc,json,random,shutil,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT))
from modules.ranking import metrics_at,rank_by_score
from pipelines.coliftrec import PRIMARY,ALL
from diffusion_experiments.round16_fcbrd import run_round16 as r16
from diffusion_experiments.round16r_cleantrain import frozen_reference as fr
PROTOCOL='ROUND16R_CLEANTRAIN_V1';PREFLIGHT=(999,1000);VARIANTS=('A1','A2','A3');EPOCHS=5;PAIR_BATCH=256;DIAG_USERS=1024;LR_SCALE=.1
RANDOM_SEEDS=tuple(range(20261601,20261609));RDIR=ROOT/'diffusion_experiments/round16r_cleantrain';OLD=ROOT/'diffusion_experiments/round16_fcbrd'
def seed_all(s): random.seed(int(s));np.random.seed(int(s));torch.manual_seed(int(s));torch.cuda.manual_seed_all(int(s))
def utility(m,b): return float(np.mean([(m[k]-b[k])/b[k] for k in PRIMARY]))
def delta_pack(m,b): return {'absolute_delta':{k:float(m[k]-b[k]) for k in ALL},'relative_delta':{k:float((m[k]-b[k])/b[k]) for k in ALL},'U':utility(m,b),'primary_positive_count':int(sum(m[k]>b[k] for k in PRIMARY)),'overall_positive_count':int(sum(m[k]>b[k] for k in ALL))}
def stat(x):
 x=np.asarray(x,np.float64);return {'mean':float(x.mean()),'median':float(np.median(x)),'fraction_positive':float(np.mean(x>0)),'p10':float(np.quantile(x,.1)),'p90':float(np.quantile(x,.9))}
def stat_ext(x):
 x=np.asarray(x,np.float64);return {'mean':float(x.mean()),'median':float(np.median(x)),'p90':float(np.quantile(x,.9)),'p95':float(np.quantile(x,.95)),'p99':float(np.quantile(x,.99)),'max':float(x.max())}
def load_backbone(seed): return r16.load_backbone(seed)
def base_probe(base,c): return r16.base_probe(base,c)
def base_dir(root,seed): return root/f'seed{seed}'/'BASE'

def prepare_base(seed,root):
 src=OLD/f'outputs/seed{seed}/BASE';dst=base_dir(root,seed);ev=json.loads((OLD/f'evidence/ROUND16_BASE_REFERENCE_AUDIT_SEED{seed}.json').read_text())
 if dst.exists():shutil.rmtree(dst)
 shutil.copytree(src,dst);p=torch.load(dst/'BASE_REFERENCE.pt',map_location='cpu',weights_only=False);b=fr.BaseModules();b.load_state_dict(p['state'],strict=True);h=fr.state_sha256(b)
 if h!=ev['state_dict_sha256']:raise RuntimeError('base hash mismatch')
 out={'status':'PASS','seed':seed,'reused_round16_base':True,'state_dict_sha256':h,'expected_sha256':ev['state_dict_sha256'],'selected_epoch':5,'selection_used_validation':False,'TEST_ACCESSED':False};(dst/'ROUND16R_REUSE_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');return out

def load_base(seed,root,device):
 p=torch.load(base_dir(root,seed)/'BASE_REFERENCE.pt',map_location='cpu',weights_only=False);b=fr.BaseModules().to(device);b.load_state_dict(p['state'],strict=True)
 for x in b.parameters():x.requires_grad=False
 b.eval();return b,p

class CleanTrain:
 def __init__(self,path,device):
  z=np.load(path);self.device=device;self.users=z['users'].astype(np.int64);self.items=z['ranked_items'].astype(np.int64);self.ctx=z['ranked_context'].astype(np.float32);self.score=z['ranked_full_scores'].astype(np.float32);self.target=z['target_item'].astype(np.int64);self.tcol=z['target_col'].astype(np.int64);self.hcols=z['hard_negative_cols'].astype(np.int64);self.hcount=z['hard_negative_count'].astype(np.int64)
  if np.any(self.hcount<=0):raise RuntimeError('clean pair has no negative')
 def batches(self,seed,epoch,max_batches=None):
  rng=np.random.default_rng(seed*10000+epoch);order=rng.permutation(len(self.users))
  for bi,s in enumerate(range(0,len(order),PAIR_BATCH)):
   if max_batches is not None and bi>=max_batches:return
   rr=order[s:s+PAIR_BATCH];nc=np.asarray([self.hcols[r,rng.integers(0,self.hcount[r])] for r in rr],np.int64);pc=self.tcol[rr]
   vals=(self.users[rr],self.target[rr],self.items[rr,nc],np.concatenate([self.score[rr,pc],self.score[rr,nc]]),self.ctx[rr,pc],self.ctx[rr,nc],nc+1)
   yield tuple(torch.as_tensor(x,device=self.device) for x in vals)
 def first_batch(self,n=32):
  rr=np.arange(min(n,len(self.users)));nc=np.asarray([self.hcols[r,0] for r in rr]);pc=self.tcol[rr]
  vals=(self.users[rr],self.target[rr],self.items[rr,nc],np.concatenate([self.score[rr,pc],self.score[rr,nc]]),self.ctx[rr,pc],self.ctx[rr,nc],nc+1)
  return tuple(torch.as_tensor(x,device=self.device) for x in vals)

class Evaluator(r16.ValidationEvaluator):
 def __init__(self,seed):
  super().__init__(seed,OLD/f'assets/seed{seed}_validation.npz');z=np.load(OLD/f'assets/seed{seed}_validation.npz');u=z['users'].astype(np.int64);it=z['items'].astype(np.int32)
  if not np.array_equal(u,self.users) or not np.array_equal(it,self.items):raise RuntimeError('validation context identity mismatch')
  self.ctx=z['context'].astype(np.float32)
 @torch.no_grad()
 def evaluate(self,model,base,user,c,variant,mode='true',control_seed=None,batch_users=PAIR_BATCH):
  model.eval();base.eval();user.eval();ab=fr.cosine_alpha_bar(device=model.device);noise=fr.build_noise_tables(model.n_items,model.device);width=self.items.shape[1];delta=np.empty_like(self.full,dtype=np.float32);rn=0.;rc=0;t0=time.time()
  for r0 in range(0,len(self.users),batch_users):
   r1=min(r0+batch_users,len(self.users));uu=torch.as_tensor(self.users[r0:r1],device=model.device);ids=torch.as_tensor(self.items[r0:r1].reshape(-1),device=model.device);ur=uu.repeat_interleave(width);ctx=None if variant!='A3' else torch.as_tensor(self.ctx[r0:r1].reshape(-1,8),device=model.device)
   bb,aa,rt,rv=fr.pair_scores(model,base,user,c,ur,ids,ctx,ab,noise,mode,control_seed);delta[r0:r1]=(aa-bb).float().cpu().numpy().reshape(r1-r0,width);rn+=float(rt.norm(dim=1).sum()+rv.norm(dim=1).sum());rc+=2*len(ids)
  final=self.full+delta;rank=rank_by_score(self.items,final);met=metrics_at(rank,self.users,self.eval_sets)
  return {'metrics':met,'rank':rank,'final_scores':final,'score_delta':delta,'mean_residual_norm':rn/max(rc,1),'inference_seconds':time.time()-t0,'mode':mode,'control_seed':control_seed}

def save_user(path,user,epoch,variant,seed): torch.save({'protocol':PROTOCOL,'seed':seed,'variant':variant,'epoch':epoch,'state':{k:v.detach().cpu() for k,v in user.state_dict().items()},'TEST_ACCESSED':False},path)
def restore_user(path,user):
 p=torch.load(path,map_location='cpu',weights_only=False);user.load_state_dict(p['state'],strict=True);return p

def hard_shell_diag(model,base,user,c,ev,variant,max_users=DIAG_USERS,batch=4096):
 us=[];pi=[];ni=[];ps=[];ns=[];pc=[];nc=[];eligible=0
 for r,u in enumerate(ev.users[:min(max_users,len(ev.users))]):
  loc={int(x):j for j,x in enumerate(ev.items[r])};poscols=[loc[int(x)] for x in ev.eval_sets[int(u)] if int(x) in loc]
  if not poscols:continue
  eligible+=1;pcol=poscols[0];pitem=int(ev.items[r,pcol])
  for x in ev.c0_rank[r,5:30]:
   x=int(x)
   if x in ev.eval_sets[int(u)]:continue
   j=loc[x];us.append(int(u));pi.append(pitem);ni.append(x);ps.append(float(ev.full[r,pcol]));ns.append(float(ev.full[r,j]))
   if variant=='A3':pc.append(ev.ctx[r,pcol]);nc.append(ev.ctx[r,j])
 if not us:raise RuntimeError('no validation hard-shell pairs')
 arr={k:[] for k in ('Dtrue','Dshuf','Dminus','mbase','mtrue','mminus','lplus','lzero','lminus','angle_t','angle_v','res_t','res_v','score_delta')};ab=fr.cosine_alpha_bar(device=model.device);noise=fr.build_noise_tables(model.n_items,model.device)
 for s in range(0,len(us),batch):
  e=min(s+batch,len(us));q=e-s;uu=torch.as_tensor(us[s:e],device=model.device);p=torch.as_tensor(pi[s:e],device=model.device);n=torch.as_tensor(ni[s:e],device=model.device);ids=torch.cat([p,n]);ur=torch.cat([uu,uu]);ctx=None
  if variant=='A3':ctx=torch.as_tensor(np.concatenate([np.asarray(pc[s:e]),np.asarray(nc[s:e])]),device=model.device)
  full=torch.as_tensor(np.concatenate([ps[s:e],ns[s:e]]),device=model.device)
  with torch.no_grad():
   b0,a0,rt,rv,d=fr.pair_scores(model,base,user,c,ur,ids,ctx,ab,noise,'true',None,True);b1,a1,_,_=fr.pair_scores(model,base,user,c,ur,ids,ctx,ab,noise,'shuffled');bm,am,_,_=fr.pair_scores(model,base,user,c,ur,ids,ctx,ab,noise,'negated')
  dit=a0-b0;dis=a1-b1;dim=am-bm;dt=dit[:q]-dit[q:];ds=dis[:q]-dis[q:];dm=dim[:q]-dim[q:];mbase=full[:q]-full[q:];mt=mbase+dt;mm=mbase+dm;lp=F.softplus(-mt);lz=F.softplus(-mbase);lm=F.softplus(-mm)
  for k,v in [('Dtrue',dt),('Dshuf',ds),('Dminus',dm),('mbase',mbase),('mtrue',mt),('mminus',mm),('lplus',lp),('lzero',lz),('lminus',lm)]:arr[k].append(v.cpu().numpy())
  arr['angle_t'].append(d['text']['angle_deg'].cpu().numpy());arr['angle_v'].append(d['visual']['angle_deg'].cpu().numpy());arr['res_t'].append(rt.norm(dim=1).cpu().numpy());arr['res_v'].append(rv.norm(dim=1).cpu().numpy());arr['score_delta'].append(dit.cpu().numpy())
 z={k:np.concatenate(v) for k,v in arr.items()};D=z['Dtrue'];I=D-z['Dshuf'];lp=z['lplus'];lz=z['lzero'];lm=z['lminus'];best=np.argmin(np.stack([lp,lz,lm],1),axis=1);corr=np.abs(z['score_delta'])
 return {'sample_users':min(max_users,len(ev.users)),'eligible_users':eligible,'pair_count':len(us),'definition':'first 1024 Validation users; frozen C0 rank 6-30; all non-positive candidates',
  'Delta_true':stat(D),'Delta_identity':stat(I),'fraction_Delta_true_gt0':float(np.mean(D>0)),'fraction_Delta_true_gt_shuf':float(np.mean(I>0)),
  'L_plus_minus_zero':stat(lp-lz),'L_minus_minus_zero':stat(lm-lz),'L_plus_minus_minus':stat(lp-lm),'fraction_Lplus_lt_zero':float(np.mean(lp<lz)),'fraction_Lplus_lt_minus':float(np.mean(lp<lm)),
  'tri_action_oracle':{'PLUS_BEST_fraction':float(np.mean(best==0)),'ZERO_BEST_fraction':float(np.mean(best==1)),'MINUS_BEST_fraction':float(np.mean(best==2))},
  'angle_text':stat_ext(z['angle_t']),'angle_visual':stat_ext(z['angle_v']),'residual_text':stat_ext(z['res_t']),'residual_visual':stat_ext(z['res_v']),'score_correction_abs':stat_ext(corr),'near_zero_score_correction_fraction':float(np.mean(corr<1e-6)),'small_score_correction_fraction':float(np.mean(corr<1e-4)),'TEST_ACCESSED':False}

def train_epoch(model,base,user,opt,c,ab,tc,variant,epoch,seed,max_batches=None):
 seed_all(seed*10000+epoch);model.eval();base.eval();user.train();keys=['L_rec','L_gain','L_identity','L_zero','L_bound','total_loss','m_base','m_true','m_shuf','Delta_true_mean','Delta_true_median','fraction_Delta_true_gt0','Delta_identity_mean','fraction_Delta_true_gt_shuf','L_plus','L_pair_base','L_minus','L_plus_minus_base','L_minus_minus_base','L_plus_minus_minus','fraction_Lplus_lt_base','fraction_Lplus_lt_minus','near_zero_action_fraction','small_action_fraction','correction_abs_mean','correction_abs_max','raw_residual_text_true','raw_residual_visual_true','cap_fraction_text','cap_fraction_visual','angle_text_mean','angle_text_max','angle_visual_mean','angle_visual_max'];sums={k:0. for k in keys};n=0;rmin=999;rmax=-1
 for users,pos,neg,full,pc,nc,ranks in tc.batches(seed,epoch,max_batches):
  cp=pc if variant=='A3' else None;cn=nc if variant=='A3' else None;opt.zero_grad(set_to_none=True);loss,p=fr.preference_terms(model,base,user,c,users,pos,neg,full,ab,variant,cp,cn);loss.backward()
  if any(x.grad is not None for x in base.parameters()) or any(x.grad is not None for x in model.parameters()):raise RuntimeError('frozen parameter received gradient')
  opt.step();n+=1
  for k in keys:sums[k]+=float(p[k].detach())
  rmin=min(rmin,int(ranks.min()));rmax=max(rmax,int(ranks.max()))
 out={k:v/max(n,1) for k,v in sums.items()};out.update({'batches':n,'hard_rank_min':rmin,'hard_rank_max':rmax,'preference_t_min':3,'preference_t_max':3,'base_hash':fr.state_sha256(base)});return out

def train_pair_diag(model,base,user,c,tc,variant,seed,max_batches=16):
 seed_all(20261700+seed);ab=fr.cosine_alpha_bar(device=model.device);D=[];I=[];Dm=[];Mb=[];corr=[]
 with torch.no_grad():
  for users,pos,neg,full,pc,nc,_ in tc.batches(seed,97,max_batches):
   cp=pc if variant=='A3' else None;cn=nc if variant=='A3' else None;_,p=fr.preference_terms(model,base,user,c,users,pos,neg,full,ab,variant,cp,cn,True);D.append(p['Dtrue_vec'].cpu().numpy());I.append((p['Dtrue_vec']-p['Dshuf_vec']).cpu().numpy());Dm.append(p['Dminus_vec'].cpu().numpy());Mb.append(p['mbase_vec'].cpu().numpy());corr.append(np.abs(p['dtrue_item'].cpu().numpy()))
 D=np.concatenate(D);I=np.concatenate(I);Dm=np.concatenate(Dm);Mb=np.concatenate(Mb);corr=np.concatenate(corr);lp=np.logaddexp(0,-(Mb+D));lz=np.logaddexp(0,-Mb);lm=np.logaddexp(0,-(Mb+Dm));best=np.argmin(np.stack([lp,lz,lm],1),axis=1)
 return {'pair_count':int(len(D)),'Delta_true':stat(D),'Delta_identity':stat(I),'fraction_Delta_true_gt0':float(np.mean(D>0)),'fraction_Delta_true_gt_shuf':float(np.mean(I>0)),'L_plus_minus_zero':stat(lp-lz),'fraction_Lplus_lt_zero':float(np.mean(lp<lz)),'fraction_Lplus_lt_minus':float(np.mean(lp<lm)),'tri_action_oracle':{'PLUS_BEST_fraction':float(np.mean(best==0)),'ZERO_BEST_fraction':float(np.mean(best==1)),'MINUS_BEST_fraction':float(np.mean(best==2))},'score_correction_abs':stat_ext(corr),'near_zero_score_correction_fraction':float(np.mean(corr<1e-6)),'small_score_correction_fraction':float(np.mean(corr<1e-4)),'TEST_ACCESSED':False}

def run_variant(seed,variant,root,max_batches=None):
 tp=RDIR/f'assets/seed{seed}_train_clean.npz';out=root/f'seed{seed}'/variant;shutil.rmtree(out,ignore_errors=True);out.mkdir(parents=True,exist_ok=True)
 model,_,cfg,_,c,ab=load_backbone(seed);base,_=load_base(seed,root,model.device);h0=fr.state_sha256(base);p0=base_probe(base,c);user=fr.clone_user_from_base(base,8 if variant=='A3' else 0).to(model.device);opt=torch.optim.Adam(user.parameters(),lr=float(cfg['learning_rate'])*LR_SCALE,weight_decay=float(cfg['weight_decay'] or 0.0));tc=CleanTrain(tp,model.device);ev=Evaluator(seed);logs=[]
 for ep in range(1,EPOCHS+1):
  tr=train_epoch(model,base,user,opt,c,ab,tc,variant,ep,seed,max_batches);val=ev.evaluate(model,base,user,c,variant,'true');mech=hard_shell_diag(model,base,user,c,ev,variant);sane=mech['Delta_true']['mean']>0 and mech['Delta_identity']['mean']>0;save_user(out/f'checkpoint_ep{ep}.pt',user,ep,variant,seed)
  rec={'epoch':ep,'train':tr,'validation_metrics':val['metrics'],'U_vs_C0':utility(val['metrics'],ev.c0_metrics),'mechanism':mech,'mechanism_sane':bool(sane),'base_hash':fr.state_sha256(base),'base_probe_max_abs_diff':float(np.max(np.abs(base_probe(base,c)-p0)))};logs.append(rec)
  print(json.dumps({'seed':seed,'variant':variant,'epoch':ep,'U':rec['U_vs_C0'],'R20':val['metrics']['R20'],'D':mech['Delta_true']['mean'],'Dfrac':mech['fraction_Delta_true_gt0'],'I':mech['Delta_identity']['mean'],'Ifrac':mech['fraction_Delta_true_gt_shuf'],'dirfrac':mech['fraction_Lplus_lt_zero']},sort_keys=True),flush=True)
 sane_rows=[x for x in logs if x['mechanism_sane']];ineligible=not bool(sane_rows);chosen=max(sane_rows if sane_rows else logs,key=lambda x:x['validation_metrics']['R20']);bep=chosen['epoch'];shutil.copy2(out/f'checkpoint_ep{bep}.pt',out/'best_checkpoint.pt');restore_user(out/'best_checkpoint.pt',user)
 true=ev.evaluate(model,base,user,c,variant,'true');shuf=ev.evaluate(model,base,user,c,variant,'shuffled');zero=ev.evaluate(model,base,user,c,variant,'zero');neg=ev.evaluate(model,base,user,c,variant,'negated');mech=hard_shell_diag(model,base,user,c,ev,variant);td=train_pair_diag(model,base,user,c,tc,variant,seed)
 bh=fr.state_sha256(base);pd=float(np.max(np.abs(base_probe(base,c)-p0)));maxang=max(mech['angle_text']['max'],mech['angle_visual']['max']);construct=json.loads((RDIR/f'evidence/ROUND16R_TRAIN_CONSTRUCTION_SEED{seed}.json').read_text())
 result={'status':'PASS','protocol':PROTOCOL,'seed':seed,'variant':variant,'epochs':5,'best_epoch':bep,'selection_policy':'mechanism_sane mean Delta_true>0 AND mean(Delta_true-Delta_shuf)>0; then max Validation R20; else max R20 report-only','PASS_INELIGIBLE':ineligible,
  'C0_metrics':ev.c0_metrics,'best_metrics':true['metrics'],'best_vs_C0':delta_pack(true['metrics'],ev.c0_metrics),'identity_ablation':{'label':'CF_ID_SHUFFLED_CTX_FIXED' if variant=='A3' else 'CF_ID_SHUFFLED','true':delta_pack(true['metrics'],ev.c0_metrics),'shuffled':delta_pack(shuf['metrics'],ev.c0_metrics),'zero':delta_pack(zero['metrics'],ev.c0_metrics)},'negated':delta_pack(neg['metrics'],ev.c0_metrics),
  'hard_shell_mechanism':mech,'train_pair_diagnostic':td,'epoch_logs':logs,'reference':{'initial_hash':h0,'final_hash':bh,'hash_unchanged':h0==bh,'probe_max_abs_diff':pd,'zero_ranking_exact':bool(np.array_equal(zero['rank'],ev.c0_rank))},'bound':{'max_angle':maxang},
  'clean_train':{'users':len(tc.users),'m_base_abs_max':construct['m_base_abs_max'],'same_row_only':True,'off_row_positive_count':0},'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(out/'best_validation_rankings.npz',users=ev.users,ranked_items=true['rank'],candidate_items=ev.items,final_scores=true['final_scores']);del model,base,user,opt;torch.cuda.empty_cache();gc.collect();return result

def random_controls(seed,variant,root):
 r=json.loads((root/f'seed{seed}/{variant}/result.json').read_text());model,_,_,_,c,_=load_backbone(seed);base,_=load_base(seed,root,model.device);user=fr.clone_user_from_base(base,8 if variant=='A3' else 0).to(model.device);restore_user(root/f'seed{seed}/{variant}/best_checkpoint.pt',user);ev=Evaluator(seed);vals=[]
 for rs in RANDOM_SEEDS:
  x=ev.evaluate(model,base,user,c,variant,'random',rs);vals.append({'seed':rs,'metrics':x['metrics'],'vs_C0':delta_pack(x['metrics'],ev.c0_metrics)})
 tu=r['best_vs_C0']['U'];us=np.asarray([x['vs_C0']['U'] for x in vals]);out={'seed':seed,'variant':variant,'random_seeds':list(RANDOM_SEEDS),'controls':vals,'mean_U_random':float(us.mean()),'std_U_random':float(us.std()),'min_U_random':float(us.min()),'max_U_random':float(us.max()),'U_true':tu,'U_negated':r['negated']['U'],'true_percentile_vs_random':float(100*np.mean(us<tu)),'PASS':bool(tu>us.mean() and tu>r['negated']['U']),'TEST_ACCESSED':False};del model,base,user;torch.cuda.empty_cache();return out

def _grad_norm(mod): return float(sum(float(p.grad.detach().float().pow(2).sum()) for p in mod.parameters() if p.grad is not None)**.5)
def gate_c(seed):
 E=RDIR/'evidence';si=json.loads((E/f'ROUND16R_SELF_INCLUSION_SEED{seed}.json').read_text());sr=json.loads((E/f'ROUND16R_SAME_ROW_SEED{seed}.json').read_text());co=json.loads((E/f'ROUND16R_TRAIN_CONSTRUCTION_SEED{seed}.json').read_text());sc=json.loads((E/f'ROUND16R_CONTEXT_SCALE_SEED{seed}.json').read_text())
 checks={'self_inclusion_zero':si['violations']==0,'off_row_positive_zero':sr['off_row_positive_count']==0,'extrapolation_zero':sr['target_context_extrapolation_count']==0,'nan_zero':co['nan_count']==0,'inf_zero':co['inf_count']==0,'context_abs_lt_100':co['all_score_like_abs_lt_100'],'m_base_abs_lt_100':co['m_base_abs_lt_100'],'scale_consistent':sc['status']=='PASS'};return {'seed':seed,'checks':checks,'PASS':all(checks.values())}

def smoke(root,evid):
 shutil.rmtree(root,ignore_errors=True);root.mkdir(parents=True,exist_ok=True);out={'protocol':PROTOCOL,'Gate_C':{},'seeds':{},'TEST_ACCESSED':False}
 for seed in PREFLIGHT:
  gcate=gate_c(seed);out['Gate_C'][str(seed)]=gcate
  if not gcate['PASS']:raise RuntimeError(f'Gate C FAIL seed{seed}')
  b=prepare_base(seed,root);model,_,cfg,_,c,ab=load_backbone(seed);base,_=load_base(seed,root,model.device);h0=fr.state_sha256(base);p0=base_probe(base,c);tc=CleanTrain(RDIR/f'assets/seed{seed}_train_clean.npz',model.device);ev=Evaluator(seed);uu,pp,nn,full,pc,nc,ranks=tc.first_batch();sr={'base':b,'variants':{},'validation_evaluator_users':int(len(ev.users)),'validation_C0_metrics':ev.c0_metrics}
  if float(full.abs().max())>=100:raise RuntimeError('smoke mbase/full scale impossible')
  for v in VARIANTS:
   user=fr.clone_user_from_base(base,8 if v=='A3' else 0).to(model.device);cp=pc if v=='A3' else None;cn=nc if v=='A3' else None;user.zero_grad(set_to_none=True);loss,parts=fr.preference_terms(model,base,user,c,uu,pp,nn,full,ab,v,cp,cn);loss.backward();gn=_grad_norm(user);frozen=all(p.grad is None for p in base.parameters()) and all(p.grad is None for p in model.parameters());ma=max(float(parts['angle_text_max']),float(parts['angle_visual_max']))
   if not torch.isfinite(loss) or gn<=0 or not frozen or ma>5.01:raise RuntimeError(f'{v} smoke failed')
   opt=torch.optim.Adam(user.parameters(),lr=float(cfg['learning_rate'])*LR_SCALE);opt.step();user.zero_grad(set_to_none=True);_,q=fr.preference_terms(model,base,user,c,uu,pp,nn,full,ab,v,cp,cn);q['L_zero'].backward();zg=_grad_norm(user)
   sr['variants'][v]={'loss':float(loss.detach()),'user_grad_norm':gn,'frozen_grad_none':frozen,'max_angle':ma,'m_base_mean':float(parts['m_base']),'Delta_true_mean':float(parts['Delta_true_mean']),'L_zero_after_update_grad_norm':zg,'direct_delta_margin':True,'context_dim':8 if v=='A3' else 0};del user,opt
  sr['base_hash_unchanged']=fr.state_sha256(base)==h0;sr['base_probe_max_abs_diff']=float(np.max(np.abs(base_probe(base,c)-p0)));sr['PASS']=sr['base_hash_unchanged'] and sr['base_probe_max_abs_diff']<=1e-7 and all(x['frozen_grad_none'] for x in sr['variants'].values());out['seeds'][str(seed)]=sr;del model,base;torch.cuda.empty_cache();gc.collect()
 out['PASS']=all(x['PASS'] for x in out['seeds'].values()) and all(x['PASS'] for x in out['Gate_C'].values());evid.mkdir(parents=True,exist_ok=True);(evid/'ROUND16R_PREFLIGHT_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'PASS':out['PASS'],'Gate_C':out['Gate_C']},sort_keys=True));return out

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['smoke','base','variant','random'],required=True);ap.add_argument('--seed',type=int);ap.add_argument('--variant',choices=VARIANTS);ap.add_argument('--root',default=str(RDIR/'outputs'));ap.add_argument('--evidence',default=str(RDIR/'evidence'));a=ap.parse_args();root=Path(a.root);evid=Path(a.evidence)
 if a.mode=='smoke':smoke(RDIR/'outputs_smoke',evid)
 elif a.mode=='base':print(json.dumps(prepare_base(a.seed,root),sort_keys=True))
 elif a.mode=='variant':print(json.dumps({'seed':a.seed,'variant':a.variant,'status':run_variant(a.seed,a.variant,root)['status']},sort_keys=True))
 else:print(json.dumps(random_controls(a.seed,a.variant,root),sort_keys=True))
if __name__=='__main__':main()
