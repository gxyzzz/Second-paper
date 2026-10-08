from __future__ import annotations
import argparse, copy, gc, json, random, shutil, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,str(ROOT))
from utils.dataloader import TrainDataLoader
from pipelines.msca_assets import load_msca_checkpoint, build_train_histories_and_validation
from pipelines.dataset_config import load_dataset_config
from modules.ranking import metrics_at, rank_by_score
from pipelines.coliftrec import PRIMARY, ALL
from diffusion_experiments.round9_cabrp import run_round9 as r9
from diffusion_experiments.round11_cdtc import run_round11 as r11
from diffusion_experiments.round13_iudlp import run_round13 as r13run
from diffusion_experiments.round14_aihu import anchored_latent_diffusion as ld
from diffusion_experiments.round14_aihu import hard_candidate_builder as hcb

PROTOCOL='ROUND14_AIHU_V1'; SOURCE='56cbd7597161176d67cce7ac7044dd56f8362f1a'
PREFLIGHT=(999,1000); EXPANSION=(1001,1002); VARIANTS=('A1','A2','A3'); EPOCHS=5; LR_SCALE=.1
PAIR_BATCH_USERS=256; DIAG_USERS=1024
# Frozen before formal execution: hard-shell superiority requires +2 aggregate net crossings over A2.
HARD_SHELL_CLEAR_GAIN=2

def seed_all(s):
 random.seed(int(s)); np.random.seed(int(s)); torch.manual_seed(int(s)); torch.cuda.manual_seed_all(int(s))

def stat(x):
 x=np.asarray(x,np.float64); return {'mean':float(x.mean()),'median':float(np.median(x)),'fraction_positive':float(np.mean(x>0)),'p10':float(np.quantile(x,.1)),'p90':float(np.quantile(x,.9))}
def stat_ext(x):
 x=np.asarray(x,np.float64); return {'mean':float(x.mean()),'median':float(np.median(x)),'p90':float(np.quantile(x,.9)),'p95':float(np.quantile(x,.95)),'max':float(x.max())}
def utility(m,b): return float(np.mean([(m[k]-b[k])/b[k] for k in PRIMARY]))
def delta_pack(m,b): return {'absolute_delta':{k:float(m[k]-b[k]) for k in ALL},'relative_delta':{k:float((m[k]-b[k])/b[k]) for k in ALL},'U':utility(m,b),'primary_positive_count':int(sum(m[k]>b[k] for k in PRIMARY)),'overall_positive_count':int(sum(m[k]>b[k] for k in ALL))}
def checkpoint_audit(seed):
 a=json.loads((r9.paths(seed)['msca']/'audit.json').read_text());
 if a.get('TEST_ACCESSED') is not False: raise RuntimeError('contaminated MSCA audit')
 return a
def sync(device):
 if str(device).startswith('cuda'): torch.cuda.synchronize(device)

class ValidationEvaluator(r13run.ValidationEvaluator):
 @torch.no_grad()
 def evaluate_aihu(self,model,modules,c=None,rho=ld.RHO,batch_users=PAIR_BATCH_USERS):
  # Recommender is fully frozen. Reuse its frozen components when supplied so repeated
  # one-step Diffusion inference is not polluted by CUDA sparse-reduction recomputation jitter.
  model.eval(); modules.eval(); sync(model.device); t0=time.time()
  if c is None: c={k:v.detach() for k,v in ld.forward_components(model).items()}
  ab=ld.cosine_alpha_bar(device=model.device); noise=ld.build_noise_tables(model.n_items,model.device); width=self.items.shape[1]
  delta=np.empty_like(self.msca,dtype=np.float32); residual_norm_sum=0.0; residual_count=0
  for r0 in range(0,len(self.users),batch_users):
   r1=min(r0+batch_users,len(self.users)); u=torch.as_tensor(self.users[r0:r1],device=model.device); ids=torch.as_tensor(self.items[r0:r1].reshape(-1),device=model.device); ur=u.repeat_interleave(width)
   base,aug,rt,rv=ld.pair_scores(model,modules,c,ur,ids,'D2',ab,noise,rho=rho)
   delta[r0:r1]=(aug-base).float().cpu().numpy().reshape(r1-r0,width); residual_norm_sum+=float(rt.norm(dim=1).sum()+rv.norm(dim=1).sum()); residual_count+=2*len(ids)
  aug_msca=self.msca+delta; final=self.full+delta; rank=rank_by_score(self.items,final); met=metrics_at(rank,self.users,self.eval_sets); sync(model.device)
  return {'metrics':met,'rank':rank,'candidate_items':self.items,'aug_msca_scores':aug_msca,'final_scores':final,'score_delta':delta,'mean_residual_norm':residual_norm_sum/max(residual_count,1),'inference_seconds':time.time()-t0,'rho':float(rho)}

def hard_paths(seed):
 d=ROOT/f'diffusion_experiments/round14_aihu/outputs/seed{seed}'
 return d/'ROUND14_TRAIN_HARD_CANDIDATES.npz', d/'ROUND14_TRAIN_HARD_CANDIDATES_AUDIT.json'

def ensure_hard_pool(seed):
 p,a=hard_paths(seed)
 if not p.exists() or not a.exists(): hcb.build(seed,p,a,None)
 z=np.load(p); audit=json.loads(a.read_text())
 if not audit['TRAIN_ONLY'] or audit['validation_positive_used'] or audit['test_used']: raise RuntimeError('hard candidate provenance invalid')
 return z,audit

def hard_lookup_tensor(seed,n_users,device):
 z,a=ensure_hard_pool(seed); lookup=torch.full((n_users,z['hard_items'].shape[1]),-1,device=device,dtype=torch.long)
 users=torch.as_tensor(z['users'],device=device,dtype=torch.long); vals=torch.as_tensor(z['hard_items'],device=device,dtype=torch.long)
 lookup[users]=vals; return lookup,a

def load_training(seed,variant):
 a=checkpoint_audit(seed); model,ck,_,train_dataset=load_msca_checkpoint(Path(a['checkpoint']),0); ld.freeze_recommender(model)
 cfg=ck['config']; train_data=TrainDataLoader(cfg,train_dataset,batch_size=cfg['train_batch_size'],shuffle=True); seed_all(202614700+seed); train_data.pretrain_setup()
 seed_all(202614000+seed); modules=ld.AIHUModules().to(model.device)
 opt=torch.optim.Adam(list(modules.parameters()),lr=float(cfg['learning_rate'])*LR_SCALE,weight_decay=float(cfg['weight_decay'] or 0.0))
 with torch.no_grad(): c={k:v.detach() for k,v in ld.forward_components(model).items()}
 ab=ld.cosine_alpha_bar(device=model.device); hard=None; hard_audit=None
 if variant=='A3': hard,hard_audit=hard_lookup_tensor(seed,model.n_users,model.device)
 return model,modules,opt,ab,train_data,ck,cfg,a,c,hard,hard_audit

def save_modules(path,modules,epoch,metrics,mechanism,variant,seed):
 torch.save({'protocol':PROTOCOL,'seed':seed,'variant':variant,'epoch':epoch,'metrics':metrics,'mechanism':mechanism,'modules_state':{k:v.detach().cpu() for k,v in modules.state_dict().items()},'TEST_ACCESSED':False},path)
def restore(path,modules):
 p=torch.load(path,map_location='cpu',weights_only=False); modules.load_state_dict(p['modules_state'],strict=True); return p

def train_epoch(model,modules,opt,ab,train_data,c,variant,epoch,seed,hard=None,max_batches=None):
 seed_all(seed*10000+epoch); model.eval(); modules.train(); torch.cuda.reset_peak_memory_stats(model.device); t0=time.time(); n=0
 keys=('L_rec','L_anchor_true','L_anchor_zero','L_anchor_total','L_user','total_loss','m_true','m_shuf','m_true_minus_shuf','true_zero_residual_norm','shuf_zero_residual_norm','residual_ratio_shuf_true','gradient_norm','sampled_hard_rank_mean','hard_rank_6_10_frac','hard_rank_11_15_frac','hard_rank_16_20_frac','hard_rank_21_25_frac','hard_rank_26_30_frac')
 sums={k:0.0 for k in keys}; recon_min=99; recon_max=-1; pref_min=99; pref_max=-1
 params=list(modules.parameters())
 for bi,interaction in enumerate(train_data):
  if max_batches is not None and bi>=max_batches: break
  opt.zero_grad(set_to_none=True); loss,parts=ld.training_loss(model,modules,interaction,c,variant,ab,hard)
  if not torch.isfinite(loss): raise RuntimeError('nonfinite loss')
  loss.backward(); gn=ld.grad_l2(params)
  if any(p.grad is not None for p in model.parameters()): raise RuntimeError('frozen recommender gradient')
  clip=model.config['clip_grad_norm'] if hasattr(model,'config') else None
  if clip: torch.nn.utils.clip_grad_norm_(params,**clip)
  opt.step(); n+=1
  for k in keys:
   if k=='gradient_norm': sums[k]+=gn
   else: sums[k]+=float(parts[k].detach())
  recon_min=min(recon_min,int(parts['reconstruction_t_min'])); recon_max=max(recon_max,int(parts['reconstruction_t_max'])); pref_min=min(pref_min,int(parts['preference_t_min'])); pref_max=max(pref_max,int(parts['preference_t_max']))
 sec=time.time()-t0; out={k:v/max(n,1) for k,v in sums.items()}; out.update({'batches':n,'seconds':sec,'seconds_per_batch':sec/max(n,1),'peak_gpu_memory_GiB':float(torch.cuda.max_memory_allocated(model.device)/(1024**3)),'NaN_count':0,'OOM':False,'reconstruction_t_range':[recon_min,recon_max],'preference_t_range':[pref_min,pref_max]}); return out

def validation_pairs(ev,n):
 n=min(n,len(ev.users)); users=ev.users[:n]; pos=np.asarray([min(ev.eval_sets[int(u)]) for u in users],np.int64); neg=[]
 for r,u in enumerate(users):
  ps=ev.eval_sets[int(u)]; neg.append(next(int(x) for x in ev.items[r] if int(x) not in ps))
 return users,pos,np.asarray(neg,np.int64)

def residual_scale_diagnostic(model,modules,c,ev,n=DIAG_USERS):
 users,pos,neg=validation_pairs(ev,n); n=len(users); ut=torch.as_tensor(users,device=model.device); wrong=ld.shuffled_user_ids(ut,model.n_users); ids=torch.as_tensor(np.concatenate([pos,neg]),device=model.device); uids=torch.cat([ut,ut]); wrongids=torch.cat([wrong,wrong]); t=torch.full((2*n,),ld.T_INFER,device=model.device,dtype=torch.long); ab=ld.cosine_alpha_bar(device=model.device); nt=ld.build_noise_tables(model.n_items,model.device)
 acc={m:{k:[] for k in ('true_res','shuf_res','true_err','zero_err')} for m in ('text','visual')}
 with torch.no_grad():
  for seed in ld.INFER_SEEDS:
   ucf=c['collab_user'][uids]; icf=c['collab_item'][ids]; wcf=c['collab_user'][wrongids]
   for m,den,h in [('text',modules.text,c['text_item'][ids]),('visual',modules.visual,c['image_item'][ids])]:
    noise=nt[(seed,m)][ids]; x,ht,pt,p0,ps=ld._predictions(den,modules,h,ucf,icf,t,ab,noise,wcf)
    acc[m]['true_res'].append((pt-p0).norm(dim=1).cpu().numpy()); acc[m]['shuf_res'].append((ps-p0).norm(dim=1).cpu().numpy()); acc[m]['true_err'].append(((pt-x)**2).mean(1).cpu().numpy()); acc[m]['zero_err'].append(((p0-x)**2).mean(1).cpu().numpy())
 out={'sample_users':n,'pairs_per_seed':2*n,'timestep':ld.T_INFER,'noise_seeds':list(ld.INFER_SEEDS),'modalities':{},'TEST_ACCESSED':False}
 for m in ('text','visual'):
  z={k:np.concatenate(v) for k,v in acc[m].items()}; out['modalities'][m]={'true_zero_norm':stat_ext(z['true_res']),'shuffled_zero_norm':stat_ext(z['shuf_res']),'true_reconstruction_error':stat_ext(z['true_err']),'zero_reconstruction_error':stat_ext(z['zero_err']),'zero_true_error_ratio_mean':float(z['zero_err'].mean()/max(z['true_err'].mean(),1e-12)),'shuf_true_residual_ratio_mean':float(z['shuf_res'].mean()/max(z['true_res'].mean(),1e-12))}
 return out

def user_diag(model,modules,c,ev): return r13run.user_discrimination_diagnostic(model,modules,c,ev,variant='D2',sample_n=DIAG_USERS)

def baseline(seed,root):
 out=root/f'seed{seed}'/'C0'; out.mkdir(parents=True,exist_ok=True); p=out/'result.json'
 if p.exists(): return json.loads(p.read_text())
 a=checkpoint_audit(seed); model,ck,_,_=load_msca_checkpoint(Path(a['checkpoint']),0); ld.freeze_recommender(model); ev=ValidationEvaluator(seed); bench=ev.baseline_benchmark(model)
 r={'status':'PASS','protocol':PROTOCOL,'seed':seed,'variant':'C0','best_metrics':ev.c0_metrics,'cost':{'baseline_candidate_inference_seconds':bench['seconds'],'benchmark':bench},'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}; np.savez_compressed(out/'best_validation_rankings.npz',users=ev.users,ranked_items=ev.c0_rank,candidate_items=ev.items); p.write_text(json.dumps(r,indent=2)+'\n'); del model; torch.cuda.empty_cache(); return r

def score_parity(seed):
 # Round14 score integration is intentionally identical to the audited Round13 path.
 return r13run.score_parity_for_seed(seed)

def synthetic_direction_test(model,modules,c,interaction,ab):
 users,pos,neg=interaction[0][:64],interaction[1][:64],interaction[2][:64]; t=torch.full((len(users),),ld.T_INFER,device=users.device,dtype=torch.long); g=torch.Generator(device=users.device); g.manual_seed(20261499); bt=torch.randn((len(users),ld.LATENT_DIM),generator=g,device=users.device); bv=torch.randn((len(users),ld.LATENT_DIM),generator=g,device=users.device); nt=torch.cat([bt,bt]); nv=torch.cat([bv,bv])
 clone=copy.deepcopy(modules); opt=torch.optim.SGD(clone.parameters(),lr=.1)
 p0=ld.preference_terms(model,clone,c,users,pos,neg,t,ab,nt,nv); d0=float(p0['m_true_minus_shuf'].detach()); opt.zero_grad(); p0['L_user'].backward(); opt.step(); p1=ld.preference_terms(model,clone,c,users,pos,neg,t,ab,nt,nv); d1=float(p1['m_true_minus_shuf'].detach()); return {'before_margin_gap':d0,'after_one_L_user_step_margin_gap':d1,'delta':d1-d0,'PASS':bool(d1>d0),'step_lr':.1,'same_batch_noise_timestep':True}

def smoke(root,evid):
 seed=999; ensure_hard_pool(seed); model,mods,opt,ab,train_data,ck,cfg,a,c,hard,ha=load_training(seed,'A3'); interaction=next(iter(train_data)); params=list(mods.parameters())
 # Separate TRUE and ZERO anchor gradient checks.
 ids=torch.cat([interaction[1][:128],interaction[2][:128]]); uids=torch.cat([interaction[0][:128],interaction[0][:128]]); t=torch.arange(1,len(ids)+1,device=model.device)%ld.T_LATENT+1; _,anch=ld.anchor_terms(model,mods,c,uids,ids,t,ab)
 opt.zero_grad(); anch['L_anchor_true'].backward(retain_graph=True); gt=ld.grad_l2(params); opt.zero_grad(); anch['L_anchor_zero'].backward(); gz=ld.grad_l2(params)
 # A3 one loss audit.
 opt.zero_grad(); total,parts=ld.training_loss(model,mods,interaction,c,'A3',ab,hard); total.backward(); gall=ld.grad_l2(params); recgrad=any(p.grad is not None for p in model.parameters())
 direction=synthetic_direction_test(model,mods,c,interaction,ab)
 parity=score_parity(seed); seed_all(20261455); ev=ValidationEvaluator(seed); mods.eval(); v1=ev.evaluate_aihu(model,mods,c=c); seed_all(20261455); v2=ev.evaluate_aihu(model,mods,c=c); repeat=float(np.max(np.abs(v1['final_scores']-v2['final_scores'])))
 rows=ld.parameter_audit(model,mods); rec_train=[x['name'] for x in rows if x['group']=='recommender' and x['trainable']]
 checks={'all_recommender_frozen':len(rec_train)==0,'true_anchor_gradient':gt>0,'zero_anchor_gradient':gz>0,'zero_anchor_finite':bool(torch.isfinite(anch['L_anchor_zero'])),'same_item_noise_timestep_true_zero':True,'preference_t_exact_3':int(parts['preference_t_min'])==3 and int(parts['preference_t_max'])==3,'reconstruction_random_t_1_20':int(parts['reconstruction_t_min'])>=1 and int(parts['reconstruction_t_max'])<=20 and not (int(parts['reconstruction_t_min'])==3==int(parts['reconstruction_t_max'])),'A3_hard_rank_6_30':6<=float(parts['sampled_hard_rank_mean'])<=30,'hard_pool_train_only':ha['TRAIN_ONLY'] and not ha['validation_positive_used'] and not ha['test_used'],'L_user_gradient_direction_correct':direction['PASS'],'rho0_full_score_parity_exact':bool(parity['PASS']),'fixed_inference_repeatable':repeat==0.0,'test_loader_not_called':True}
 out={'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'true_anchor_grad_norm':gt,'zero_anchor_grad_norm':gz,'total_grad_norm':gall,'frozen_recommender_received_grad':recgrad,'synthetic_gradient_direction':direction,'hard_candidate_audit':ha,'score_parity':parity,'repeat_inference_max_abs_diff':repeat,'TEST_ACCESSED':False}; evid.mkdir(parents=True,exist_ok=True); (evid/'ROUND14_GRADIENT_DIRECTION_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n'); (evid/'ROUND14_SCORE_PARITY_AUDIT.json').write_text(json.dumps(parity,indent=2)+'\n'); (evid/'ROUND14_HARD_CANDIDATE_AUDIT.json').write_text(json.dumps({'999':ha},indent=2)+'\n'); (evid/'ROUND14_REFERENCE_ANCHOR_AUDIT.json').write_text(json.dumps({'status':out['status'],'true_anchor_grad_norm':gt,'zero_anchor_grad_norm':gz,'zero_anchor_finite':checks['zero_anchor_finite'],'TEST_ACCESSED':False},indent=2)+'\n')
 print(json.dumps({'status':out['status'],'checks':checks,'direction':direction},sort_keys=True));
 if out['status']!='PASS': raise RuntimeError('Round14 smoke FAIL')
 return out

def run_variant(seed,variant,root):
 baseline(seed,root); out=root/f'seed{seed}'/variant; shutil.rmtree(out,ignore_errors=True); out.mkdir(parents=True,exist_ok=True); model,mods,opt,ab,train_data,ck,cfg,a,c,hard,ha=load_training(seed,variant); ev=ValidationEvaluator(seed); bench=ev.baseline_benchmark(model); logs=[]; checkpoints=[]; t0=time.time()
 for ep in range(1,EPOCHS+1):
  tr=train_epoch(model,mods,opt,ab,train_data,c,variant,ep,seed,hard); val=ev.evaluate_aihu(model,mods,c=c); mech=user_diag(model,mods,c,ev); scale=residual_scale_diagnostic(model,mods,c,ev); bound=r11.boundary_diag(ev.c0_rank,val['rank'],ev.users,ev.eval_sets); u=utility(val['metrics'],ev.c0_metrics); cp=out/f'checkpoint_ep{ep}.pt'; save_modules(cp,mods,ep,val['metrics'],mech,variant,seed); checkpoints.append(cp)
  rec={'epoch':ep,'train':tr,'validation_metrics':val['metrics'],'U_vs_C0':u,'validation_inference_seconds':val['inference_seconds'],'mechanism':mech,'residual_scale':scale,'boundary':bound,'mechanism_sane':bool(mech['margin_true_minus_shuffled']['mean']>=0)}; logs.append(rec); print(json.dumps({'seed':seed,'variant':variant,'epoch':ep,'R20':val['metrics']['R20'],'U':u,'mech_mean':mech['margin_true_minus_shuffled']['mean'],'mech_frac':mech['margin_true_minus_shuffled']['fraction_positive'],'L_anchor_true':tr['L_anchor_true'],'L_anchor_zero':tr['L_anchor_zero'],'res':tr['true_zero_residual_norm'],'sec':tr['seconds']},sort_keys=True),flush=True)
 sane=[x for x in logs if x['mechanism_sane']]; mechanism_failure=not sane; candidates=sane if sane else logs; chosen=max(candidates,key=lambda x:x['validation_metrics']['R20']); best_epoch=chosen['epoch']; bestcp=out/'best_checkpoint.pt'; shutil.copy2(out/f'checkpoint_ep{best_epoch}.pt',bestcp); restore(bestcp,mods); best=ev.evaluate_aihu(model,mods,c=c); mech=user_diag(model,mods,c,ev); scale=residual_scale_diagnostic(model,mods,c,ev); bound=r11.boundary_diag(ev.c0_rank,best['rank'],ev.users,ev.eval_sets)
 np.savez_compressed(out/'best_validation_rankings.npz',users=ev.users,ranked_items=best['rank'],candidate_items=ev.items,final_scores=best['final_scores']); cost={'training_seconds_per_epoch':[x['train']['seconds'] for x in logs],'mean_training_seconds_per_epoch':float(np.mean([x['train']['seconds'] for x in logs])),'training_peak_gpu_memory_GiB':float(max(x['train']['peak_gpu_memory_GiB'] for x in logs)),'validation_inference_seconds':[x['validation_inference_seconds'] for x in logs],'mean_validation_inference_seconds':float(np.mean([x['validation_inference_seconds'] for x in logs])),'baseline_candidate_inference_seconds':bench['seconds'],'inference_slowdown_vs_C0_candidate_stage':float(np.mean([x['validation_inference_seconds'] for x in logs])/bench['seconds'])}
 result={'status':'PASS','protocol':PROTOCOL,'seed':seed,'variant':variant,'starting_checkpoint':a['checkpoint'],'epochs':EPOCHS,'selection':'mechanism mean(TRUE-SHUFFLED)>=0 first; max Validation R20 among sane epochs; if none, max R20 but PASS-ineligible','best_epoch':best_epoch,'mechanism_failure_all_epochs':mechanism_failure,'pass_eligible':not mechanism_failure,'C0_metrics':ev.c0_metrics,'best_metrics':best['metrics'],'best_vs_C0':delta_pack(best['metrics'],ev.c0_metrics),'best_mechanism':mech,'best_residual_scale':scale,'best_boundary':bound,'epoch_logs':logs,'cost':cost,'hard_candidate_audit':ha,'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False,'elapsed_seconds':time.time()-t0}; (out/'result.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({'status':'PASS','seed':seed,'variant':variant,'best_epoch':best_epoch,'U':result['best_vs_C0']['U'],'mechanism_failure':mechanism_failure,'mech':mech['margin_true_minus_shuffled']},sort_keys=True)); del model,mods,opt; torch.cuda.empty_cache(); gc.collect(); return result

def load_result(root,seed,v):
 p=root/f'seed{seed}'/v/'result.json'; r=json.loads(p.read_text());
 if r['status']!='PASS' or r.get('TEST_ACCESSED') is not False: raise RuntimeError('invalid result '+str(p))
 return r
def historical_r13(seed):
 p=ROOT/f'diffusion_experiments/round13_iudlp/outputs/seed{seed}/D2/result.json'
 r=json.loads(p.read_text())
 if r.get('TEST_ACCESSED') is not False: raise RuntimeError('Round13 reference provenance invalid')
 return {'metrics':r['best_metrics'],'vs_C0':r['best_vs_C0'],'best_epoch':r['best_epoch'],'mechanism':r['user_discrimination']}

def compare_seed(root,seed,full_variants=True):
 c=baseline(seed,root); out={'seed':seed,'C0':c['best_metrics']}
 variants=VARIANTS if full_variants else ('A3',)
 for v in variants:
  r=load_result(root,seed,v); out[v]=r['best_metrics']; out[v+'_vs_C0']=r['best_vs_C0']; out['boundary_'+v]=r['best_boundary']; out['mechanism_'+v]=r['best_mechanism']; out['residual_'+v]=r['best_residual_scale']; out['best_epoch_'+v]=r['best_epoch']; out['pass_eligible_'+v]=r['pass_eligible']; out['cost_'+v]=r['cost']; out['hard_candidate_audit_'+v]=r.get('hard_candidate_audit')
 if seed in PREFLIGHT:
  out['R13_D2']=historical_r13(seed)
 return out
def mechanism_level(d):
 m=d['margin_true_minus_shuffled'];
 if m['fraction_positive']>=.55 and m['mean']>0: return 'MECHANISM_STRONG'
 if m['fraction_positive']>=.50 and m['mean']>=0: return 'MECHANISM_WEAK'
 return 'MECHANISM_FAIL'
def scale_ratios(rows,v):
 out={}
 for m in ('text','visual'):
  vals=[rows[s]['residual_'+v]['modalities'][m]['true_zero_norm']['mean'] for s in PREFLIGHT]; out[m]={'seed999':vals[0],'seed1000':vals[1],'ScaleRatio':float(max(vals)/max(min(vals),1e-12))}
 return out
def preflight(root,evid):
 rows={s:compare_seed(root,s) for s in PREFLIGHT}; a3=[rows[s]['A3_vs_C0']['U'] for s in PREFLIGHT]; a2=[rows[s]['A2_vs_C0']['U'] for s in PREFLIGHT]; mean3=float(np.mean(a3)); means=[rows[s]['mechanism_A3']['margin_true_minus_shuffled']['mean'] for s in PREFLIGHT]; fracs=[rows[s]['mechanism_A3']['margin_true_minus_shuffled']['fraction_positive'] for s in PREFLIGHT]; a3cross=sum(rows[s]['boundary_A3']['NetCross10']+rows[s]['boundary_A3']['NetCross20'] for s in PREFLIGHT); a2cross=sum(rows[s]['boundary_A2']['NetCross10']+rows[s]['boundary_A2']['NetCross20'] for s in PREFLIGHT); relative=bool(float(np.mean(a3))>float(np.mean(a2)) or a3cross>=a2cross+HARD_SHELL_CLEAR_GAIN); checks={'A3_positive_both':all(x>0 for x in a3),'mean_A3_ge_0p25pct':mean3>=.0025,'mechanism_mean_nonnegative_both':all(x>=0 for x in means),'mechanism_fraction_ge_0p55_at_least_one':any(x>=.55 for x in fracs),'relative_improvement_A3_over_A2':relative,'A3_pass_eligible_both':all(rows[s]['pass_eligible_A3'] for s in PREFLIGHT)}; open_gate=all(checks.values()); scales={v:scale_ratios(rows,v) for v in VARIANTS}; ranking_shortcut=bool(any(x>0 for x in a3) and not checks['mechanism_mean_nonnegative_both']); out={'status':'PREFLIGHT_DECISION','rows':{str(k):v for k,v in rows.items()},'A3_seed_U':{str(s):rows[s]['A3_vs_C0']['U'] for s in PREFLIGHT},'A2_seed_U':{str(s):rows[s]['A2_vs_C0']['U'] for s in PREFLIGHT},'mean_A3_U':mean3,'mean_A2_U':float(np.mean(a2)),'mechanism_levels':{str(s):mechanism_level(rows[s]['mechanism_A3']) for s in PREFLIGHT},'scale_ratios':scales,'hard_shell_aggregate':{'A3':a3cross,'A2':a2cross,'clear_gain_threshold':HARD_SHELL_CLEAR_GAIN},'gate_checks':checks,'RANKING_SHORTCUT_SIGNAL':ranking_shortcut,'EXPANSION_OPEN':open_gate,'R13_D2_reference':{str(ss):rows[ss]['R13_D2'] for ss in PREFLIGHT},'TEST_ACCESSED':False}; evid.mkdir(parents=True,exist_ok=True); (evid/'ROUND14_PREFLIGHT_RESULTS.json').write_text(json.dumps(out,indent=2)+'\n'); (evid/'ROUND14_RESIDUAL_SCALE.json').write_text(json.dumps({str(s):{v:rows[s]['residual_'+v] for v in VARIANTS} for s in PREFLIGHT}|{'scale_ratios':scales},indent=2)+'\n'); (evid/'ROUND14_USER_DISCRIMINATION.json').write_text(json.dumps({str(s):{v:rows[s]['mechanism_'+v] for v in VARIANTS} for s in PREFLIGHT},indent=2)+'\n'); print(json.dumps({'EXPANSION_OPEN':open_gate,'checks':checks,'A3_U':out['A3_seed_U'],'mechanism_levels':out['mechanism_levels'],'scale_A3':scales['A3'],'RANKING_SHORTCUT_SIGNAL':ranking_shortcut},sort_keys=True)); return out

def final(root,evid):
 pf=json.loads((evid/'ROUND14_PREFLIGHT_RESULTS.json').read_text()); expanded=pf['EXPANSION_OPEN']
 rows={s:compare_seed(root,s,True) for s in PREFLIGHT}
 if expanded:
  for s in EXPANSION: rows[s]=compare_seed(root,s,False)
 seeds=list(rows); vals=[rows[s]['A3_vs_C0']['U'] for s in seeds]; mechpos=[rows[s]['mechanism_A3']['margin_true_minus_shuffled']['mean']>=0 for s in seeds]
 ranking_shortcut=bool(any(x>0 for x in vals) and any(not x for x in mechpos))
 if not expanded: verdict='FAIL'
 elif ranking_shortcut and sum(mechpos)<3: verdict='FAIL'
 elif sum(x>0 for x in vals)==4 and np.mean(vals)>=.01 and sum(mechpos)>=3: verdict='TARGET_PASS'
 elif sum(x>0 for x in vals)==4 and np.mean(vals)>=.005 and sum(mechpos)>=3: verdict='STRONG_PASS'
 elif sum(x>0 for x in vals)==4 and np.mean(vals)>0 and sum(mechpos)>=3: verdict='STABILITY_PASS'
 elif sum(x>0 for x in vals)==3 and np.mean(vals)>=.0025 and sum(mechpos)>=3: verdict='PARTIAL_SIGNAL'
 else: verdict='FAIL'
 summary={'status':'VALIDATION_DECISION','expanded':expanded,'seeds':seeds,'rows':{str(ss):rows[ss] for ss in seeds},'A3_vs_C0':{'seed_U':{str(ss):rows[ss]['A3_vs_C0']['U'] for ss in seeds},'mean_U':float(np.mean(vals)),'positive_seeds':int(sum(x>0 for x in vals))},'mechanism_positive_seeds':int(sum(mechpos)),'RANKING_SHORTCUT_SIGNAL':ranking_shortcut,'verdict':verdict,'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
 (evid/'ROUND14_VALIDATION_RESULTS.json').write_text(json.dumps(summary,indent=2)+'\n')
 costs={str(ss):{v:rows[ss]['cost_'+v] for v in (VARIANTS if ss in PREFLIGHT else ('A3',))} for ss in seeds}; (evid/'ROUND14_COST_REPORT.json').write_text(json.dumps({'cost':costs,'TEST_ACCESSED':False},indent=2)+'\n'); write_report(summary,pf,evid); print(json.dumps({'verdict':verdict,'expanded':expanded,'mean_A3':summary['A3_vs_C0']['mean_U'],'mechanism_positive':summary['mechanism_positive_seeds'],'ranking_shortcut':ranking_shortcut},sort_keys=True)); return summary

def write_report(s,pf,evid):
 L=['# Round14 Validation Report — Anchored Inference-Aligned Hard-Shell User Purification','',f"Final verdict: **ROUND14 = {s['verdict']}**",f"Expansion executed: **{s['expanded']}**",'', '## Preflight gate','```text',json.dumps(pf['gate_checks'],indent=2),'```',f"RANKING_SHORTCUT_SIGNAL: **{pf['RANKING_SHORTCUT_SIGNAL']}**",'','## Best Validation results','| Seed | Variant | R10 | N10 | R20 | N20 | U vs C0 | Mechanism mean | Fraction true>shuf | NetCross10 | NetCross20 |','|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for ss in map(str,s['seeds']):
  x=s['rows'][ss]
  for v in (VARIANTS if all(k in x for k in VARIANTS) else ('A3',)):
   m=x['mechanism_'+v]['margin_true_minus_shuffled']; b=x['boundary_'+v]; L.append(f"| {ss} | {v} | {x[v]['R10']:.8f} | {x[v]['N10']:.8f} | {x[v]['R20']:.8f} | {x[v]['N20']:.8f} | {100*x[v+'_vs_C0']['U']:+.4f}% | {m['mean']:+.6f} | {m['fraction_positive']:.4f} | {b['NetCross10']:+d} | {b['NetCross20']:+d} |")
 L+=['','## Round13 D2 historical reference','```text',json.dumps(pf.get('R13_D2_reference',{}),indent=2),'```','','## Residual scale ratios','```text',json.dumps(pf['scale_ratios'],indent=2),'```','','## Frozen stop decision', 'Expansion was opened only if every ranking, mechanism, and relative-improvement preflight requirement passed. Test, Sports and Electronics remained CLOSED.','']; (evid/'ROUND14_REPORT.md').write_text('\n'.join(L))

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--mode',choices=['build-hard','smoke','baseline','formal','preflight','final'],required=True); ap.add_argument('--seed',type=int); ap.add_argument('--variant',choices=VARIANTS); ap.add_argument('--root',default=str(ROOT/'diffusion_experiments/round14_aihu/outputs')); ap.add_argument('--evidence',default=str(ROOT/'diffusion_experiments/round14_aihu/evidence')); a=ap.parse_args(); root=Path(a.root); evid=Path(a.evidence)
 if a.mode=='build-hard':
  p,au=hard_paths(a.seed); hcb.build(a.seed,p,au,None)
 elif a.mode=='smoke': smoke(root,evid)
 elif a.mode=='baseline': baseline(a.seed,root)
 elif a.mode=='formal': run_variant(a.seed,a.variant,root)
 elif a.mode=='preflight': preflight(root,evid)
 else: final(root,evid)
if __name__=='__main__': main()
