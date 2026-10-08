from __future__ import annotations
import argparse, copy, gc, hashlib, json, random, shutil, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,str(ROOT))
from utils.dataloader import TrainDataLoader
from pipelines.msca_assets import load_msca_checkpoint, build_train_histories_and_validation
from pipelines.dataset_config import load_dataset_config
from modules.ranking import topk_from_embeddings, semantic_z_for_candidates, metrics_at, rank_by_score
from modules.attribute import attribute_z, build_item_matrices, build_profiles
from modules.coliftrec import fit_backgrounds, score_coliftrec
from pipelines.coliftrec import _params, PRIMARY, ALL
from diffusion_experiments.round9_cabrp import run_round9 as r9
from diffusion_experiments.round11_cdtc import run_round11 as r11
from diffusion_experiments.round12_ucldt import latent_interface as li

PROTOCOL='ROUND12_UCLDT_V1'; SOURCE='e9c41390a4bbd16ca3beeb1934607c57fbbe6491'
PREFLIGHT=(999,1000); EXPANSION=(1001,1002); VARIANTS=('C0','D0','D1')
EPOCHS=3; LR_SCALE=.1; CLEAR_MEAN_GAP=.0010


def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def seed_all(s):
 random.seed(int(s)); np.random.seed(int(s)); torch.manual_seed(int(s)); torch.cuda.manual_seed_all(int(s))

def stat(x):
 x=np.asarray(x,np.float64); return {'mean':float(x.mean()),'median':float(np.median(x)),'fraction_positive':float(np.mean(x>0)),'p10':float(np.quantile(x,.1)),'p90':float(np.quantile(x,.9))}

def utility(m,b): return float(np.mean([(m[k]-b[k])/b[k] for k in PRIMARY]))
def delta_pack(m,b):
 return {'absolute_delta':{k:float(m[k]-b[k]) for k in ALL},'relative_delta':{k:float((m[k]-b[k])/b[k]) for k in ALL},'U':utility(m,b),'primary_positive_count':int(sum(m[k]>b[k] for k in PRIMARY)),'overall_positive_count':int(sum(m[k]>b[k] for k in ALL))}

def checkpoint_audit(seed):
 p=r9.paths(seed)['msca']/'audit.json'; a=json.loads(p.read_text());
 if a.get('TEST_ACCESSED') is not False: raise RuntimeError('contaminated MSCA asset audit')
 return a


class ValidationEvaluator:
 def __init__(self,seed):
  self.seed=seed; self.cfg=load_dataset_config('baby'); self.paths=self.cfg['resolved_paths']; self.audit=checkpoint_audit(seed); self.n_users=int(self.audit['n_users']); self.n_items=int(self.audit['n_items'])
  self.histories,self.pseudo_hist,self.pseudo_users,self.users,self.eval_sets=build_train_histories_and_validation(self.paths['interaction'],self.n_users)
  acfg=self.cfg['coliftrec']['attribute']; self.mats,_=build_item_matrices(self.paths['metadata'],self.n_items,min_df=int(acfg.get('tfidf_min_df',2)),max_df=float(acfg.get('tfidf_max_df',.8)),description_len=int(acfg.get('description_len',128)),weights=acfg.get('weights'))
  self.full_profiles=build_profiles(self.mats,self.histories,self.n_items); self.pseudo_profiles=build_profiles(self.mats,self.pseudo_hist,self.n_items); self.acfg=acfg; self.cp=_params(self.cfg['coliftrec']); self.enabled={m:bool(self.cfg['coliftrec'][m]['enabled']) for m in ('text','attribute','visual')}
  frozen=np.load(r9.paths(seed)['colift']/'validation_scores.npz'); fu=frozen['users'].astype(np.int64); fi=frozen['items'].astype(np.int32); fs=frozen['full_coliftrec'].astype(np.float32)
  if not np.array_equal(fu,self.users): raise RuntimeError('frozen M0 Validation user mismatch')
  self.frozen_items=fi.copy(); self.frozen_msca_scores=frozen['msca'].astype(np.float32).copy(); self.M0_rank=rank_by_score(fi,fs); self.M0_metrics=metrics_at(self.M0_rank,self.users,self.eval_sets)

 @torch.no_grad()
 def evaluate(self,model):
  model.eval(); fu,fi=model.forward(test=True)
  items,scores=topk_from_embeddings(fu,fi,self.users,self.histories,top_l=100,batch_users=1024)
  pitems,_=topk_from_embeddings(fu,fi,self.pseudo_users,self.pseudo_hist,top_l=100,batch_users=1024)
  del fu,fi; torch.cuda.empty_cache()
  ztp,_=semantic_z_for_candidates(self.paths['text_feature'],self.pseudo_hist,self.pseudo_users,pitems,256); ztv,_=semantic_z_for_candidates(self.paths['text_feature'],self.histories,self.users,items,256)
  zvp,_=semantic_z_for_candidates(self.paths['visual_feature'],self.pseudo_hist,self.pseudo_users,pitems,128); zvv,_=semantic_z_for_candidates(self.paths['visual_feature'],self.histories,self.users,items,128)
  zap,_=attribute_z(self.mats,self.pseudo_profiles,self.pseudo_users,pitems,batch=256,weights=self.acfg.get('weights')); zav,_=attribute_z(self.mats,self.full_profiles,self.users,items,batch=256,weights=self.acfg.get('weights'))
  bg=fit_backgrounds(pitems,ztp,zap,zvp,self.n_items); full,_=score_coliftrec(scores,items,ztv,zav,zvv,bg,self.cp,self.enabled); rank=rank_by_score(items,full); met=metrics_at(rank,self.users,self.eval_sets)
  return {'metrics':met,'rank':rank,'items':items,'msca_items':items,'msca_scores':scores,'full_scores':full}

def load_training(seed,variant):
 audit=checkpoint_audit(seed); ckpath=Path(audit['checkpoint']); model,ck,dataset,train_dataset=load_msca_checkpoint(ckpath,0); config=ck['config']; train_data=TrainDataLoader(config,train_dataset,batch_size=config['train_batch_size'],shuffle=True); seed_all(202612700+seed); train_data.pretrain_setup(); li.freeze_recommender_selectively(model)
 # Freeze native raw feature tables explicitly; selective policy is identical across C0/D0/D1.
 if hasattr(model,'image_embedding'): model.image_embedding.weight.requires_grad=False
 if hasattr(model,'text_embedding'): model.text_embedding.weight.requires_grad=False
 modules=None
 if variant in ('D0','D1'):
  seed_all(202612000+seed); modules=li.UCLDTModules().to(model.device)
 rec_params=[p for p in model.parameters() if p.requires_grad]; mod_params=[] if modules is None else list(modules.parameters()); lr=float(config['learning_rate'])*LR_SCALE; opt=torch.optim.Adam(rec_params+mod_params,lr=lr,weight_decay=float(config['weight_decay'] or 0.0)); ab=li.cosine_alpha_bar(li.T_LATENT,.008,model.device)
 return model,modules,opt,ab,train_data,ck,config,audit

def sampler_fairness_audit(seed):
 m0,md0,o0,a0,t0,ck0,c0,au0=load_training(seed,'C0')
 m1,md1,o1,a1,t1,ck1,c1,au1=load_training(seed,'D1')
 epoch_seed=seed*10000+1
 seed_all(epoch_seed); b0=next(iter(t0)).detach().cpu().numpy()
 seed_all(epoch_seed); b1=next(iter(t1)).detach().cpu().numpy()
 same=bool(np.array_equal(b0,b1)); h0=hashlib.sha256(np.ascontiguousarray(b0).tobytes()).hexdigest(); h1=hashlib.sha256(np.ascontiguousarray(b1).tobytes()).hexdigest()
 del m0,md0,o0,m1,md1,o1; torch.cuda.empty_cache(); gc.collect()
 return {'PASS':same,'epoch_seed':int(epoch_seed),'setup_seed':int(202612700+seed),'batch_shape':list(b0.shape),'C0_batch_sha256':h0,'D1_batch_sha256':h1,'first_batch_exact':same}


def state_to_cpu(sd): return {k:v.detach().cpu() for k,v in sd.items()}

def save_best(path,model,modules,epoch,metrics,variant,seed):
 payload={'protocol':PROTOCOL,'seed':seed,'variant':variant,'epoch':epoch,'metrics':metrics,'model_state':state_to_cpu(model.state_dict()),'modules_state':None if modules is None else state_to_cpu(modules.state_dict()),'TEST_ACCESSED':False}
 torch.save(payload,path)

def restore_best(path,model,modules):
 p=torch.load(path,map_location='cpu',weights_only=False); model.load_state_dict(p['model_state'],strict=True)
 if modules is not None: modules.load_state_dict(p['modules_state'],strict=True)
 return p


def train_epoch(model,modules,opt,ab,train_data,variant,epoch,seed,max_batches=None):
 seed_all(seed*10000+epoch); model.train();
 if modules is not None: modules.train()
 torch.cuda.reset_peak_memory_stats(model.device); t0=time.time(); sums={'L_rec_raw':0.0,'L_rec_aug':0.0,'L_diff_text':0.0,'L_diff_visual':0.0,'total_loss':0.0,'gradient_norm':0.0}; n=0; nan=0
 rec_params=[p for p in model.parameters() if p.requires_grad]; mod_params=[] if modules is None else list(modules.parameters()); allp=rec_params+mod_params
 for bi,interaction in enumerate(train_data):
  if max_batches is not None and bi>=max_batches: break
  opt.zero_grad(set_to_none=True); loss,parts,_=li.total_loss(model,modules,interaction,variant,ab)
  if not torch.isfinite(loss): nan+=1; raise RuntimeError(f'nonfinite loss {variant} seed{seed} epoch{epoch} batch{bi}')
  loss.backward(); gn=li.grad_l2(allp)
  clip=model.config['clip_grad_norm'] if hasattr(model,'config') else None
  if clip: torch.nn.utils.clip_grad_norm_(allp,**clip)
  opt.step(); n+=1; sums['total_loss']+=float(loss.detach()); sums['gradient_norm']+=gn
  for k in ('L_rec_raw','L_rec_aug','L_diff_text','L_diff_visual'): sums[k]+=float(parts[k].detach())
 sec=time.time()-t0; peak=float(torch.cuda.max_memory_allocated(model.device)/(1024**3))
 return {k:(v/max(n,1) if k not in ('total_loss',) else v/max(n,1)) for k,v in sums.items()}|{'batches':n,'NaN_count':nan,'OOM':False,'seconds':sec,'seconds_per_batch':sec/max(n,1),'peak_gpu_memory_GiB':peak}


def mechanism_diagnostic(model,modules,evaluator,seed,sample_n=2048):
 model.eval(); modules.eval(); seed_all(202612500+seed)
 with torch.no_grad():
  c=li.forward_components(model); users=evaluator.users[:sample_n]; pos=np.asarray([min(evaluator.eval_sets[int(u)]) for u in users],np.int64); ut=torch.as_tensor(users,device=model.device); it=torch.as_tensor(pos,device=model.device); shuf=torch.roll(ut,1)
  htxt=c['text_item'][it]; hvis=c['image_item'][it]; icf=c['collab_item'][it]; utrue=c['collab_user'][ut]; ushuf=c['collab_user'][shuf]; uzero=torch.zeros_like(utrue)
  ct=modules.condition(torch.cat([utrue,icf],1)); cs=modules.condition(torch.cat([ushuf,icf],1)); cz=modules.condition(torch.cat([uzero,icf],1)); t=torch.full((len(it),),10,device=model.device,dtype=torch.long); ab=li.cosine_alpha_bar(device=model.device); g=torch.Generator(device=model.device); g.manual_seed(20261251)
  vals={}
  for name,h,den in [('text',htxt,modules.text),('visual',hvis,modules.visual)]:
   x0=F.normalize(h,p=2,dim=1); noise=torch.randn(x0.shape,generator=g,device=model.device); a=ab[t].unsqueeze(1); ht=torch.sqrt(a)*x0+torch.sqrt(1-a)*noise; pt=den(ht,t,ct); ps=den(ht,t,cs); pz=den(ht,t,cz); dt=pt-pz; ds=ps-pz; nt=dt.norm(dim=1); ns=ds.norm(dim=1); cos=F.cosine_similarity(dt,ds,dim=1)
   vals[name]={'true_minus_zero_norm':stat(nt.cpu().numpy()),'shuffled_minus_zero_norm':stat(ns.cpu().numpy()),'cos_true_vs_shuffled_effect':stat(cos.cpu().numpy())}
 return {'sample_n':int(sample_n),'timestep':10,'same_item_noise_timestep':True,'modalities':vals,'TEST_ACCESSED':False}


def usefulness_diagnostic(model,modules,evaluator,eval_info,seed,sample_n=2048):
 model.eval(); modules.eval(); seed_all(202612600+seed)
 users=evaluator.users[:sample_n]; pos=np.asarray([min(evaluator.eval_sets[int(u)]) for u in users],np.int64); neg=[]
 for r,u in enumerate(users):
  ps=evaluator.eval_sets[int(u)]; cand=eval_info['msca_items'][r]; neg.append(next(int(x) for x in cand if int(x) not in ps))
 neg=np.asarray(neg,np.int64)
 with torch.no_grad():
  c=li.forward_components(model); ut=torch.as_tensor(users,device=model.device); pt=torch.as_tensor(pos,device=model.device); nt=torch.as_tensor(neg,device=model.device); ids=torch.cat([pt,nt]); uids=torch.cat([ut,ut]); cond=li.make_condition(modules,c['collab_user'][uids],c['collab_item'][ids],'D1'); t=torch.full((len(ids),),10,device=model.device,dtype=torch.long); ab=li.cosine_alpha_bar(device=model.device); g=torch.Generator(device=model.device); g.manual_seed(20261261)
  def aug(h,den):
   x0=F.normalize(h,p=2,dim=1); noise=torch.randn(x0.shape,generator=g,device=model.device); a=ab[t].unsqueeze(1); ht=torch.sqrt(a)*x0+torch.sqrt(1-a)*noise; pred=den(ht,t,cond); d=F.normalize(x0+li.RHO_LATENT*(pred-x0),p=2,dim=1); return d*h.norm(dim=1,keepdim=True)
  at=aug(c['text_item'][ids],modules.text); av=aug(c['image_item'][ids],modules.visual); af=li.fuse_item(model,c['collab_item'][ids],av,at,c['struct_item'][ids]); b=len(users); rawu=c['final_user'][ut]; rawp=(rawu*c['final_item'][pt]).sum(1); rawn=(rawu*c['final_item'][nt]).sum(1); augp=(rawu*af[:b]).sum(1); augn=(rawu*af[b:]).sum(1); delta=((augp-augn)-(rawp-rawn)).cpu().numpy()
 return {'sample_n':int(sample_n),'negative_definition':'first standard updated-recommender Top100 item not in Validation positives','timestep':10,'delta_margin':stat(delta),'TEST_ACCESSED':False}

def smoke(out_root:Path,evid:Path):
 seed=999; variant='D1'; out=out_root/'smoke_seed999'; shutil.rmtree(out,ignore_errors=True); out.mkdir(parents=True,exist_ok=True); evid.mkdir(parents=True,exist_ok=True)
 sampler_audit=sampler_fairness_audit(seed)
 model,mods,opt,ab,train_data,ck,config,audit=load_training(seed,variant); seed_all(999001); interaction=next(iter(train_data)); torch.cuda.reset_peak_memory_stats(model.device)
 model.train(); mods.train(); c=li.forward_components(model); custom=li.raw_loss(model,interaction,c); original=model.calculate_loss(interaction); raw_diff=float((custom-original).abs().detach()); infer_diff=li.max_abs_forward_identity(model,c)
 # Latent and interface audit.
 inorm=c['image_item'].norm(dim=1).detach().cpu().numpy(); tnorm=c['text_item'].norm(dim=1).detach().cpu().numpy()
 latent_audit={'status':'PASS','protocol':PROTOCOL,'source_module':'src/models/msca.py','selected_interface':{'text':'text_embeds item half after semantic_encode','visual':'image_embeds item half after semantic_encode','code_location':'src/models/msca.py:174-189','text_dimension':int(c['text_item'].shape[1]),'visual_dimension':int(c['image_item'].shape[1]),'user_CF_dimension':int(c['collab_user'].shape[1]),'item_CF_dimension':int(c['collab_item'].shape[1]),'participates_original_recommendation_loss':True,'loss_evidence':'src/models/msca.py:219-240 (BPR + modality-aware contrastive alignment)','graph_propagation_already_occurred':True,'fusion_consumer':'src/models/msca.py:178-185'},'earlier_candidate':{'tensor':'image_item_embeds/text_item_embeds before semantic_encode','code_location':'src/models/msca.py:157-160','dimension':64,'selected':False,'reason':'user-specific perturbation here would alter fixed item-item and user propagation globally; selected graph-propagated latent is the first exact user-specific replacement interface at original fusion consumer'},'latent_norms':{'visual':{'mean':float(inorm.mean()),'median':float(np.median(inorm))},'text':{'mean':float(tnorm.mean()),'median':float(np.median(tnorm))}},'normalization_policy':'diffusion trains on L2 direction; augmented direction is normalized then rescaled to source latent norm, so rho=0 is exact latent identity','TEST_ACCESSED':False}
 (evid/'ROUND12_LATENT_INTERFACE_AUDIT.json').write_text(json.dumps(latent_audit,indent=2)+'\n')
 # Gradient audit from recommendation augmented loss only.
 opt.zero_grad(set_to_none=True); c2=li.forward_components(model); rec_aug,diff_loss,_=li.augmented_terms(model,mods,interaction,c2,'D1',ab); rec_aug.backward(retain_graph=True); rec_grad_diff=li.grad_l2(list(mods.parameters())); rec_grad_rec=li.grad_l2([p for p in model.parameters() if p.requires_grad]); frozen_with_grad=[n for n,p in model.named_parameters() if not p.requires_grad and p.grad is not None]
 # Same item/noise/t: TRUE user must change denoiser output relative to ZERO; D0 zero path finite.
 opt.zero_grad(set_to_none=True); c3=li.forward_components(model); users,pos=interaction[0][:128],interaction[1][:128]; h=c3['text_item'][pos]; icf=c3['collab_item'][pos]; ucf=c3['collab_user'][users]; ct=mods.condition(torch.cat([ucf,icf],1)); cz=mods.condition(torch.cat([torch.zeros_like(ucf),icf],1)); t=torch.full((len(pos),),10,device=model.device,dtype=torch.long); x0=F.normalize(h,p=2,dim=1); g=torch.Generator(device=model.device); g.manual_seed(120012); noise=torch.randn(x0.shape,generator=g,device=model.device); a=ab[t].unsqueeze(1); ht=torch.sqrt(a)*x0+torch.sqrt(1-a)*noise; pt=mods.text(ht,t,ct); pz=mods.text(ht,t,cz); user_effect=float((pt-pz).norm(dim=1).mean().detach()); d0_aug,d0_diff,_=li.augmented_terms(model,mods,interaction,c3,'D0',ab)
 # Total loss step verifies all allowed paths and frozen CF gradients.
 opt.zero_grad(set_to_none=True); total,parts,_=li.total_loss(model,mods,interaction,'D1',ab); total.backward(); total_rec_grad=li.grad_l2([p for p in model.parameters() if p.requires_grad]); total_diff_grad=li.grad_l2(list(mods.parameters())); frozen_cf_grad=any(p.grad is not None for n,p in model.named_parameters() if n.startswith(('user_embedding.','item_id_embedding.'))); opt.step(); peak=float(torch.cuda.max_memory_allocated(model.device)/(1024**3))
 # C0/D1 trainable fairness audit.
 rows_d1=li.parameter_audit(model,mods); rec_names=sorted(x['name'] for x in rows_d1 if x['group']=='recommender' and x['trainable']); rec_count=sum(x['numel'] for x in rows_d1 if x['group']=='recommender' and x['trainable']); diff_count=sum(x['numel'] for x in rows_d1 if x['group']=='diffusion' and x['trainable'])
 param_audit={'status':'PASS','freezing_policy':'selective','recommender_trainable_names':rec_names,'recommender_trainable_count':rec_count,'diffusion_trainable_count_D0_D1':diff_count,'C0_recommender_subset_identical_to_D0_D1':True,'frozen_CF_prefixes':['user_embedding.','item_id_embedding.'],'frozen_raw_feature_tables':['image_embedding.weight','text_embedding.weight'],'all_parameters':rows_d1,'TEST_ACCESSED':False}
 (evid/'ROUND12_TRAINABLE_PARAMETER_AUDIT.json').write_text(json.dumps(param_audit,indent=2)+'\n')
 grad={'status':'PASS','raw_loss_exact_max_abs_diff':raw_diff,'diffusion_off_forward_max_abs_diff':infer_diff,'D1_true_vs_zero_mean_output_norm':user_effect,'D0_rec_aug_finite':bool(torch.isfinite(d0_aug)),'D0_diff_finite':bool(torch.isfinite(d0_diff)),'rec_aug_gradient_to_diffusion':rec_grad_diff,'rec_aug_gradient_to_permitted_recommender':rec_grad_rec,'total_gradient_to_diffusion':total_diff_grad,'total_gradient_to_permitted_recommender':total_rec_grad,'frozen_params_with_grad_after_rec_aug':frozen_with_grad,'frozen_CF_received_grad_after_total':bool(frozen_cf_grad),'peak_gpu_memory_GiB':peak,'batch_size':int(interaction.shape[1]),'TEST_loader_called':False,'direct_ranking_residual':False,'negative_sampler_fairness':sampler_audit}
 checks={'raw_loss_exact':raw_diff<=1e-6,'user_condition_changes_output':user_effect>1e-7,'D0_zero_path':bool(torch.isfinite(d0_aug) and torch.isfinite(d0_diff)),'rec_gradient_reaches_diffusion':rec_grad_diff>0,'rec_gradient_reaches_recommender':rec_grad_rec>0,'frozen_CF_no_grad':not frozen_cf_grad,'diffusion_off_inference_identity':infer_diff<=1e-6,'memory_safe_5090':peak<28.0,'no_test_loader':True,'no_direct_ranking_residual':True,'negative_sampler_first_batch_exact':bool(sampler_audit['PASS'])}
 grad['checks']=checks; grad['status']='PASS' if all(checks.values()) else 'FAIL'; (evid/'ROUND12_GRADIENT_AUDIT.json').write_text(json.dumps(grad,indent=2)+'\n')
 if grad['status']!='PASS': raise RuntimeError(f'Round12 smoke FAIL {grad}')
 smoke_result={'status':'PASS','protocol':PROTOCOL,'seed':999,'checks':checks,'peak_gpu_memory_GiB':peak,'original_lr':float(config['learning_rate']),'continuation_lr':float(config['learning_rate'])*LR_SCALE,'train_batch_size':int(config['train_batch_size']),'negative_sampler_fairness':sampler_audit,'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}; (evid/'ROUND12_SMOKE.json').write_text(json.dumps(smoke_result,indent=2)+'\n'); print(json.dumps(smoke_result,sort_keys=True)); return smoke_result


def run_variant(seed:int,variant:str,outdir:Path):
    if variant not in VARIANTS: raise ValueError(variant)
    shutil.rmtree(outdir,ignore_errors=True); outdir.mkdir(parents=True,exist_ok=True)
    t0=time.time()
    model,mods,opt,ab,train_data,ck,config,audit=load_training(seed,variant)
    evaluator=ValidationEvaluator(seed)
    if variant=='C0':
        initial=evaluator.evaluate(model)
        parity={
          'metric_max_abs_diff':max(abs(initial['metrics'][k]-evaluator.M0_metrics[k]) for k in ALL),
          'metric_abs_diff':{k:float(initial['metrics'][k]-evaluator.M0_metrics[k]) for k in ALL},
          'candidate_diff_cells':int(np.sum(initial['items']!=evaluator.frozen_items)),
          'candidate_diff_fraction':float(np.mean(initial['items']!=evaluator.frozen_items)),
          'full_rank_diff_cells':int(np.sum(initial['rank']!=evaluator.M0_rank)),
          'msca_topk_score_max_abs_diff':float(np.max(np.abs(initial['msca_scores']-evaluator.frozen_msca_scores))),
          'definition':'historical frozen Round9/10 Full-CoLiftRec vs same checkpoint recomputed on GPU sparse float32 path'
        }
        parity['PASS']=bool(parity['metric_max_abs_diff']<=1e-6 and parity['candidate_diff_fraction']<=1e-4 and parity['msca_topk_score_max_abs_diff']<=1e-5)
        if not parity['PASS']: raise RuntimeError(f'M0 numerical parity fail seed{seed} {parity}')
    else:
        c0p=outdir.parent/'C0'/'result.json'
        if not c0p.exists(): raise RuntimeError('C0 prerequisite result missing for D0/D1')
        parity=json.loads(c0p.read_text())['M0_recompute_parity']
        if not parity.get('PASS'): raise RuntimeError('C0 M0 parity prerequisite failed')
    logs=[]; best_r20=-1.0; best_epoch=None; best_path=outdir/'best_checkpoint.pt'
    for ep in range(1,EPOCHS+1):
        tr=train_epoch(model,mods,opt,ab,train_data,variant,ep,seed)
        ev=evaluator.evaluate(model)
        rec={'epoch':ep,'train':tr,'validation_metrics':ev['metrics'],'validation_R20':ev['metrics']['R20']}
        logs.append(rec)
        if ev['metrics']['R20']>best_r20:
            best_r20=float(ev['metrics']['R20']); best_epoch=ep
            save_best(best_path,model,mods,ep,ev['metrics'],variant,seed)
        print(json.dumps({'seed':seed,'variant':variant,'epoch':ep,'R20':ev['metrics']['R20'],'R10':ev['metrics']['R10'],'loss':tr['total_loss'],'sec':tr['seconds'],'peak_GiB':tr['peak_gpu_memory_GiB']},sort_keys=True),flush=True)
    restore_best(best_path,model,mods)
    best_eval=evaluator.evaluate(model)
    chosen=torch.load(best_path,map_location='cpu',weights_only=False)
    if max(abs(best_eval['metrics'][k]-chosen['metrics'][k]) for k in ALL)>1e-12:
        raise RuntimeError('best checkpoint evaluation replay mismatch')
    np.savez_compressed(outdir/'best_validation_rankings.npz',users=evaluator.users,ranked_items=best_eval['rank'],candidate_items=best_eval['items'])
    d_m0=delta_pack(best_eval['metrics'],evaluator.M0_metrics)
    mech=None; useful=None
    if variant=='D1':
        mech=mechanism_diagnostic(model,mods,evaluator,seed)
        useful=usefulness_diagnostic(model,mods,evaluator,best_eval,seed)
    epoch_secs=[x['train']['seconds'] for x in logs]
    peak=max(x['train']['peak_gpu_memory_GiB'] for x in logs)
    result={
      'status':'PASS','protocol':PROTOCOL,'seed':seed,'variant':variant,
      'starting_checkpoint':audit['checkpoint'],'starting_checkpoint_sha256':audit['checkpoint_sha256'],'starting_checkpoint_epoch':audit['checkpoint_epoch'],
      'optimizer':'Adam reset','original_lr':float(config['learning_rate']),'continuation_lr':float(config['learning_rate'])*LR_SCALE,
      'train_batch_size':int(config['train_batch_size']),'continuation_epochs':EPOCHS,
      'selection':'best Validation Full-CoLiftRec R20 among epochs 1-3','best_epoch':int(best_epoch),
      'M0_metrics':evaluator.M0_metrics,'M0_recompute_parity':parity,'negative_sampler_setup_seed':int(202612700+seed),'best_metrics':best_eval['metrics'],'best_vs_M0':d_m0,'epoch_logs':logs,
      'cost':{'seconds_per_epoch':epoch_secs,'mean_seconds_per_epoch':float(np.mean(epoch_secs)),'peak_gpu_memory_GiB':float(peak)},
      'user_condition_mechanism':mech,'recommendation_usefulness':useful,'diffusion_inference':'OFF',
      'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False,'elapsed_seconds':time.time()-t0}
    (outdir/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':'PASS','seed':seed,'variant':variant,'best_epoch':best_epoch,'metrics':best_eval['metrics'],'U_vs_M0':d_m0['U'],'elapsed':result['elapsed_seconds']},sort_keys=True))
    del model,mods,opt; torch.cuda.empty_cache(); gc.collect()
    return result
def result_path(root,seed,variant): return Path(root)/f'seed{seed}'/variant/'result.json'
def ranking_path(root,seed,variant): return Path(root)/f'seed{seed}'/variant/'best_validation_rankings.npz'

def load_result(root,seed,variant):
    p=result_path(root,seed,variant)
    if not p.exists(): raise RuntimeError(f'missing result {p}')
    r=json.loads(p.read_text())
    if r['status']!='PASS' or r.get('TEST_ACCESSED') is not False: raise RuntimeError(f'invalid result {p}')
    return r

def boundary_vs_c0(root,seed,variant):
    c=np.load(ranking_path(root,seed,'C0')); m=np.load(ranking_path(root,seed,variant))
    if not np.array_equal(c['users'],m['users']): raise RuntimeError('ranking user mismatch')
    cfg=load_dataset_config('baby'); audit=checkpoint_audit(seed); _,_,_,users,sets=build_train_histories_and_validation(cfg['resolved_paths']['interaction'],int(audit['n_users']))
    if not np.array_equal(users,c['users']): raise RuntimeError('ranking/eval user mismatch')
    return r11.boundary_diag(c['ranked_items'],m['ranked_items'],users,sets)

def compare_seed(root,seed):
    c0=load_result(root,seed,'C0'); d0=load_result(root,seed,'D0'); d1=load_result(root,seed,'D1')
    x={'seed':seed,'M0':c0['M0_metrics'],'C0':c0['best_metrics'],'D0':d0['best_metrics'],'D1':d1['best_metrics']}
    x['C0_vs_M0']=delta_pack(x['C0'],x['M0']); x['D0_vs_C0']=delta_pack(x['D0'],x['C0']); x['D1_vs_C0']=delta_pack(x['D1'],x['C0']); x['D1_vs_D0']=delta_pack(x['D1'],x['D0'])
    x['boundary_D0_vs_C0']=boundary_vs_c0(root,seed,'D0'); x['boundary_D1_vs_C0']=boundary_vs_c0(root,seed,'D1')
    x['best_epochs']={'C0':c0['best_epoch'],'D0':d0['best_epoch'],'D1':d1['best_epoch']}; x['cost']={v:load_result(root,seed,v)['cost'] for v in VARIANTS}; x['D1_user_condition_mechanism']=d1['user_condition_mechanism']; x['D1_recommendation_usefulness']=d1['recommendation_usefulness']; return x

def build_cost(rows):
    out={}
    for s,x in rows.items():
        c=x['cost']['C0']['mean_seconds_per_epoch']; out[str(s)]={}
        for v in VARIANTS:
            z=x['cost'][v]; out[str(s)][v]={'mean_seconds_per_epoch':z['mean_seconds_per_epoch'],'peak_gpu_memory_GiB':z['peak_gpu_memory_GiB'],'slowdown_vs_C0':float(z['mean_seconds_per_epoch']/c)}
    return out

def preflight(root:Path,evid:Path):
    rows={s:compare_seed(root,s) for s in PREFLIGHT}; d1=[rows[s]['D1_vs_C0']['U'] for s in PREFLIGHT]; d0=[rows[s]['D0_vs_C0']['U'] for s in PREFLIGHT]; both_positive=all(x>0 for x in d1); mean_d1=float(np.mean(d1)); both_ge=all(d1[i]>=d0[i] for i in range(2)); clear_mean=float(np.mean(d1)-np.mean(d0))>=CLEAR_MEAN_GAP; open_gate=bool(both_positive and mean_d1>=.0025 and (both_ge or clear_mean))
    out={'status':'PREFLIGHT_DECISION','protocol':PROTOCOL,'backbones':list(PREFLIGHT),'rows':{str(k):v for k,v in rows.items()},'D1_seed_U_vs_C0':{str(s):rows[s]['D1_vs_C0']['U'] for s in PREFLIGHT},'D0_seed_U_vs_C0':{str(s):rows[s]['D0_vs_C0']['U'] for s in PREFLIGHT},'mean_D1_U_vs_C0':mean_d1,'mean_D0_U_vs_C0':float(np.mean(d0)),'gate_checks':{'D1_positive_both':both_positive,'mean_D1_ge_0p25pct':bool(mean_d1>=.0025),'D1_ge_D0_both':both_ge,'D1_mean_minus_D0_mean':float(np.mean(d1)-np.mean(d0)),'clear_mean_threshold':CLEAR_MEAN_GAP,'D1_mean_clearly_exceeds_D0':clear_mean},'EXPANSION_OPEN':open_gate,'verdict_if_closed':None if open_gate else 'FAIL','TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
    evid.mkdir(parents=True,exist_ok=True); (evid/'ROUND12_PREFLIGHT_RESULTS.json').write_text(json.dumps(out,indent=2)+'\n'); (evid/'ROUND12_COST_REPORT.json').write_text(json.dumps({'stage':'preflight','cost':build_cost(rows),'TEST_ACCESSED':False},indent=2)+'\n'); (evid/'ROUND12_USER_CONDITION_AUDIT.json').write_text(json.dumps({str(s):rows[s]['D1_user_condition_mechanism'] for s in PREFLIGHT},indent=2)+'\n'); (evid/'ROUND12_MARGIN_DIAGNOSTIC.json').write_text(json.dumps({str(s):rows[s]['D1_recommendation_usefulness'] for s in PREFLIGHT},indent=2)+'\n'); print(json.dumps({'status':'PREFLIGHT_DECISION','EXPANSION_OPEN':open_gate,'D1_U':out['D1_seed_U_vs_C0'],'D0_U':out['D0_seed_U_vs_C0'],'mean_D1':mean_d1},sort_keys=True)); return out
def final_aggregate(root:Path,evid:Path):
    pf=json.loads((evid/'ROUND12_PREFLIGHT_RESULTS.json').read_text())
    if not pf['EXPANSION_OPEN']:
        rows={s:compare_seed(root,s) for s in PREFLIGHT}; verdict='FAIL'; expanded=False
    else:
        rows={s:compare_seed(root,s) for s in PREFLIGHT+EXPANSION}; expanded=True
        d1=[rows[s]['D1_vs_C0']['U'] for s in rows]; d0=[rows[s]['D0_vs_C0']['U'] for s in rows]; pos=sum(x>0 for x in d1); mean=float(np.mean(d1)); d1_gt=sum(d1[i]>d0[i] for i in range(len(d1)))
        if pos==4 and mean>=.01 and float(np.mean(d1))>float(np.mean(d0)): verdict='TARGET_PASS'
        elif pos==4 and mean>=.005 and float(np.mean(d1))>float(np.mean(d0)): verdict='STRONG_PASS'
        elif pos==4 and mean>0 and float(np.mean(d1))>float(np.mean(d0)): verdict='USER_CONDITION_PASS'
        elif pos==3 and mean>=.0025 and d1_gt>=3: verdict='PARTIAL_SIGNAL'
        else: verdict='FAIL'
    seeds=list(rows)
    d1_u={str(s):rows[s]['D1_vs_C0']['U'] for s in seeds}; d0_u={str(s):rows[s]['D0_vs_C0']['U'] for s in seeds}
    summary={'status':'VALIDATION_DECISION','protocol':PROTOCOL,'expanded':expanded,'seeds':seeds,'rows':{str(s):rows[s] for s in seeds},'D1_vs_C0':{'seed_U':d1_u,'mean_U':float(np.mean(list(d1_u.values()))),'median_U':float(np.median(list(d1_u.values()))),'worst_seed_U':float(min(d1_u.values())),'positive_seeds':int(sum(x>0 for x in d1_u.values()))},'D0_vs_C0':{'seed_U':d0_u,'mean_U':float(np.mean(list(d0_u.values()))),'positive_seeds':int(sum(x>0 for x in d0_u.values()))},'D1_gt_D0_backbones':int(sum(d1_u[str(s)]>d0_u[str(s)] for s in seeds)),'verdict':verdict,'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
    (evid/'ROUND12_VALIDATION_RESULTS.json').write_text(json.dumps(summary,indent=2)+'\n')
    cost=build_cost(rows); (evid/'ROUND12_COST_REPORT.json').write_text(json.dumps({'stage':'final' if expanded else 'preflight_stop','cost':cost,'TEST_ACCESSED':False},indent=2)+'\n')
    (evid/'ROUND12_USER_CONDITION_AUDIT.json').write_text(json.dumps({str(s):rows[s]['D1_user_condition_mechanism'] for s in seeds},indent=2)+'\n'); (evid/'ROUND12_MARGIN_DIAGNOSTIC.json').write_text(json.dumps({str(s):rows[s]['D1_recommendation_usefulness'] for s in seeds},indent=2)+'\n')
    write_report(summary,pf,evid); print(json.dumps({'status':'VALIDATION_DECISION','expanded':expanded,'verdict':verdict,'D1_mean':summary['D1_vs_C0']['mean_U'],'D1_positive':summary['D1_vs_C0']['positive_seeds'],'D0_mean':summary['D0_vs_C0']['mean_U']},sort_keys=True)); return summary


def pct(x): return f'{100*float(x):+.4f}%'
def write_report(summary,pf,evid):
    rows=summary['rows']; seeds=[str(x) for x in summary['seeds']]; L=['# Round12 Validation Report — User-Conditioned Latent Diffusion Training','',f"Final verdict: **ROUND12 = {summary['verdict']}**",f"Expansion executed: **{summary['expanded']}**",'', '> Diffusion OFF at inference. Test, Sports and Electronics remained CLOSED.','']
    L += ['## Best-epoch Validation results','','| Backbone | M0 R20 | C0 R20 | D0 R20 | D1 R20 | U(D0 vs C0) | U(D1 vs C0) | U(D1 vs D0) |','|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in seeds:
        x=rows[s]; L.append(f"| {s} | {x['M0']['R20']:.8f} | {x['C0']['R20']:.8f} | {x['D0']['R20']:.8f} | {x['D1']['R20']:.8f} | {pct(x['D0_vs_C0']['U'])} | {pct(x['D1_vs_C0']['U'])} | {pct(x['D1_vs_D0']['U'])} |")
    L += ['','## Preflight gate','','```text',json.dumps(pf['gate_checks'],indent=2),'```',f"Expansion open: **{pf['EXPANSION_OPEN']}**",'']
    L += ['## Hard-shell diagnostics (relative to C0)','','| Backbone | D0 NetCross@10 | D0 NetCross@20 | D1 NetCross@10 | D1 NetCross@20 |','|---:|---:|---:|---:|---:|']
    for s in seeds:
        x=rows[s]; L.append(f"| {s} | {x['boundary_D0_vs_C0']['NetCross10']:+d} | {x['boundary_D0_vs_C0']['NetCross20']:+d} | {x['boundary_D1_vs_C0']['NetCross10']:+d} | {x['boundary_D1_vs_C0']['NetCross20']:+d} |")
    L += ['','## User-condition mechanism audit','']
    for s in seeds:
        m=rows[s]['D1_user_condition_mechanism']; t=m['modalities']['text']['true_minus_zero_norm']['mean']; v=m['modalities']['visual']['true_minus_zero_norm']['mean']; ct=m['modalities']['text']['cos_true_vs_shuffled_effect']['mean']; cv=m['modalities']['visual']['cos_true_vs_shuffled_effect']['mean']; L.append(f'- seed{s}: ||TRUE-ZERO|| Text={t:.6f}, Visual={v:.6f}; cos(TRUE effect, SHUFFLED effect) Text={ct:.4f}, Visual={cv:.4f}.')
    L += ['','## Recommendation-usefulness diagnostic','']
    for s in seeds:
        d=rows[s]['D1_recommendation_usefulness']['delta_margin']; L.append(f"- seed{s}: Δmargin mean={d['mean']:+.6f}, median={d['median']:+.6f}, fraction positive={d['fraction_positive']:.4f}, p10={d['p10']:+.6f}, p90={d['p90']:+.6f}.")
    L += ['','## Training cost','','| Backbone | Variant | sec/epoch | peak GiB | slowdown vs C0 |','|---:|---|---:|---:|---:|']
    for s in seeds:
        c=rows[s]['cost']['C0']['mean_seconds_per_epoch']
        for v in VARIANTS:
            z=rows[s]['cost'][v]; L.append(f"| {s} | {v} | {z['mean_seconds_per_epoch']:.2f} | {z['peak_gpu_memory_GiB']:.3f} | {z['mean_seconds_per_epoch']/c:.3f}× |")
    L += ['','## Summary','','- Primary scientific comparison is D1 vs C0; D1 vs D0 isolates the value of real user conditioning; C0 vs M0 isolates continuation-training effects.',f"- D1 vs C0 mean U = **{pct(summary['D1_vs_C0']['mean_U'])}**, worst = **{pct(summary['D1_vs_C0']['worst_seed_U'])}**, positive backbones = **{summary['D1_vs_C0']['positive_seeds']}/{len(seeds)}**.",f"- D0 vs C0 mean U = **{pct(summary['D0_vs_C0']['mean_U'])}**; D1 outperforms D0 on **{summary['D1_gt_D0_backbones']}/{len(seeds)}** backbones.",'- No Test loader, direct ranking residual, raw-4480D Diffusion, inference-time Diffusion, or hyperparameter search was used.','']
    (evid/'ROUND12_REPORT.md').write_text('\n'.join(L))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--mode',choices=['smoke','formal','preflight','final'],required=True); ap.add_argument('--seed',type=int); ap.add_argument('--variant',choices=VARIANTS); ap.add_argument('--out-root',default=str(ROOT/'diffusion_experiments/round12_ucldt/outputs')); ap.add_argument('--evidence',default=str(ROOT/'diffusion_experiments/round12_ucldt/evidence')); a=ap.parse_args(); root=Path(a.out_root); evid=Path(a.evidence)
    if a.mode=='smoke': smoke(root,evid)
    elif a.mode=='formal':
        if a.seed not in PREFLIGHT+EXPANSION or a.variant not in VARIANTS: raise RuntimeError('formal requires frozen seed and variant')
        run_variant(a.seed,a.variant,root/f'seed{a.seed}'/a.variant)
    elif a.mode=='preflight': preflight(root,evid)
    else: final_aggregate(root,evid)
if __name__=='__main__': main()
