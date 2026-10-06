from __future__ import annotations
import argparse, hashlib, io, json, math, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch, yaml
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'src'))
from diffusion_experiments.models.round3_candidate_energy import DenoiserCore,ConditionalDenoiser,DirectEvidenceScorer,block_loss,q_sample,probe_specs,candidate_energy,evidence_advantage,boundary_gate,apply_evidence
from diffusion_experiments.models.round1_residual import cosine_alpha_bars
from diffusion_experiments.scripts.round1_build_assets import dev_labels
from modules.ranking import metrics_at,sha256_file
PRIMARY=('R10','N10','R20','N20'); ALL=PRIMARY+('R50','N50')
def sha(p): return sha256_file(Path(p))
def model_hash(model):
 b=io.BytesIO(); torch.save(model.state_dict(),b); return hashlib.sha256(b.getvalue()).hexdigest()
def result_vs(base,new):
 rel={k:(float(new[k])-float(base[k]))/max(float(base[k]),1e-12) for k in PRIMARY}; return {'metrics':{k:float(new[k]) for k in ALL},'relative_primary':rel,'U':float(np.mean(list(rel.values()))),'delta_R50':float(new['R50']-base['R50']),'delta_N50':float(new['N50']-base['N50'])}
def protected(res,c):
 r=res['relative_primary']; return bool(sum(v>=0 for v in r.values())>=int(c['min_nonnegative_primary']) and min(r.values())>=-float(c['max_primary_relative_regression']) and res['delta_R50']>=float(c['min_absolute_delta_R50']) and res['delta_N50']>=float(c['min_absolute_delta_N50']))
def probe_labels(users,targets): return {int(u):{int(t)} for u,t in zip(users,targets)}
def batches(n,batch,rng):
 p=rng.permutation(n)
 for s in range(0,n,batch): yield p[s:s+batch]
def finite(name,x):
 if not torch.isfinite(x).all(): raise FloatingPointError(f'nonfinite {name}')
def grad_norm(loss,params):
 gs=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True); return float(math.sqrt(sum(float((g.detach()**2).sum()) for g in gs if g is not None)))
def block_components(pred,target,dims):
 out=[]; s=0
 for d in dims:
  d=int(d); out.append((pred[:,s:s+d]-target[:,s:s+d]).pow(2).mean()); s+=d
 return out
def load_bg(run_dir,latent_dim,hidden,time_dim,dropout,device):
 r=json.loads((Path(run_dir)/'result.json').read_text()); ck=torch.load(Path(run_dir)/'last.pt',map_location=device,weights_only=False)
 m=DenoiserCore(latent_dim,hidden,time_dim,dropout).to(device); m.load_state_dict(ck['model']); m.eval()
 for p in m.parameters(): p.requires_grad_(False)
 return m,r

def make_dataset(which,frozen,context):
 pz=np.load(frozen/'probe_top100.npz'); tz=np.load(frozen/'probe_targets.npz'); dz=np.load(frozen/'dev_top100.npz')
 if which=='dev':
  users=dz['users'].astype(np.int64); items=dz['items'].astype(np.int32); s0=dz['s0'].astype(np.float32); labels=dev_labels(ROOT/'data/baby/baby.inter',users); targets=None
 else:
  mask=tz['internal'].astype(bool) if which=='internal' else tz['reranker_train'].astype(bool)
  users=pz['users'][mask].astype(np.int64); items=pz['items'][mask].astype(np.int32); s0=pz['s0'][mask].astype(np.float32); targets=tz['target_items'][mask].astype(np.int32); labels=probe_labels(users,targets)
 return {'users':users,'items':items,'s0':s0,'labels':labels,'targets':targets,'context':context[users]}

def evidence_C(model,data,latent,device,batch=512):
 out=[]; model.eval()
 with torch.no_grad():
  for st in range(0,len(data['users']),batch):
   en=min(st+batch,len(data['users'])); ids=data['items'][st:en,5:30]; z=torch.as_tensor(latent[ids],device=device); c=torch.as_tensor(data['context'][st:en],device=device); B,K,D=z.shape
   out.append(model(z.reshape(B*K,D),c[:,None,:].expand(-1,K,-1).reshape(B*K,-1)).reshape(B,K).cpu().numpy())
 return np.concatenate(out).astype(np.float32)

def evidence_D(model,bg,data,latent,dims,scale,cfg,manifest,bundle,mode,device,batch=256,spec_override=None,context_override=None):
 specs=spec_override or probe_specs(mode,manifest['schedule'],manifest['ae_t_index_zero_based']); aa=[]; cc=[]; bb=[]; model.eval(); bg.eval()
 with torch.no_grad():
  for st in range(0,len(data['users']),batch):
   en=min(st+batch,len(data['users'])); users=data['users'][st:en]; ids=data['items'][st:en,5:30]; z=torch.as_tensor(latent[ids],device=device); ca=data['context'][st:en] if context_override is None else context_override[st:en]; c=torch.as_tensor(ca,device=device)
   a,ec,eb=evidence_advantage(model,bg,z,c,users,dims,scale,cfg['dataset'],bundle,specs,int(cfg['noise']['diffusion_steps'])); aa.append(a.cpu().numpy()); cc.append(ec.cpu().numpy()); bb.append(eb.cpu().numpy())
 return np.concatenate(aa).astype(np.float32),np.concatenate(cc).astype(np.float32),np.concatenate(bb).astype(np.float32)

def eval_evidence(data,evidence,cfg,gate_override=None):
 base=metrics_at(data['items'],data['users'],data['labels']); gate=boundary_gate(data['s0']) if gate_override is None else gate_override; out={}
 for eta in cfg['formal']['eta_candidates']:
  ranked,delta=apply_evidence(data['items'],data['s0'],evidence,float(eta),float(cfg['formal']['clip_c']),gate); r=result_vs(base,metrics_at(ranked,data['users'],data['labels'])); r['protected']=protected(r,cfg['protection']); r['changed_rows']=int(np.any(ranked!=data['items'],axis=1).sum()); r['nonzero_delta']=int((np.abs(delta)>1e-12).sum()); out[str(eta)]=r
 return {'baseline':{k:float(base[k]) for k in ALL},'etas':out}
def choose(ev):
 cand=[(float(k),v) for k,v in ev['etas'].items() if v['protected']]; return max(cand,key=lambda x:x[1]['U']) if cand else (0.0,ev['etas']['0.0'])
def background_scale(model,mode,cfg,manifest,frozen,latent,dims,device):
 pz=np.load(frozen/'probe_top100.npz'); tz=np.load(frozen/'probe_targets.npz'); mask=tz['reranker_train'].astype(bool); users=pz['users'][mask].astype(np.int64); items=pz['items'][mask,5:30].astype(np.int32); specs=probe_specs(mode,manifest['schedule'],manifest['ae_t_index_zero_based']); vals=[]
 model.eval()
 with torch.no_grad():
  for st in range(0,len(users),256):
   en=min(st+256,len(users)); z=torch.as_tensor(latent[items[st:en]],device=device); e=candidate_energy(model,z,None,users[st:en],dims,cfg['dataset'],int(cfg['probe_bundle_main']),specs,int(cfg['noise']['diffusion_steps']),bg=True); vals.append(e.cpu().numpy())
 e=np.concatenate(vals).astype(np.float32); centered=e-e.mean(1,keepdims=True); raw=float(centered.std()); return max(raw,1e-3),{'raw_centered_std':raw,'scale_floor':1e-3,'mean_energy':float(e.mean()),'std_energy':float(e.std()),'queries':int(len(users)),'candidates_per_query':25}

def train_background(kind,out,cfg,manifest,latent,dims,device,mode_limit=None):
 mc=cfg['model']; seed=int(cfg['background_seed']); torch.manual_seed(seed); np.random.seed(seed); model=DenoiserCore(latent.shape[1],int(mc['hidden_dim']),int(mc['time_dim']),float(mc['dropout'])).to(device); opt=torch.optim.AdamW(model.parameters(),lr=float(mc['learning_rate']),weight_decay=float(mc['weight_decay'])); gen=torch.Generator(device=device); gen.manual_seed(seed*1009+11); hist=[]; updates=0; start=time.time(); torch.cuda.reset_peak_memory_stats(device); T=int(cfg['noise']['diffusion_steps']); ae_t=int(manifest['ae_t_index_zero_based'])
 epochs=2 if mode_limit=='smoke' else int(mc['epochs']); catalog=np.arange(len(latent))
 if mode_limit=='smoke': catalog=np.sort(np.random.default_rng(2026100604).choice(catalog,size=min(512,len(catalog)),replace=False))
 for ep in range(1,epochs+1):
  model.train(); rng=np.random.default_rng(seed+ep); acc=0.; nacc=0; blocks=np.zeros(len(dims),np.float64)
  for jj in batches(len(catalog),int(mc['batch_size']),rng):
   idx=catalog[jj]; clean=torch.as_tensor(latent[idx],device=device); n=len(idx)
   if kind=='BG_DM': t=torch.randint(0,T,(n,),generator=gen,device=device)
   else: t=torch.full((n,),ae_t,device=device,dtype=torch.long)
   noise=torch.randn(clean.shape,generator=gen,device=device); pred=model(q_sample(clean,t,noise,T),t); ls=block_components(pred,clean,dims); loss=torch.stack(ls).mean(); finite('bg_loss',loss); opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); updates+=1
   acc+=float(loss.detach())*n; nacc+=n
   for j,x in enumerate(ls): blocks[j]+=float(x.detach())*n
  rec={'epoch':ep,'loss':acc/nacc,'block_mse':[float(x/nacc) for x in blocks]}; hist.append(rec); (out/'history.json').write_text(json.dumps(hist,indent=2)+'\n')
 model.eval(); scale,sd=background_scale(model,'DM' if kind=='BG_DM' else 'AE',cfg,manifest,ROOT/cfg['frozen_assets_dir'],latent,dims,device)
 ck={'kind':kind,'model':model.state_dict(),'epoch':epochs,'scale_bg':scale,'seed':seed}; torch.save(ck,out/'last.pt')
 result={'status':'COMPLETE','kind':kind,'seed':seed,'epochs':epochs,'parameter_count':sum(p.numel() for p in model.parameters()),'optimizer_steps':updates,'train_seconds':time.time()-start,'peak_cuda_memory_bytes':int(torch.cuda.max_memory_allocated(device)),'scale_bg':scale,'scale_diagnostic':sd,'last_loss':hist[-1],'model_hash':model_hash(model),'access':{'CONFIRM_ACCESSED':False,'TEST_ACCESSED':False}}
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n'); return result
def pref_loss_energy(model,bg,zpos,zneg,mask,ctx,users,dims,scale,cfg,mode,pref_gen,manifest):
 B,D=zpos.shape; K=zneg.shape[1]; T=int(cfg['noise']['diffusion_steps'])
 if mode=='DM':
  choices=torch.as_tensor([int(x['t_index_zero_based']) for x in manifest['schedule']],device=zpos.device); sel=torch.randint(0,len(choices),(B,),generator=pref_gen,device=zpos.device); tq=choices[sel]
 else:
  tq=torch.full((B,),int(manifest['ae_t_index_zero_based']),device=zpos.device,dtype=torch.long)
 noise=torch.randn((B,D),generator=pref_gen,device=zpos.device); zall=torch.cat([zpos[:,None,:],zneg],1); N=K+1; tflat=tq[:,None].expand(-1,N).reshape(-1); nflat=noise[:,None,:].expand(-1,N,-1).reshape(-1,D); clean=zall.reshape(-1,D); xt=q_sample(clean,tflat,nflat,T); cflat=ctx[:,None,:].expand(-1,N,-1).reshape(-1,ctx.shape[-1]); cp=model(xt,tflat,cflat); ec=block_loss(cp,clean,dims).reshape(B,N)
 with torch.no_grad(): bp=bg(xt,tflat); eb=block_loss(bp,clean,dims).reshape(B,N)
 A=(eb-ec)/float(scale); apos=A[:,0:1]; an=A[:,1:]; per=F.softplus(an-apos); mf=mask.float(); lp=(per*mf).sum(1)/mf.sum(1).clamp_min(1); margin=(apos.squeeze(1)-(an*mf).sum(1)/mf.sum(1).clamp_min(1))
 return lp.mean(),{'margin':float(margin.detach().mean()),'A_pos':float(apos.detach().mean()),'A_neg':float((an.detach()*mf).sum()/mf.sum().clamp_min(1)),'Econd_pos':float(ec[:,0].detach().mean()),'Econd_neg':float((ec[:,1:].detach()*mf).sum()/mf.sum().clamp_min(1)),'Ebg_pos':float(eb[:,0].detach().mean()),'Ebg_neg':float((eb[:,1:].detach()*mf).sum()/mf.sum().clamp_min(1))}

def compute_evidence(kind,model,bg,data,latent,dims,scale,cfg,manifest,bundle,device,context_override=None,spec_override=None):
 if kind=='C': return evidence_C(model,data,latent,device),None,None
 return evidence_D(model,bg,data,latent,dims,scale,cfg,manifest,bundle,'AE' if kind=='AE_PREF' else 'DM',device,spec_override=spec_override,context_override=context_override)

def train_conditional(kind,seed,out,cfg,manifest,asset_dir,frozen,latent,context,dims,sup,device,bg_dir=None,mode_limit=None):
 mc=cfg['model']; torch.manual_seed(int(seed)); np.random.seed(int(seed)); bg=None; bgr=None; scale=None
 if kind=='C': model=DirectEvidenceScorer(latent.shape[1],context.shape[1],int(mc['hidden_dim']),float(mc['dropout'])).to(device)
 else:
  bg,bgr=load_bg(bg_dir,latent.shape[1],int(mc['hidden_dim']),int(mc['time_dim']),float(mc['dropout']),device); scale=float(bgr['scale_bg']); model=ConditionalDenoiser(latent.shape[1],context.shape[1],int(mc['hidden_dim']),int(mc['time_dim']),float(mc['dropout'])).to(device).init_from_background(bg,int(seed)+41)
 init_hash=model_hash(model); params=[p for p in model.parameters() if p.requires_grad]; opt=torch.optim.AdamW(params,lr=float(mc['learning_rate']),weight_decay=float(mc['weight_decay'])); recon_gen=torch.Generator(device=device); recon_gen.manual_seed(int(seed)*1009+13); pref_gen=torch.Generator(device=device); pref_gen.manual_seed(int(seed)*1013+29)
 users=sup['users'].astype(np.int64); pos=sup['positive_items'].astype(np.int64); neg=sup['negative_items'].astype(np.int64); nmask=sup['negative_mask'].astype(bool); train_idx=np.arange(len(users)); smoke_seed=2026100604
 if mode_limit=='smoke': train_idx=np.sort(np.random.default_rng(smoke_seed).choice(train_idx,size=min(256,len(train_idx)),replace=False))
 dev=make_dataset('dev',frozen,context); internal=make_dataset('internal',frozen,context)
 if mode_limit=='smoke':
  rr=np.random.default_rng(smoke_seed); di=np.sort(rr.choice(len(dev['users']),size=512,replace=False)); ii=np.sort(rr.choice(len(internal['users']),size=512,replace=False))
  for data,sel in [(dev,di),(internal,ii)]:
   for k in ['users','items','s0','context']:
    data[k]=data[k][sel]
   if data['targets'] is not None: data['targets']=data['targets'][sel]; data['labels']=probe_labels(data['users'],data['targets'])
   else: data['labels']=dev_labels(ROOT/'data/baby/baby.inter',data['users'])
 epochs=2 if mode_limit=='smoke' else int(mc['epochs']); eval_every=1 if mode_limit=='smoke' else int(cfg['formal']['eval_every']); min_epochs=1 if mode_limit=='smoke' else int(cfg['formal']['min_epochs']); patience=int(cfg['formal']['patience_evals']); T=int(cfg['noise']['diffusion_steps']); ae_t=int(manifest['ae_t_index_zero_based']); hist=[]; best=None; bad=0; updates=0; start=time.time(); torch.cuda.reset_peak_memory_stats(device); ckdir=out/'checkpoints'; ckdir.mkdir(parents=True,exist_ok=True)
 for ep in range(1,epochs+1):
  model.train(); rng=np.random.default_rng(int(seed)+ep); sums={'loss':0.,'den':0.,'pref':0.,'reg':0.,'n':0}; bsum=np.zeros(len(dims),np.float64); diag_sum={}; first_grad=None
  for rows in batches(len(train_idx),int(mc['batch_size']),rng):
   ix=train_idx[rows]; u=users[ix]; zpos=torch.as_tensor(latent[pos[ix]],device=device); ctx=torch.as_tensor(context[u],device=device); mask=torch.as_tensor(nmask[ix],device=device); zneg=torch.as_tensor(latent[np.maximum(neg[ix],0)],device=device); n=len(ix)
   if kind=='C':
    qpos=model(zpos,ctx); B,K,D=zneg.shape; qneg=model(zneg.reshape(B*K,D),ctx[:,None,:].expand(-1,K,-1).reshape(B*K,-1)).reshape(B,K); mf=mask.float(); lp=((F.softplus(qneg-qpos[:,None])*mf).sum(1)/mf.sum(1).clamp_min(1)).mean(); reg=float(mc['direct_l2'])*(qpos.pow(2).mean()+((qneg.pow(2)*mf).sum()/mf.sum().clamp_min(1))); lden=torch.zeros((),device=device); loss=lp+reg; ldiag={}
   else:
    if kind=='AE_PREF': t=torch.full((n,),ae_t,device=device,dtype=torch.long)
    else: t=torch.randint(0,T,(n,),generator=recon_gen,device=device)
    noise=torch.randn(zpos.shape,generator=recon_gen,device=device); pred=model(q_sample(zpos,t,noise,T),t,ctx); comps=block_components(pred,zpos,dims); lden=torch.stack(comps).mean(); lp=torch.zeros((),device=device); reg=torch.zeros((),device=device); ldiag={}
    if kind in ('D_PREF','AE_PREF'): lp,ldiag=pref_loss_energy(model,bg,zpos,zneg,mask,ctx,u,dims,scale,cfg,'AE' if kind=='AE_PREF' else 'DM',pref_gen,manifest)
    loss=lden+float(mc['lambda_pref'])*lp
   finite('conditional_loss',loss)
   if first_grad is None:
    first_grad={'total':grad_norm(loss,params)}
    if kind!='C': first_grad['den']=grad_norm(lden,params)
    if kind in ('C','D_PREF','AE_PREF'): first_grad['pref']=grad_norm(lp,params)
    if kind in ('D_PREF','AE_PREF') and (not np.isfinite(first_grad['pref']) or first_grad['pref']<=0): raise RuntimeError('preference gradient zero/nonfinite')
   opt.zero_grad(set_to_none=True); loss.backward()
   for p in params:
    if p.grad is not None and not torch.isfinite(p.grad).all(): raise FloatingPointError('nonfinite conditional gradient')
   opt.step(); updates+=1; sums['loss']+=float(loss.detach())*n; sums['den']+=float(lden.detach())*n; sums['pref']+=float(lp.detach())*n; sums['reg']+=float(reg.detach())*n; sums['n']+=n
   if kind!='C':
    for j,x in enumerate(comps): bsum[j]+=float(x.detach())*n
   for k,v in ldiag.items(): diag_sum[k]=diag_sum.get(k,0.)+float(v)*n
  rec={'epoch':ep,'train':{k:(v/max(sums['n'],1) if k!='n' else int(v)) for k,v in sums.items()},'optimizer_steps_total':updates,'first_batch_grad_norm':first_grad}
  if kind!='C': rec['train']['block_mse']=[float(x/max(sums['n'],1)) for x in bsum]
  if diag_sum: rec['preference_diagnostic']={k:v/max(sums['n'],1) for k,v in diag_sum.items()}
  if ep%eval_every==0:
   de,_,_=compute_evidence(kind,model,bg,dev,latent,dims,scale,cfg,manifest,int(cfg['probe_bundle_main']),device); ie,_,_=compute_evidence(kind,model,bg,internal,latent,dims,scale,cfg,manifest,int(cfg['probe_bundle_main']),device); dr=eval_evidence(dev,de,cfg); ir=eval_evidence(internal,ie,cfg); eta,dsel=choose(dr); rec['dev']=dr; rec['dev']['chosen_eta']=eta; rec['dev']['chosen_U']=dsel['U']; rec['internal_at_dev_eta']=ir['etas'][str(eta)]
   state={'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch':ep,'seed':int(seed),'kind':kind,'recon_rng':recon_gen.get_state(),'pref_rng':pref_gen.get_state()}; torch.save(state,ckdir/f'epoch_{ep:02d}.pt')
   if ep>=min_epochs:
    score=float(dsel['U'])
    if best is None or score>best['U']+1e-12:
     best={'epoch':ep,'eta':float(eta),'U':score,'dev_result':dsel,'internal_result':ir['etas'][str(eta)]}; torch.save(state,out/'best.pt'); bad=0
    else: bad+=1
  hist.append(rec); (out/'history.json').write_text(json.dumps(hist,indent=2)+'\n')
  if ep>=min_epochs and bad>=patience: break
 if best is None: raise RuntimeError('no checkpoint selected')
 return finish_conditional(kind,seed,out,cfg,manifest,frozen,latent,context,dims,sup,device,bg,bgr,scale,model,best,init_hash,hist,updates,start)
def subset_data(data,mask):
 idx=np.flatnonzero(mask); d={k:(v[idx] if isinstance(v,np.ndarray) and len(v)==len(mask) else v) for k,v in data.items()};
 if d.get('targets') is not None: d['labels']=probe_labels(d['users'],d['targets'])
 else: d['labels']=dev_labels(ROOT/'data/baby/baby.inter',d['users'])
 return d

def final_selected(data,evidence,eta,cfg,gate=None):
 base=metrics_at(data['items'],data['users'],data['labels']); g=boundary_gate(data['s0']) if gate is None else gate; ranked,delta=apply_evidence(data['items'],data['s0'],evidence,float(eta),float(cfg['formal']['clip_c']),g); r=result_vs(base,metrics_at(ranked,data['users'],data['labels'])); r['protected']=protected(r,cfg['protection']); r['changed_rows']=int(np.any(ranked!=data['items'],axis=1).sum()); r['nonzero_delta']=int((np.abs(delta)>1e-12).sum()); return r,ranked,delta

def finish_conditional(kind,seed,out,cfg,manifest,frozen,latent,context,dims,sup,device,bg,bgr,scale,model,best,init_hash,hist,updates,start):
 ck=torch.load(out/'best.pt',map_location=device,weights_only=False); model.load_state_dict(ck['model']); model.eval(); eta=float(best['eta']); train=make_dataset('train',frozen,context); dev=make_dataset('dev',frozen,context); internal=make_dataset('internal',frozen,context)
 main=int(cfg['probe_bundle_main']); secondary=int(cfg['probe_bundle_secondary']); t0=time.perf_counter(); de,dec,deb=compute_evidence(kind,model,bg,dev,latent,dims,scale,cfg,manifest,main,device); infer=time.perf_counter()-t0; ie,iec,ieb=compute_evidence(kind,model,bg,internal,latent,dims,scale,cfg,manifest,main,device); te,tec,teb=compute_evidence(kind,model,bg,train,latent,dims,scale,cfg,manifest,main,device)
 dmain=eval_evidence(dev,de,cfg); imain=eval_evidence(internal,ie,cfg); tmain=eval_evidence(train,te,cfg); ds,_,_=compute_evidence(kind,model,bg,dev,latent,dims,scale,cfg,manifest,secondary,device); ins,_,_=compute_evidence(kind,model,bg,internal,latent,dims,scale,cfg,manifest,secondary,device)
 sec_dev,_,_=final_selected(dev,ds,eta,cfg); sec_int,_,_=final_selected(internal,ins,eta,cfg); sel_dev,drank,ddelta=final_selected(dev,de,eta,cfg); sel_int,irank,idelta=final_selected(internal,ie,eta,cfg); sel_train,trank,tdelta=final_selected(train,te,eta,cfg); g1=np.ones_like(de,np.float32); g1dev,_,_=final_selected(dev,de,eta,cfg,g1)
 # fixed original window subsets
 trw=(sup['positive_natural_rank']>=6)&(sup['positive_natural_rank']<=30); tz=np.load(frozen/'probe_targets.npz'); iw=(tz['target_rank'][tz['internal'].astype(bool)]>=6)&(tz['target_rank'][tz['internal'].astype(bool)]<=30)
 train856=subset_data(train,trw); int189=subset_data(internal,iw); rtrain856,_,_=final_selected(train856,te[trw],eta,cfg); rint189,_,_=final_selected(int189,ie[iw],eta,cfg)
 # invariants
 zero,_=apply_evidence(dev['items'],dev['s0'],de,0.0,float(cfg['formal']['clip_c']),boundary_gate(dev['s0']))
 if not np.array_equal(zero,dev['items']) or not np.array_equal(drank[:,:5],dev['items'][:,:5]) or not np.array_equal(drank[:,30:],dev['items'][:,30:]): raise RuntimeError('deployment invariant failed')
 for a,b in zip(dev['items'],drank):
  if set(a.tolist())!=set(b.tolist()): raise RuntimeError('candidate set changed')
 if bg is not None and model_hash(bg)!=bgr['model_hash']: raise RuntimeError('frozen background changed')
 np.savez_compressed(out/'predictions.npz',dev_users=dev['users'],dev_base=dev['items'],dev_new=drank,dev_evidence_main=de,dev_evidence_secondary=ds,dev_delta=ddelta,internal_users=internal['users'],internal_targets=internal['targets'],internal_base=internal['items'],internal_new=irank,internal_evidence_main=ie,internal_evidence_secondary=ins,internal_delta=idelta,train_users=train['users'],train_targets=train['targets'],train_base=train['items'],train_new=trank,train_evidence_main=te,train_delta=tdelta)
 result={'status':'COMPLETE','kind':kind,'seed':int(seed),'initial_model_hash':init_hash,'parameter_count':sum(p.numel() for p in model.parameters()),'background_dir':str(bgr and bgr.get('kind')),'background_scale':scale,'best':best,'final_dev_main':dmain,'final_internal_main':imain,'final_train_main':tmain,'selected_dev_main':sel_dev,'selected_internal_main':sel_int,'selected_train_main':sel_train,'selected_train_window856':rtrain856,'selected_internal_window189':rint189,'secondary_bundle_at_frozen_eta':{'dev':sec_dev,'internal':sec_int},'g1_control_at_frozen_eta':g1dev,'optimizer_steps':updates,'epochs_completed':len(hist),'train_seconds':time.time()-start,'main_bundle_dev_energy_seconds':infer,'peak_cuda_memory_bytes':int(torch.cuda.max_memory_allocated(device)),'model_hash':model_hash(model),'access':{'CONFIRM_ACCESSED':False,'TEST_ACCESSED':False}}
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n'); return result
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--assets',required=True); ap.add_argument('--kind',choices=['BG_DM','BG_AE','C','D_GEN','D_PREF','AE_PREF'],required=True); ap.add_argument('--seed',type=int); ap.add_argument('--background'); ap.add_argument('--out',required=True); ap.add_argument('--mode',choices=['smoke','formal'],default='formal'); a=ap.parse_args()
 out=Path(a.out);
 if out.exists() and any(out.iterdir()): raise RuntimeError('refuse overwrite nonempty run')
 out.mkdir(parents=True,exist_ok=True); asset=Path(a.assets); cfg=yaml.safe_load((ROOT/'diffusion_experiments/configs/round3_baby.yaml').read_text()); manifest=json.loads((asset/'manifest.json').read_text())
 if manifest['access']['CONFIRM_ACCESSED'] or manifest['access']['TEST_ACCESSED'] or cfg['access']['confirm_open'] or cfg['access']['test_open']: raise RuntimeError('closed-set invariant')
 for n,h in manifest['artifacts'].items():
  if sha(asset/n)!=h: raise RuntimeError(f'asset hash mismatch {n}')
 if not torch.cuda.is_available() or '5090' not in torch.cuda.get_device_name(0): raise RuntimeError('RTX5090 required')
 device=torch.device('cuda:0'); lc=np.load(asset/'latent_context.npz'); sup=np.load(asset/'supervision.npz'); latent=lc['item_latent'].astype(np.float32); context=lc['user_context'].astype(np.float32); dims=lc['block_dims'].astype(np.int64).tolist(); frozen=ROOT/cfg['frozen_assets_dir']
 if a.kind.startswith('BG_'): res=train_background(a.kind,out,cfg,manifest,latent,dims,device,a.mode if a.mode=='smoke' else None)
 else:
  if a.seed is None: raise RuntimeError('conditional seed required')
  if a.mode=='formal' and int(a.seed) not in [int(x) for x in cfg['conditional_seeds']]: raise RuntimeError('unregistered seed')
  bgdir=None if a.kind=='C' else Path(a.background or '')
  if a.kind!='C' and (not bgdir or not (bgdir/'result.json').exists()): raise RuntimeError('background run required')
  res=train_conditional(a.kind,int(a.seed),out,cfg,manifest,asset,frozen,latent,context,dims,sup,device,bgdir,a.mode if a.mode=='smoke' else None)
 # bind runtime identity after successful scientific output
 p=out/'result.json'; res=json.loads(p.read_text()); res.update({'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'tracked_diff_names':subprocess.check_output(['git','diff','--name-only'],cwd=ROOT,text=True).splitlines(),'config_sha256':sha(ROOT/'diffusion_experiments/configs/round3_baby.yaml'),'asset_manifest_sha256':sha(asset/'manifest.json'),'asset_artifacts':manifest['artifacts'],'mode':a.mode,'gpu_name':torch.cuda.get_device_name(0),'gpu_uuid':subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()[0].strip()}); p.write_text(json.dumps(res,indent=2)+'\n'); print(json.dumps({'status':res['status'],'kind':a.kind,'seed':a.seed,'best':res.get('best'),'scale_bg':res.get('scale_bg')},sort_keys=True))
if __name__=='__main__': main()
