from __future__ import annotations
import argparse,json,math,random,sys,time
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'src'))
from diffusion_experiments.models.round8_bounded_residual import BoundedResidualDDPM,cosine_alpha_bar,residual_start,ddim_reverse_differentiable,decode_residual
from diffusion_experiments.modules.round8_common import cfg_round8,sha,state_hash,load_interactions,train_frame,unique_histories,git_sha

def sample_negs(users,hsets,observed,rng):
 out=np.empty(len(users),np.int64)
 for j,u in enumerate(users):
  while True:
   x=int(observed[rng.integers(0,len(observed))])
   if x not in hsets[int(u)]: out[j]=x; break
 return out

def score_torch(delta,item,center,q,R):
 dn=torch.linalg.vector_norm(delta,dim=1); s=torch.maximum(q,R*dn); return (delta*(item-center)).sum(1)/s.clamp_min(1e-12),s

def grad_vec(loss,params): return torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
def grad_norm(gs): return math.sqrt(sum(float(g.detach().square().sum()) for g in gs if g is not None))
def grad_cos(a,b):
 dot=sum(float((x.detach()*y.detach()).sum()) for x,y in zip(a,b) if x is not None and y is not None); return dot/max(grad_norm(a)*grad_norm(b),1e-12)

def save_resume(p,model,opt,step,event_rng,neg_rng,tgen,history,diag):
 torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'step':step,'event_rng':event_rng.bit_generator.state,'neg_rng':neg_rng.bit_generator.state,'tgen':tgen.get_state(),'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'python_rng':random.getstate(),'history':history,'diag':diag},p)

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--backbone-seed',type=int,required=True); ap.add_argument('--diffusion-seed',type=int,required=True); ap.add_argument('--assets',required=True); ap.add_argument('--out',required=True); ap.add_argument('--mode',choices=['pilot','formal'],required=True); ap.add_argument('--resume',action='store_true'); a=ap.parse_args()
 out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
 if not a.resume and any(out.iterdir()): raise RuntimeError(f'refuse overwrite nonempty {out}')
 cfg=cfg_round8(); dc=cfg['diffusion']; updates=int(dc['pilot_updates'] if a.mode=='pilot' else dc['total_updates']); warm=int(dc['warmup_updates']); device=torch.device('cuda:0'); assets=Path(a.assets); audit=json.loads((assets/'audit.json').read_text())
 if int(audit['backbone_seed'])!=a.backbone_seed: raise RuntimeError('asset identity mismatch')
 ev=np.load(assets/'events.npz'); dep=np.load(assets/'deployment.npz'); users_all=ev['users'].astype(np.int64); pos_all=ev['pos'].astype(np.int64); valid_all=ev['valid'].astype(bool); y_all=ev['y_target'].astype(np.float32); cond_all=ev['cond'].astype(np.float32); h_all=ev['history_raw'].astype(np.float32); b_all=ev['budget'].astype(np.float32); m_all=ev['m'].astype(np.float32); R_all=ev['R'].astype(np.float32); q_all=ev['q'].astype(np.float32); item_raw_np=dep['collab_item'].astype(np.float32); observed=dep['observed_items'].astype(np.int64)
 train=train_frame(load_interactions()); histories=unique_histories(train,int(dep['users'].shape[0])); hsets=[set(map(int,h)) for h in histories]
 torch.manual_seed(int(a.diffusion_seed)); torch.cuda.manual_seed_all(int(a.diffusion_seed)); np.random.seed(int(a.diffusion_seed)%2**32); random.seed(int(a.diffusion_seed))
 model=BoundedResidualDDPM(hidden_dim=int(dc['hidden_dim']),time_dim=int(dc['time_dim']),dropout=float(dc['dropout']),hidden_layers=int(dc['hidden_layers']),radius=float(dc['radius'])).to(device); params=[p for p in model.parameters() if p.requires_grad]; initial_hash=state_hash(model); opt=torch.optim.AdamW(params,lr=float(dc['lr']),weight_decay=float(dc['weight_decay']))
 alpha=cosine_alpha_bar(int(dc['steps']),float(dc['cosine_s']),device=device); path=[int(x) for x in dc['ddim_path']]; t_edit=int(dc['t_edit']); pref_w=float(dc['pref_weight']); temp=float(dc['pref_temperature']); radius=float(dc['radius']); bs=int(dc['batch_size']); diag_every=int(dc['grad_diag_every']); ck_every=int(dc['checkpoint_every'])
 event_rng=np.random.default_rng(int(a.diffusion_seed)+100003*int(a.backbone_seed)); neg_rng=np.random.default_rng(int(a.diffusion_seed)+200003*int(a.backbone_seed)); tgen=torch.Generator(device=device); tgen.manual_seed(int(a.diffusion_seed)+300007*int(a.backbone_seed)); history=[]; diagnostics=[]; start=1; latest=out/'resume_latest.pt'
 if a.resume:
  if not latest.exists(): raise RuntimeError('--resume requested but checkpoint missing')
  ck=torch.load(latest,map_location='cpu',weights_only=False); model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer']); start=int(ck['step'])+1
  event_rng.bit_generator.state=ck['event_rng']; neg_rng.bit_generator.state=ck['neg_rng']; tgen.set_state(ck['tgen']); torch.set_rng_state(ck['cpu_rng']); torch.cuda.set_rng_state_all(ck['cuda_rng']); random.setstate(ck['python_rng']); history=ck['history']; diagnostics=ck['diag']
 item_raw=torch.as_tensor(item_raw_np,device=device); illegal=0; nonfinite=0; t0=time.time(); torch.cuda.reset_peak_memory_stats(device)
 for step in range(start,updates+1):
  ix=event_rng.integers(0,len(users_all),size=bs); ix=ix[valid_all[ix]]
  if len(ix)==0: continue
  u_np=users_all[ix]; p_np=pos_all[ix]; n_np=sample_negs(u_np,hsets,observed,neg_rng); illegal+=sum(int(n) in hsets[int(u)] for u,n in zip(u_np,n_np))
  p=torch.as_tensor(p_np,dtype=torch.long,device=device); n=torch.as_tensor(n_np,dtype=torch.long,device=device)
  target=torch.as_tensor(y_all[ix],device=device); cond=torch.as_tensor(cond_all[ix],device=device); h=torch.as_tensor(h_all[ix],device=device); budget=torch.as_tensor(b_all[ix],device=device); center=torch.as_tensor(m_all[ix],device=device); RR=torch.as_tensor(R_all[ix],device=device); qq=torch.as_tensor(q_all[ix],device=device)
  t=torch.randint(1,int(dc['steps'])+1,(len(ix),),generator=tgen,device=device); eps=torch.randn((len(ix),64),generator=tgen,device=device); at=alpha[t].unsqueeze(1); signal=at.sqrt()*target; noise=(1-at).sqrt()*eps; yt=signal+noise
  model.train(); pred,_=model.forward_with_raw(yt,t,cond); den=((pred-target)**2).mean()
  epsa=torch.randn((len(ix),64),generator=tgen,device=device); start_y=residual_start(epsa,alpha,t_edit); was=model.training; model.eval(); Y,last_raw=ddim_reverse_differentiable(model,start_y,cond,alpha,path,return_last_raw=True); model.train(was)
  path_loss=((Y-target)**2).mean(); delta=decode_residual(Y,budget,radius); rp,_=score_torch(delta,item_raw[p],center,qq,RR); rn,_=score_torch(delta,item_raw[n],center,qq,RR); pref=F.softplus((rn-rp)/temp).mean(); total=den+path_loss+(pref_w*pref if step>warm else 0.0)
  if not torch.isfinite(total): nonfinite+=1; raise RuntimeError(f'nonfinite total at step {step}')
  if step%diag_every==0:
   gd=grad_vec(den,params); gp=grad_vec(path_loss,params); gf=grad_vec(pref,params); yn=torch.linalg.vector_norm(Y,dim=1); dn=torch.linalg.vector_norm(delta,dim=1); hn=torch.linalg.vector_norm(h,dim=1); gn=torch.linalg.vector_norm(h+delta,dim=1); ratio=dn/budget; gratio=gn/hn.clamp_min(1e-12); maxscore=torch.maximum(rp.abs(),rn.abs()).max()
   if yn.max()>radius+1e-4 or ratio.max()>1.0001 or gratio.max()>1.5001 or maxscore>1.0001: raise RuntimeError(f'NUMERICAL_INVALID bound at step {step}')
   diagnostics.append({'step':step,'den_grad_norm':grad_norm(gd),'path_grad_norm':grad_norm(gp),'pref_grad_norm':grad_norm(gf),'den_path_cos':grad_cos(gd,gp),'path_pref_cos':grad_cos(gp,gf),'weighted_path_grad_norm':grad_norm(gp),'weighted_pref_grad_norm':pref_w*grad_norm(gf),'pre_v_norm_mean':float(torch.linalg.vector_norm(last_raw,dim=1).mean()),'pre_v_norm_max':float(torch.linalg.vector_norm(last_raw,dim=1).max()),'Y_norm_mean':float(yn.mean()),'Y_norm_max':float(yn.max()),'delta_over_b_mean':float(ratio.mean()),'delta_over_b_max':float(ratio.max()),'g_over_h_mean':float(gratio.mean()),'g_over_h_max':float(gratio.max()),'r_abs_max':float(maxscore),'margin_rp_minus_rn':float((rp-rn).mean()),'pref_temperature':temp})
  opt.zero_grad(set_to_none=True); total.backward(); rawgn=float(torch.nn.utils.clip_grad_norm_(params,float(dc['grad_clip'])).detach()); opt.step()
  if step==1 or step%100==0 or step==updates:
   history.append({'step':step,'den_loss':float(den.detach()),'path_loss':float(path_loss.detach()),'pref_loss':float(pref.detach()),'total_loss':float(total.detach()),'raw_grad_norm_before_clip':rawgn,'phase':'warmup' if step<=warm else 'joint','signal_rms':float(torch.sqrt((signal**2).mean())),'noise_rms':float(torch.sqrt((noise**2).mean()))})
  if step%ck_every==0 and step<updates: save_resume(latest,model,opt,step,event_rng,neg_rng,tgen,history,diagnostics)
 final=out/'generator.pt'; torch.save({'model':model.state_dict(),'backbone_seed':a.backbone_seed,'diffusion_seed':a.diffusion_seed,'config':dc,'initial_hash':initial_hash,'final_hash':state_hash(model),'assets_audit_sha256':sha(assets/'audit.json')},final)
 if latest.exists(): latest.unlink()
 result={'status':'COMPLETE','protocol_version':cfg['protocol_version'],'mode':a.mode,'backbone_seed':a.backbone_seed,'diffusion_seed':a.diffusion_seed,'updates':updates,'warmup_updates':warm,'parameter_count':sum(p.numel() for p in params),'initial_hash':initial_hash,'final_hash':state_hash(model),'generator_sha256':sha(final),'illegal_negative_count':illegal,'nonfinite_count':nonfinite,'history':history,'gradient_diagnostics':diagnostics,'peak_cuda_allocated_gib':torch.cuda.max_memory_allocated(device)/(1024**3),'elapsed_seconds':time.time()-t0,'git_sha':git_sha(),'access':{'TRAIN':True,'VALIDATION_LABELS_USED':False,'TEST_LABELS_USED':False}}
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'status':'COMPLETE','mode':a.mode,'cell':f'{a.backbone_seed}x{a.diffusion_seed}','last':history[-1],'diag_last':diagnostics[-1] if diagnostics else None,'peak_gib':result['peak_cuda_allocated_gib'],'sha':result['generator_sha256']},sort_keys=True))
if __name__=='__main__': main()
