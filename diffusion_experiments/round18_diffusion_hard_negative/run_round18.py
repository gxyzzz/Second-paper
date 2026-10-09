from __future__ import annotations
import argparse,gc,json,shutil,time
from pathlib import Path
import numpy as np
import torch
from diffusion_experiments.round18_diffusion_hard_negative.round18_core import *

EVID=RDIR/'evidence'; ASSET=RDIR/'assets'; OUT=RDIR/'outputs'

def writej(name,x): EVID.mkdir(parents=True,exist_ok=True);(EVID/name).write_text(json.dumps(x,indent=2)+'\n')
def model_hash(model):
 h=hashlib.sha256()
 for k,v in sorted(model.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()

def prepare_seed(seed):
 model,ck,ds,tr,a=load_start(seed,True); reps=frozen_representations(model); rp=save_frozen_repr(seed,model,reps);h=HistoryData(seed);ev=ChronoEvents(h)
 data={'seed':seed,'checkpoint':a['checkpoint'],'checkpoint_epoch':int(ck['epoch']),'checkpoint_sha256':sha256_file(a['checkpoint']),
       'n_users':h.n_users,'n_items':h.n_items,'clean_users':int(len(h.clean_users)),'chronological_train_events':int(len(ev.train[0])),'chronological_audit_events':int(len(ev.audit[0])),
       'diffusion_train_users':int(len(ev.train_users)),'diffusion_audit_users':int(len(ev.audit_users)),'user_split_overlap':int(len(ev.train_users&ev.audit_users)),
       'history_len':HIST_LEN,'target_self_inclusion_clean':int(sum(int(t) in set(h.histories[int(u)][:-1]) for u,t in zip(h.clean_users,h.targets))),
       'x0_definition':'starting-backbone frozen final_item 64D','x0_shape':[64],'id_token_shape':list(reps['id_item'].shape),'visual_token_shape':list(reps['visual_item'].shape),'text_token_shape':list(reps['text_item'].shape),
       'frozen_repr_path':str(rp),'TEST_ACCESSED':False}
 del model;torch.cuda.empty_cache();return data

def preflight():
 data={'protocol':PROTOCOL,'seeds':{},'TEST_ACCESSED':False};band={'protocol':PROTOCOL,'seeds':{},'TEST_ACCESSED':False};smoke={'protocol':PROTOCOL,'seeds':{},'TEST_ACCESSED':False}
 for seed in SEEDS:
  d=prepare_seed(seed);data['seeds'][str(seed)]=d;h=HistoryData(seed);ba={}
  for ep in range(1,5):
   cand,ranks,scores,valid=legal_band_rows(h,ep);el=valid.any(1);viol=0
   for r in np.where(el)[0]:
    obs=set(h.histories[int(h.clean_users[r])][:-1]);tar=int(h.targets[r]);ids=cand[r,valid[r]];viol+=sum(int(x)==tar or int(x) in obs or int(x)<0 or int(x)>=h.n_items for x in ids)
   ba[str(ep)]={'band':list(BANDS[ep]),'coverage':float(el.mean()),'eligible':int(el.sum()),'violations':int(viol),'LOW_BAND_COVERAGE':bool(el.mean()<.9),'PASS':bool(el.mean()>=.9 and viol==0)}
  band['seeds'][str(seed)]=ba
  model,ck,ds,tr,a=load_start(seed,True);reps=load_repr(seed,model.device);hd=HistoryData(seed);events=ChronoEvents(hd);net=HCDDiffuser(64).to(model.device);sched=DiffusionSchedule(model.device)
  u,p,tg=[x[:32] for x in events.train];net.zero_grad(set_to_none=True);loss,pred,x0,xt,tt=generator_loss(net,sched,reps,hd,u,p,tg,model.device);loss.backward();ng=float(sum(float(q.grad.square().sum()) for q in net.parameters() if q.grad is not None)**.5);mg=all(q.grad is None for q in model.parameters())
  # reverse smoke + real negative + original MSCA one-batch FT smoke
  with torch.no_grad(): st=reverse_stages(net,sched,reps,hd,hd.clean_users[:32],device=model.device,batch=32)
  cand,ranks,scores,valid=legal_band_rows(hd,1);q=st[8];neg,_=select_from_band(cand[:32],valid[:32],q,reps['final_item'].cpu().numpy());real=bool(np.all((neg>=0)&(neg<hd.n_items)))
  del model;torch.cuda.empty_cache();model2,ck2,ds2,tr2,a2=load_start(seed,False);opt=torch.optim.Adam(model2.parameters(),lr=float(ck2['config']['learning_rate'])*FT_LR_SCALE,weight_decay=float(ck2['config']['weight_decay']));uu=torch.as_tensor(hd.clean_users[:32],device=model2.device);pp=torch.as_tensor(hd.targets[:32],device=model2.device);nn=torch.as_tensor(neg,device=model2.device);opt.zero_grad(set_to_none=True);fl=model2.calculate_loss([uu,pp,nn]);fl.backward();mgn=float(sum(float(q.grad.square().sum()) for q in model2.parameters() if q.grad is not None)**.5)
  smoke['seeds'][str(seed)]={'generator_loss':float(loss),'generator_grad_norm':ng,'MSCA_grads_none_during_generator':mg,'x0_shape':list(x0.shape),'pred_x0_shape':list(pred.shape),'forward_loss_finite':bool(torch.isfinite(loss)),'reverse_stage_shapes':{str(k):list(v.shape) for k,v in st.items()},'real_item_negative':real,'negative_not_target':bool(np.all(neg!=hd.targets[:32])),'negative_rank_band':[26,30],'MSCA_finetune_loss':float(fl.detach()),'MSCA_grad_norm':mgn,'generator_loaded_during_finetune':False,'diffusion_inference_calls_during_evaluation':0,'PASS':bool(torch.isfinite(loss) and ng>0 and mg and list(x0.shape)==[32,64] and list(pred.shape)==[32,64] and real and np.all(neg!=hd.targets[:32]) and torch.isfinite(fl) and mgn>0)}
  del model2,net,opt;torch.cuda.empty_cache();gc.collect()
 data['PASS']=all(x['user_split_overlap']==0 and x['target_self_inclusion_clean']==0 for x in data['seeds'].values());band['PASS']=all(v['PASS'] for s in band['seeds'].values() for v in s.values());smoke['PASS']=all(x['PASS'] for x in smoke['seeds'].values()) and data['PASS'] and band['PASS']
 writej('ROUND18_DATA_AUDIT.json',data);writej('ROUND18_CANDIDATE_BAND_AUDIT.json',band);writej('ROUND18_PREFLIGHT_AUDIT.json',smoke);print(json.dumps({'DATA':data['PASS'],'BAND':band['PASS'],'SMOKE':smoke['PASS']}))

def train_generator(seed):
 model,ck,ds,tr,a=load_start(seed,True); hd=HistoryData(seed); events=ChronoEvents(hd); reps=load_repr(seed,model.device); net=HCDDiffuser(64).to(model.device);sched=DiffusionSchedule(model.device);opt=torch.optim.Adam(net.parameters(),lr=GEN_LR,weight_decay=0.0);logs=[]
 for ep in range(1,GEN_EPOCHS+1):
  seed_all(202618000+seed*10+ep); order=np.random.default_rng(202618000+seed*10+ep).permutation(len(events.train[0]));net.train();tot=0.;nb=0;gn=0.
  for u,p,tg in batch_events(events.train,order,512):
   opt.zero_grad(set_to_none=True);loss,pr,x0,xt,t=generator_loss(net,sched,reps,hd,u,p,tg,model.device);loss.backward();gn=max(gn,float(sum(float(q.grad.square().sum()) for q in net.parameters() if q.grad is not None)**.5));opt.step();tot+=float(loss.detach());nb+=1
  rec={'epoch':ep,'loss':tot/nb,'batches':nb,'max_grad_norm':gn};logs.append(rec);print(json.dumps({'stage':'GEN','seed':seed,**rec}),flush=True)
 quality=generator_quality(net,sched,reps,hd,events.audit,model.device);cp=ASSET/f'seed{seed}_generator_epoch10.pt';torch.save({'protocol':PROTOCOL,'seed':seed,'epoch':10,'state':{k:v.detach().cpu() for k,v in net.state_dict().items()},'TEST_ACCESSED':False},cp)
 out={'protocol':PROTOCOL,'seed':seed,'epochs':10,'selected_epoch':10,'recommendation_validation_used_for_generator_selection':False,'optimizer':{'name':'Adam','lr':GEN_LR,'weight_decay':0.0},'train_logs':logs,'quality':quality,'generator_checkpoint':str(cp),'MSCA_frozen':True,'CoLift_frozen':True,'TEST_ACCESSED':False}
 writej(f'ROUND18_DIFFUSION_TRAIN_AUDIT_SEED{seed}.json',out);writej(f'ROUND18_DIFFUSION_GENERATOR_QUALITY_SEED{seed}.json',quality)
 del model,net,opt;torch.cuda.empty_cache();gc.collect();return out

def load_generator(seed,device):
 p=torch.load(ASSET/f'seed{seed}_generator_epoch10.pt',map_location='cpu',weights_only=False);net=HCDDiffuser(64).to(device);net.load_state_dict(p['state'],strict=True);net.eval();
 for q in net.parameters():q.requires_grad=False
 return net

def make_maps(seed):
 model,ck,ds,tr,a=load_start(seed,True);hd=HistoryData(seed);reps=load_repr(seed,model.device);net=load_generator(seed,model.device);sched=DiffusionSchedule(model.device)
 if not json.loads((EVID/f'ROUND18_DIFFUSION_GENERATOR_QUALITY_SEED{seed}.json').read_text())['PASS']:raise RuntimeError('Gate DG failed; negative mining forbidden')
 # true and deterministic shuffled histories. Shuffle only condition, keep target rows fixed.
 rng=np.random.default_rng(20261891); perm=rng.permutation(hd.clean_users); shuf_src=perm
 true=reverse_stages(net,sched,reps,hd,hd.clean_users,device=model.device,batch=256);shuf=reverse_stages(net,sched,reps,hd,hd.clean_users,shuf_src,device=model.device,batch=256)
 np.savez_compressed(ASSET/f'seed{seed}_reverse_queries.npz',users=hd.clean_users,Q25=true[8],Q50=true[16],Q75=true[24],Q100=true[32],SHUF_Q25=shuf[8],SHUF_Q50=shuf[16],SHUF_Q75=shuf[24],SHUF_Q100=shuf[32],shuffled_source_users=shuf_src)
 maps,ba,ov,hi=build_negative_maps(seed,true,shuf);audit={'seed':seed,'maps':{},'candidate_bands':ba,'F3_F4_overlap':ov,'history_shuffle':hi,'TEST_ACCESSED':False}
 for v in ('F1','F2','F3','F4'):
  p=ASSET/f'seed{seed}_{v}_negative_map.npz';z=np.load(p);viol_target=int(np.sum(z['negative']==z['positive']));viol_rank=0
  if v!='F1':
   for ep in range(1,5):
    lo,hh=BANDS[ep];m=z['epoch']==ep;viol_rank+=int(np.sum((z['frozen_rank'][m]<lo)|(z['frozen_rank'][m]>hh)))
  # explicit prefix legality
  bad=0
  for u,n in zip(z['user'],z['negative']): bad+=int(int(n) in set(hd.histories[int(u)][:-1]))
  audit['maps'][v]={'path':str(p),'sha256':sha256_file(p),'rows':int(len(z['user'])),'target_violations':viol_target,'prefix_violations':bad,'rank_band_violations':viol_rank,'real_catalog_ids':bool(np.all((z['negative']>=0)&(z['negative']<hd.n_items))),'PASS':bool(viol_target==0 and bad==0 and viol_rank==0 and np.all((z['negative']>=0)&(z['negative']<hd.n_items)))}
 writej(f'ROUND18_NEGATIVE_MAP_AUDIT_SEED{seed}.json',audit);writej(f'ROUND18_HISTORY_SHUFFLE_DIAGNOSTIC_SEED{seed}.json',{'seed':seed,'epochs':hi,'TEST_ACCESSED':False});writej(f'ROUND18_DIFFUSION_TRAJECTORY_AUDIT_SEED{seed}.json',{'seed':seed,'F3_F4_overlap':ov,'candidate_bands':ba,'F2':[x['audit'] for x in maps['F2']],'F3':[x['audit'] for x in maps['F3']],'F4':[x['audit'] for x in maps['F4']],'TEST_ACCESSED':False})
 del model,net;torch.cuda.empty_cache();gc.collect();return audit

def save_ft_checkpoint(path,model,ck,seed,variant,ep):
 torch.save({'config':ck['config'],'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'epoch':int(ck['epoch'])+ep,'round18_epoch':ep,'seed':seed,'variant':variant,'TEST_ACCESSED':False},path)

def load_map_epoch(seed,variant,ep):
 z=np.load(ASSET/f'seed{seed}_{variant}_negative_map.npz');m=z['epoch']==ep
 return z['user'][m].astype(np.int64),z['positive'][m].astype(np.int64),z['negative'][m].astype(np.int64),z['frozen_rank'][m].astype(np.int64)

def run_f0(seed,ev):
 model,ck,ds,tr,a=load_start(seed,False);h=model_hash(model);val=ev.evaluate(model)
 out={'protocol':PROTOCOL,'seed':seed,'variant':'F0','start_model_hash':h,'checkpoint':a['checkpoint'],'checkpoint_epoch':int(ck['epoch']),'backbone_metrics':val['backbone'],'colift_metrics':val['colift'],'diffusion_inference_calls':val['diffusion_inference_calls'],'TEST_ACCESSED':False}
 writej(f'ROUND18_F0_SEED{seed}.json',out);del model;torch.cuda.empty_cache();gc.collect();return out

def finetune_variant(seed,variant,ev,f0):
 model,ck,ds,tr,a=load_start(seed,False);h0=model_hash(model)
 if h0!=f0['start_model_hash']:raise RuntimeError('starting checkpoint hash mismatch')
 lr=float(ck['config']['learning_rate'])*FT_LR_SCALE;wd=float(ck['config']['weight_decay']);bs=int(ck['config']['train_batch_size']);opt=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=wd)
 outdir=OUT/f'seed{seed}/{variant}';outdir.mkdir(parents=True,exist_ok=True);logs=[]
 for ep in range(1,FT_EPOCHS+1):
  U,P,N,R=load_map_epoch(seed,variant,ep);order=np.random.default_rng(20261850+seed*10+ep).permutation(len(U));seed_all(20261850+seed*10+ep);model.train();tot=0.;nb=0;gn=0.
  for s in range(0,len(order),bs):
   ix=order[s:s+bs];uu=torch.as_tensor(U[ix],device=model.device);pp=torch.as_tensor(P[ix],device=model.device);nn=torch.as_tensor(N[ix],device=model.device)
   opt.zero_grad(set_to_none=True);loss=model.calculate_loss([uu,pp,nn]);loss.backward();gn=max(gn,float(sum(float(q.grad.square().sum()) for q in model.parameters() if q.grad is not None)**.5));opt.step();tot+=float(loss.detach());nb+=1
  save_ft_checkpoint(outdir/f'epoch{ep}.pt',model,ck,seed,variant,ep);val=ev.evaluate(model)
  rec={'epoch':ep,'train_loss':tot/max(nb,1),'batches':nb,'max_grad_norm':gn,'examples':int(len(U)),'negative_map_sha256':sha256_file(ASSET/f'seed{seed}_{variant}_negative_map.npz'),'backbone_metrics':val['backbone'],'colift_metrics':val['colift'],'backbone_vs_F0':delta_pack(val['backbone'],f0['backbone_metrics']),'colift_vs_F0':delta_pack(val['colift'],f0['colift_metrics']),'diffusion_inference_calls':val['diffusion_inference_calls']};logs.append(rec)
  print(json.dumps({'stage':'FT','seed':seed,'variant':variant,'epoch':ep,'loss':rec['train_loss'],'U_back':rec['backbone_vs_F0']['U'],'U_colift':rec['colift_vs_F0']['U'],'R20':val['colift']['R20']},sort_keys=True),flush=True)
 final=logs[-1];res={'protocol':PROTOCOL,'seed':seed,'variant':variant,'starting_model_hash':h0,'final_epoch':4,'best_epoch_selection_used':False,'optimizer':{'name':'Adam','lr':lr,'weight_decay':wd},'generator_loaded_during_finetune':False,'epoch_logs':logs,'final_backbone_metrics':final['backbone_metrics'],'final_colift_metrics':final['colift_metrics'],'final_backbone_vs_F0':final['backbone_vs_F0'],'final_colift_vs_F0':final['colift_vs_F0'],'diffusion_inference_calls':max(x['diffusion_inference_calls'] for x in logs),'TEST_ACCESSED':False}
 (outdir/'result.json').write_text(json.dumps(res,indent=2)+'\n');writej(f'ROUND18_{variant}_SEED{seed}.json',res);del model,opt;torch.cuda.empty_cache();gc.collect();return res

def formal_seed(seed):
 ev=ValidationEvaluator(seed);f0=run_f0(seed,ev);results={'F0':f0}
 for v in ('F1','F2','F3','F4'):results[v]=finetune_variant(seed,v,ev,f0)
 return results

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mode',required=True,choices=['preflight','generator','maps','formal']);ap.add_argument('--seed',type=int,choices=[999,1000]);a=ap.parse_args()
 if a.mode=='preflight':preflight()
 elif a.mode=='generator':train_generator(a.seed)
 elif a.mode=='maps':make_maps(a.seed)
 elif a.mode=='formal':formal_seed(a.seed)
if __name__=='__main__':main()
