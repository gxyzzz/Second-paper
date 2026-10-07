from __future__ import annotations
import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from diffusion_experiments.models.round5_behavior_diffusion import BehaviorDiffusion,cosine_alpha_bar,ddim_trajectory
from diffusion_experiments.modules.round5_common import cfg_round5,sha

def model_hash(m):
    h=hashlib.sha256()
    for k,v in sorted(m.state_dict().items()): h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def event_seed(u,p,seed):
    b=f'baby|{int(u)}|{int(p)}|{int(seed)}'.encode(); return int.from_bytes(hashlib.sha256(b).digest()[:8],'little') & ((1<<63)-1)
def event_noise(users,pos,seed,d):
    out=np.empty((len(users),d),np.float32)
    for i,(u,p) in enumerate(zip(users,pos)): out[i]=np.random.default_rng(event_seed(u,p,seed)).standard_normal(d).astype(np.float32)
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--assets',required=True); ap.add_argument('--out',required=True); ap.add_argument('--seed',type=int,required=True); ap.add_argument('--mode',choices=['smoke','formal'],default='formal'); a=ap.parse_args(); out=Path(a.out)
    if out.exists() and any(out.iterdir()): raise RuntimeError('refuse overwrite nonempty output')
    out.mkdir(parents=True,exist_ok=True); cfg=cfg_round5(); dc=cfg['behavior_diffusion']; ev=np.load(Path(a.assets)/'events.npz'); x0=ev['x0'].astype(np.float32); cond=ev['condition'].astype(np.float32); users=ev['users'].astype(np.int64); pos=ev['pos_items'].astype(np.int64)
    if not torch.cuda.is_available() or '5090' not in torch.cuda.get_device_name(0): raise RuntimeError('RTX5090 required')
    if a.seed not in [int(x) for x in cfg['formal_seeds']]: raise RuntimeError('unregistered seed')
    device=torch.device('cuda:0'); torch.manual_seed(a.seed); np.random.seed(a.seed & 0xffffffff); model=BehaviorDiffusion(x0.shape[1],cond.shape[1],int(dc['hidden_dim']),int(dc['time_dim']),float(dc['dropout']),int(dc['hidden_layers'])).to(device); initial_hash=model_hash(model); opt=torch.optim.AdamW(model.parameters(),lr=float(dc['lr']),weight_decay=float(dc['weight_decay'])); alpha=cosine_alpha_bar(int(dc['steps']),float(dc['cosine_s']),device=device)
    updates=30 if a.mode=='smoke' else int(dc['updates']); bs=min(int(dc['batch_size']),len(x0)); rng=np.random.default_rng(a.seed); hist=[]; bucket_sum=np.zeros(5,np.float64); bucket_n=np.zeros(5,np.int64); t0=time.time(); model.train()
    for step in range(1,updates+1):
        ix=rng.integers(0,len(x0),bs); t=rng.integers(1,int(dc['steps'])+1,bs); eps=rng.standard_normal((bs,x0.shape[1])).astype(np.float32); xb=torch.as_tensor(x0[ix],device=device); cb=torch.as_tensor(cond[ix],device=device); tt=torch.as_tensor(t,device=device,dtype=torch.long); ee=torch.as_tensor(eps,device=device); at=alpha[tt].unsqueeze(1); xt=at.sqrt()*xb+(1-at).sqrt()*ee; pred=model(xt,tt,cb); loss=((pred-xb)**2).mean(); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),float(dc['grad_clip'])); opt.step()
        with torch.no_grad():
            per=((pred-xb)**2).mean(1).detach().cpu().numpy(); bi=np.minimum((t-1)//10,4)
            for b in range(5):
                q=bi==b; bucket_sum[b]+=float(per[q].sum()); bucket_n[b]+=int(q.sum())
        if step==1 or step%100==0 or step==updates: hist.append({'step':step,'loss':float(loss.detach().cpu())})
    model.eval(); final_hash=model_hash(model)
    # fixed-noise condition sensitivity diagnostic
    take=np.arange(min(512,len(x0))); tt=torch.full((len(take),),25,device=device,dtype=torch.long); noise=torch.as_tensor(event_noise(users[take],pos[take],a.seed+991, x0.shape[1]),device=device); cb=torch.as_tensor(cond[take],device=device); at=alpha[tt].unsqueeze(1); xt=at.sqrt()*torch.as_tensor(x0[take],device=device)+(1-at).sqrt()*noise
    with torch.no_grad(): y=model(xt,tt,cb); yp=model(xt,tt,cb[torch.arange(len(take)-1,-1,-1,device=device)]); cond_delta=float((y-yp).abs().mean().cpu())
    # Generate one deterministic DDIM trajectory per FIT event; noise is keyed by (dataset,u,p,seed).
    states={int(s):np.empty((len(x0),x0.shape[1]),np.float32) for s in dc['save_states']}; final=np.empty_like(x0); gen_norm=[]; gen_cos=[]; gen_start=time.time(); path=[int(x) for x in dc['ddim_path']]; save_states=set(int(x) for x in dc['save_states'])
    for st in range(0,len(x0),512):
        en=min(st+512,len(x0)); nz=event_noise(users[st:en],pos[st:en],a.seed,x0.shape[1]); nb=torch.as_tensor(nz,device=device); cb=torch.as_tensor(cond[st:en],device=device); xf,sv=ddim_trajectory(model,nb,cb,alpha,path,save_states); final[st:en]=xf.cpu().numpy().astype(np.float32)
        for s in save_states: states[s][st:en]=sv[s].cpu().numpy().astype(np.float32)
    if not np.isfinite(final).all() or any(not np.isfinite(v).all() for v in states.values()): raise RuntimeError('non-finite DDIM trajectory')
    # Replay first events in reverse query order to verify batch/query order invariance.
    replay_ix=np.arange(min(64,len(x0)))[::-1]; nz=event_noise(users[replay_ix],pos[replay_ix],a.seed,x0.shape[1]); _,sv=ddim_trajectory(model,torch.as_tensor(nz,device=device),torch.as_tensor(cond[replay_ix],device=device),alpha,path,save_states); replay=max(float(np.max(np.abs(sv[s].cpu().numpy()-states[s][replay_ix]))) for s in save_states)
    for s in save_states:
        q=states[s]; gen_norm.append({'state':s,'mean_norm':float(np.linalg.norm(q,axis=1).mean()),'std_norm':float(np.linalg.norm(q,axis=1).std())}); den=np.maximum(np.linalg.norm(q,axis=1)*np.linalg.norm(x0,axis=1),1e-12); gen_cos.append({'state':s,'mean_cos_to_target':float(np.mean(np.sum(q*x0,axis=1)/den))})
    np.savez_compressed(out/'trajectory_states.npz',users=users.astype(np.int32),pos_items=pos.astype(np.int32),**{f'x_{s}':states[s] for s in sorted(save_states)})
    ck={'model':model.state_dict(),'seed':a.seed,'x_dim':x0.shape[1],'cond_dim':cond.shape[1],'config':dc,'alpha_bar':alpha.detach().cpu(),'initial_hash':initial_hash,'final_hash':final_hash}; torch.save(ck,out/'generator.pt')
    snr=[]
    for t in [1,10,20,30,40,50]:
        aa=float(alpha[t].cpu()); snr.append({'t':t,'signal_rms_scale':float(aa**.5),'noise_rms_scale':float((1-aa)**.5),'snr':float(aa/max(1-aa,1e-12))})
    result={'status':'COMPLETE','mode':a.mode,'seed':a.seed,'updates':updates,'initial_hash':initial_hash,'final_hash':final_hash,'parameter_count':sum(p.numel() for p in model.parameters()),'history':hist,'t_bucket_noisy_x0_mse':[float(bucket_sum[i]/max(bucket_n[i],1)) for i in range(5)],'snr':snr,'condition_sensitivity_mean_abs_delta':cond_delta,'trajectory_norms':gen_norm,'trajectory_cosine_to_target':gen_cos,'replay_max_abs_diff':replay,'train_seconds':gen_start-t0,'trajectory_seconds':time.time()-gen_start,'generator_sha256':sha(out/'generator.pt'),'trajectory_sha256':sha(out/'trajectory_states.npz'),'assets_events_sha256':sha(Path(a.assets)/'events.npz'),'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'gpu_name':torch.cuda.get_device_name(0),'access':{'DEV_ACCESSED':False,'INTERNAL_ACCESSED':False,'CONFIRM_ACCESSED':False,'TEST_ACCESSED':False}}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({'status':'COMPLETE','seed':a.seed,'updates':updates,'last_loss':hist[-1]['loss'],'cond_delta':cond_delta,'replay':replay,'hash':final_hash},sort_keys=True))
if __name__=='__main__': main()
