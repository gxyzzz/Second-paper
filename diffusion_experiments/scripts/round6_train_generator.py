from __future__ import annotations
import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path
import numpy as np, torch
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from diffusion_experiments.models.round6_user_behavior_diffusion import UserBehaviorDiffusion,cosine_alpha_bar
from diffusion_experiments.modules.round6_common import cfg_round6,sha

def model_hash(m):
    h=hashlib.sha256()
    for k,v in sorted(m.state_dict().items()): h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--assets',required=True); ap.add_argument('--out',required=True); ap.add_argument('--seed',type=int,required=True); ap.add_argument('--mode',choices=['smoke','formal'],default='formal'); a=ap.parse_args(); out=Path(a.out)
    if out.exists() and any(out.iterdir()): raise RuntimeError('refuse overwrite nonempty output')
    out.mkdir(parents=True,exist_ok=True); cfg=cfg_round6(); dc=cfg['behavior_diffusion']; ev=np.load(Path(a.assets)/'events.npz'); x0=ev['x0'].astype(np.float32); cond=ev['condition'].astype(np.float32)
    if x0.shape[1]!=64 or cond.shape[1]!=129: raise RuntimeError('Round6 dimensions invalid')
    if a.seed not in [int(x) for x in cfg['generator_seeds']]: raise RuntimeError('unregistered generator seed')
    if not torch.cuda.is_available() or '5090' not in torch.cuda.get_device_name(0): raise RuntimeError('RTX5090 required')
    device=torch.device('cuda:0'); torch.manual_seed(a.seed); np.random.seed(a.seed&0xffffffff); model=UserBehaviorDiffusion(64,129,int(dc['hidden_dim']),int(dc['time_dim']),float(dc['dropout']),int(dc['hidden_layers'])).to(device); init_hash=model_hash(model); opt=torch.optim.AdamW(model.parameters(),lr=float(dc['lr']),weight_decay=float(dc['weight_decay'])); alpha=cosine_alpha_bar(int(dc['steps']),float(dc['cosine_s']),device=device)
    updates=30 if a.mode=='smoke' else int(dc['updates']); bs=int(dc['batch_size']); rng=np.random.default_rng(a.seed); hist=[]; bsum=np.zeros(5,np.float64); bn=np.zeros(5,np.int64); sig2=np.zeros(5,np.float64); noi2=np.zeros(5,np.float64); t0=time.time(); model.train()
    for step in range(1,updates+1):
        ix=rng.integers(0,len(x0),bs); tt=rng.integers(1,int(dc['steps'])+1,bs); eps=rng.standard_normal((bs,64)).astype(np.float32)
        xb=torch.as_tensor(x0[ix],device=device); cb=torch.as_tensor(cond[ix],device=device); t=torch.as_tensor(tt,device=device,dtype=torch.long); ee=torch.as_tensor(eps,device=device); at=alpha[t].unsqueeze(1)
        signal=at.sqrt()*xb; noise=(1-at).sqrt()*ee; pred=model(signal+noise,t,cb); per=((pred-xb)**2).mean(1); loss=per.mean(); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),float(dc['grad_clip'])); opt.step(); bi=np.minimum((tt-1)//10,4)
        with torch.no_grad():
            pp=per.detach().cpu().numpy(); ss=(signal**2).mean(1).detach().cpu().numpy(); nn=(noise**2).mean(1).detach().cpu().numpy()
            for b in range(5):
                q=bi==b; bsum[b]+=pp[q].sum(); bn[b]+=q.sum(); sig2[b]+=ss[q].sum(); noi2[b]+=nn[q].sum()
        if step==1 or step%100==0 or step==updates: hist.append({'step':step,'loss':float(loss.detach().cpu())})
    model.eval(); torch.save({'model':model.state_dict(),'seed':a.seed,'x_dim':64,'cond_dim':129,'config':dc,'initial_hash':init_hash,'final_hash':model_hash(model)},out/'generator.pt')
    buckets=[{'t_range':[b*10+1,(b+1)*10],'mse':float(bsum[b]/max(bn[b],1)),'signal_rms':float(np.sqrt(sig2[b]/max(bn[b],1))),'noise_rms':float(np.sqrt(noi2[b]/max(bn[b],1))),'empirical_snr':float(sig2[b]/max(noi2[b],1e-12))} for b in range(5)]
    result={'status':'COMPLETE','mode':a.mode,'seed':a.seed,'updates':updates,'parameter_count':sum(p.numel() for p in model.parameters()),'initial_hash':init_hash,'final_hash':model_hash(model),'history':hist,'noise_buckets':buckets,'generator_sha256':sha(out/'generator.pt'),'assets_events_sha256':sha(Path(a.assets)/'events.npz'),'elapsed_seconds':time.time()-t0,'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'gpu_name':torch.cuda.get_device_name(0),'access':{'DEV_ACCESSED':False,'INTERNAL_ACCESSED':False,'CONFIRM_ACCESSED':False,'TEST_ACCESSED':False}}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({'status':'COMPLETE','seed':a.seed,'last_loss':hist[-1]['loss'],'sha':result['generator_sha256']},sort_keys=True))
if __name__=='__main__': main()
