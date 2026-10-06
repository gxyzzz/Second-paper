from __future__ import annotations
import argparse, hashlib, io, json, math, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch, yaml
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'src'))
from diffusion_experiments.models.round4_joint_preference import JointSemanticPreference,q_sample
from modules.ranking import sha256_file

def sha(p): return sha256_file(Path(p))
def model_hash(m):
    b=io.BytesIO(); torch.save(m.state_dict(),b); return hashlib.sha256(b.getvalue()).hexdigest()
def grad_norm(loss,params):
    if not loss.requires_grad: return 0.0
    gs=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
    return float(math.sqrt(sum(float((g.detach()**2).sum()) for g in gs if g is not None)))
def sample_tx(n,cfg,gen,device):
    clean=torch.rand((n,),generator=gen,device=device)<float(cfg['noise']['tx_clean_probability'])
    t=torch.randint(1,int(cfg['noise']['diffusion_steps'])+1,(n,),generator=gen,device=device); t[clean]=0
    return t
def diffuse_forward(model,z,u,c,tx,ty,lab,gen,cfg):
    eps=torch.randn(z.shape,generator=gen,device=z.device); zt=q_sample(z,tx,eps,int(cfg['noise']['diffusion_steps']))
    ep,r=model(zt,u,c,tx,ty,lab); mask=tx>0
    e=(ep-eps).pow(2); d=e.shape[-1]//2; ld=.5*(e[:,:d].mean(1)+e[:,d:].mean(1)); ld=torch.where(mask,ld,torch.zeros_like(ld))
    return ld,r,mask

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--assets',required=True); ap.add_argument('--depth',type=int,required=True); ap.add_argument('--seed',type=int,required=True); ap.add_argument('--out',required=True); ap.add_argument('--mode',choices=['smoke','formal'],default='formal'); a=ap.parse_args()
    out=Path(a.out); asset=Path(a.assets)
    if out.exists() and any(out.iterdir()): raise RuntimeError('refuse overwrite nonempty run')
    out.mkdir(parents=True,exist_ok=True); ckdir=out/'checkpoints'; ckdir.mkdir()
    cfg=yaml.safe_load((ROOT/'diffusion_experiments/configs/round4_baby.yaml').read_text()); man=json.loads((asset/'manifest.json').read_text()); L=int(a.depth); seed=int(a.seed)
    if L not in [int(x) for x in cfg['formal_depths']]: raise RuntimeError('unregistered depth')
    if a.mode=='formal' and seed not in [int(x) for x in cfg['training_seeds']]: raise RuntimeError('unregistered seed')
    if not man['depths'][str(L)]['supervision']['sufficient']: raise RuntimeError(f'stage0 blocked depth {L}')
    if man['access']['CONFIRM_ACCESSED'] or man['access']['TEST_ACCESSED']: raise RuntimeError('closed-set invariant')
    for n,h in man['artifacts'].items():
        if sha(asset/n)!=h: raise RuntimeError(f'asset hash mismatch {n}')
    if not torch.cuda.is_available() or '5090' not in torch.cuda.get_device_name(0): raise RuntimeError('RTX5090 required')
    dev=torch.device('cuda:0'); torch.manual_seed(seed); np.random.seed(seed)
    common=np.load(asset/'common.npz'); probe=np.load(asset/f'L{L}_probe.npz'); sup=np.load(asset/f'L{L}_supervision.npz')
    z_item=common['item_z_tv'].astype(np.float32); uctx=common['user_context'].astype(np.float32); items=probe['items'].astype(np.int32); s0=probe['s0'].astype(np.float32); cond=probe['candidate_cond'].astype(np.float32); A=probe['A_mask'].astype(bool)
    users=sup['users'].astype(np.int64); prow=sup['probe_rows'].astype(np.int32); pos=sup['positive_items'].astype(np.int32); posrank=sup['positive_b_rank'].astype(np.int32); negpos=sup['negative_positions'].astype(np.int32); negmask=sup['negative_mask'].astype(bool); depmask=sup['deploy_pair_mask'].astype(bool)
    mcfg=cfg['model']; model=JointSemanticPreference(64,uctx.shape[1],cond.shape[-1],int(mcfg['hidden_dim']),int(mcfg['time_dim']),int(mcfg['label_dim']),float(mcfg['dropout'])).to(dev); init_hash=model_hash(model); params=list(model.parameters())
    opt=torch.optim.AdamW(params,lr=float(mcfg['learning_rate']),weight_decay=float(mcfg['weight_decay'])); gen=torch.Generator(device=dev); gen.manual_seed(seed*1009+17); rng=np.random.default_rng(seed*1013+31)
    eligible=np.flatnonzero(negmask.sum(1)>0); allq=np.arange(len(users)); weights=np.where(depmask.sum(1)>0,float(cfg['pairing']['boundary_sampling_weight']),1.0); weights=weights[eligible]/weights[eligible].sum()
    max_updates=20 if a.mode=='smoke' else int(mcfg['max_updates']); ck_updates=set([10,20] if a.mode=='smoke' else [int(x) for x in mcfg['checkpoint_updates']]); hist=[]; first_grad=None; start=time.time(); torch.cuda.reset_peak_memory_stats(dev)
    warm=2 if a.mode=='smoke' else int(mcfg['warmup_updates']); log_acc={'total':0.,'diff':0.,'pref':0.,'deploy':0.,'reg':0.,'n':0}
    for step in range(1,max_updates+1):
        model.train(); bs=min(int(mcfg['supervised_batch_queries']),len(eligible)); uqbs=min(int(mcfg['unlabeled_batch_queries']),len(allq)); pick=rng.choice(eligible,size=bs,replace=False,p=weights); uq=rng.choice(allq,size=uqbs,replace=False)
        br=prow[pick]; pp=(posrank[pick]-1).astype(np.int64); nm=negmask[pick]; np0=np.maximum(negpos[pick],0).astype(np.int64); B=len(pick); K=nm.shape[1]
        pit=pos[pick].astype(np.int64); nit=items[br[:,None],np0].astype(np.int64); pc=cond[br,pp]; nc=cond[br[:,None],np0]; uc=uctx[users[pick]]
        # Pair corruption clocks. t_y is tied within each positive/comparison pair; t_x and Gaussian noise are item-independent.
        nflat=B*K; endpoint=torch.rand((nflat,),generator=gen,device=dev)<float(cfg['noise']['ty_mask_endpoint_probability']); ty=torch.randint(1,int(cfg['noise']['diffusion_steps'])+1,(nflat,),generator=gen,device=dev); ty[endpoint]=int(cfg['noise']['diffusion_steps']); masked=torch.rand((nflat,),generator=gen,device=dev)<(ty.float()/float(cfg['noise']['diffusion_steps']))
        labp=torch.where(masked,torch.full_like(ty,2),torch.full_like(ty,1)); labn=torch.where(masked,torch.full_like(ty,2),torch.zeros_like(ty))
        zp=torch.as_tensor(z_item[pit],device=dev)[:,None,:].expand(-1,K,-1).reshape(nflat,64); zn=torch.as_tensor(z_item[nit.reshape(-1)],device=dev); up=torch.as_tensor(uc,device=dev)[:,None,:].expand(-1,K,-1).reshape(nflat,-1); cp=torch.as_tensor(pc,device=dev)[:,None,:].expand(-1,K,-1).reshape(nflat,-1); cn=torch.as_tensor(nc.reshape(nflat,-1),device=dev)
        txp=sample_tx(nflat,cfg,gen,dev); txn=sample_tx(nflat,cfg,gen,dev); ldp,rp,_=diffuse_forward(model,zp,up,cp,txp,ty,labp,gen,cfg); ldn,rn,_=diffuse_forward(model,zn,up,cn,txn,ty,labn,gen,cfg)
        valid=torch.as_tensor(nm.reshape(-1),device=dev); pairdiff=.5*(ldp+ldn); qdiff=(pairdiff.reshape(B,K)*torch.as_tensor(nm,device=dev).float()).sum(1)/torch.as_tensor(nm,device=dev).float().sum(1).clamp_min(1); sdiff=qdiff.mean()
        pm=(valid & masked).reshape(B,K); pl=F.softplus(-(rp-rn)).reshape(B,K); qpref=(pl*pm.float()).sum(1)/pm.float().sum(1).clamp_min(1); pref=qpref[pm.sum(1)>0].mean() if (pm.sum(1)>0).any() else torch.zeros((),device=dev)
        regmask=(valid & masked); reg=((rp.pow(2)+rn.pow(2))[regmask].mean()/2.0) if regmask.any() else torch.zeros((),device=dev)
        # Explicit clean+MASK deployment loss only for registered A-internal local pairs.
        dm=torch.as_tensor((depmask[pick]&nm).reshape(-1),device=dev); zclean_p=zp; zclean_n=zn; zeros=torch.zeros(nflat,dtype=torch.long,device=dev); fifty=torch.full((nflat,),50,dtype=torch.long,device=dev); masklab=torch.full((nflat,),2,dtype=torch.long,device=dev)
        _,rdp=model(zclean_p,up,cp,zeros,fifty,masklab); _,rdn=model(zclean_n,up,cn,zeros,fifty,masklab)
        sp=torch.as_tensor(s0[br,pp],device=dev)[:,None].expand(-1,K).reshape(-1); sn=torch.as_tensor(s0[br[:,None],np0].reshape(-1),device=dev); eta=float(mcfg['deploy_eta_train']); temp=float(mcfg['deploy_temperature']); dl=F.softplus(-(sp+eta*torch.tanh(rdp)-sn-eta*torch.tanh(rdn))/temp).reshape(B,K); dmm=(depmask[pick]&nm); qdep=(dl*torch.as_tensor(dmm,device=dev).float()).sum(1)/torch.as_tensor(dmm,device=dev).float().sum(1).clamp_min(1); qhas=torch.as_tensor(np.asarray(dmm).sum(1)>0,device=dev); deploy=qdep[qhas].mean() if qhas.any() else torch.zeros((),device=dev)
        # Unlabeled natural-candidate semantic diffusion: 50% A when A exists, otherwise the full natural pool.
        ur=prow[uq]; upos=[]
        for r in ur:
            aa=np.flatnonzero(A[r])
            if len(aa) and rng.random()<0.5: upos.append(int(rng.choice(aa)))
            else: upos.append(int(rng.integers(0,L)))
        upos=np.asarray(upos,np.int64); uit=items[ur,upos]; uz=torch.as_tensor(z_item[uit],device=dev); uu=torch.as_tensor(uctx[users[uq]],device=dev); ucand=torch.as_tensor(cond[ur,upos],device=dev)
        utx=sample_tx(len(uq),cfg,gen,dev); uty=torch.full((len(uq),),50,dtype=torch.long,device=dev); ulab=torch.full((len(uq),),2,dtype=torch.long,device=dev)
        uld,_,_=diffuse_forward(model,uz,uu,ucand,utx,uty,ulab,gen,cfg); udiff=uld.mean(); diff=(sdiff*B+udiff*len(uq))/float(B+len(uq))
        if step<=warm: total=float(mcfg['lambda_diff'])*diff
        else: total=pref+deploy+float(mcfg['lambda_diff'])*diff+float(mcfg['lambda_logit_l2'])*reg
        if not torch.isfinite(total): raise FloatingPointError('nonfinite total loss')
        shared=[*model.fc1.parameters(),*model.fc2.parameters()]
        if first_grad is None and step>warm:
            first_grad={'step':step,'diff_shared':grad_norm(diff,shared),'pref_shared':grad_norm(pref,shared),'deploy_shared':grad_norm(deploy,shared),'total_shared':grad_norm(total,shared)}
            for key in ['diff_shared','pref_shared','deploy_shared','total_shared']:
                if not np.isfinite(first_grad[key]) or first_grad[key]<=0: raise RuntimeError(f'zero/nonfinite shared gradient {key}: {first_grad}')
        opt.zero_grad(set_to_none=True); total.backward(); gn=torch.nn.utils.clip_grad_norm_(params,float(mcfg['grad_clip']))
        if not torch.isfinite(torch.as_tensor(gn)): raise FloatingPointError('nonfinite grad norm')
        opt.step()
        for k,v in [('total',total),('diff',diff),('pref',pref),('deploy',deploy),('reg',reg)]: log_acc[k]+=float(v.detach())
        log_acc['n']+=1
        if step%50==0 or step in ck_updates or step==max_updates:
            rec={'step':step,'loss':{k:(v/max(log_acc['n'],1) if k!='n' else int(v)) for k,v in log_acc.items()},'first_joint_shared_grad':first_grad,'supervised_queries_per_step':int(B),'unlabeled_queries_per_step':int(len(uq)),'masked_pair_fraction':float(masked.float().mean().detach()),'deploy_pair_fraction':float(dm.float().mean().detach())}
            hist.append(rec); (out/'history.json').write_text(json.dumps(hist,indent=2)+'\n'); log_acc={'total':0.,'diff':0.,'pref':0.,'deploy':0.,'reg':0.,'n':0}
        if step in ck_updates:
            state={'model':model.state_dict(),'optimizer':opt.state_dict(),'step':step,'seed':seed,'depth':L,'torch_generator_state':gen.get_state(),'numpy_rng_state':rng.bit_generator.state,'initial_model_hash':init_hash}
            torch.save(state,ckdir/f'update_{step:04d}.pt')
    result={'status':'COMPLETE','protocol_version':cfg['protocol_version'],'depth':L,'seed':seed,'mode':a.mode,'initial_model_hash':init_hash,'model_hash':model_hash(model),'parameter_count':sum(p.numel() for p in model.parameters()),'optimizer_steps':max_updates,'checkpoints':sorted(ck_updates),'first_joint_shared_grad':first_grad,'train_seconds':time.time()-start,'peak_cuda_memory_bytes':int(torch.cuda.max_memory_allocated(dev)),'supervision_stats':man['depths'][str(L)]['supervision'],'asset_manifest_sha256':sha(asset/'manifest.json'),'config_sha256':sha(ROOT/'diffusion_experiments/configs/round4_baby.yaml'),'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'tracked_diff_names':subprocess.check_output(['git','diff','--name-only'],cwd=ROOT,text=True).splitlines(),'gpu_name':torch.cuda.get_device_name(0),'gpu_uuid':subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()[0].strip(),'access':{'DEV_ACCESSED':False,'INTERNAL_ACCESSED':False,'CONFIRM_ACCESSED':False,'TEST_ACCESSED':False}}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({'status':'COMPLETE','depth':L,'seed':seed,'steps':max_updates,'first_joint_shared_grad':first_grad},sort_keys=True))

if __name__=='__main__': main()
