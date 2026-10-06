from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'src'))
from diffusion_experiments.models.round1_residual import BoundaryResidualDiffusion, ddim_sample
from diffusion_experiments.scripts.round1_train_residual import rerank
from modules.ranking import metrics_at


def label_sets(users):
    keep=set(map(int,users)); out={int(u):set() for u in users}
    for c in pd.read_csv(ROOT/'data/baby/baby.inter',sep='\t',usecols=['userID','itemID','x_label'],chunksize=50000):
        z=c[(c.x_label==1)&c.userID.isin(keep)]
        for u,g in z.groupby('userID'): out[int(u)].update(g.itemID.astype(int).tolist())
    return out

def cutoff_changes(base,new,users,labels,k):
    corrected=broken=both=neither=0
    for r,u in enumerate(users):
        pos=labels[int(u)]; a=bool(set(base[r,:k].tolist())&pos); b=bool(set(new[r,:k].tolist())&pos)
        if (not a) and b: corrected+=1
        elif a and (not b): broken+=1
        elif a and b: both+=1
        else: neither+=1
    return {'corrected':corrected,'broken':broken,'net':corrected-broken,'both_hit':both,'both_miss':neither}

def target_rank(items,target):
    ranks=[]
    for row,t in zip(items,target):
        p=np.flatnonzero(row==t); ranks.append(int(p[0])+1 if len(p) else 0)
    return np.asarray(ranks,dtype=np.int16)

def run_one(run_dir:Path,assets:Path):
    cfg=yaml.safe_load((ROOT/'diffusion_experiments/configs/round1_baby.yaml').read_text()); mc=cfg['model']; rc=cfg['residual']; sampling=list(map(int,cfg['formal']['sampling_seeds']))
    result=json.loads((run_dir/'result.json').read_text()); ck=torch.load(run_dir/'best.pt',map_location='cpu',weights_only=False)
    model=BoundaryResidualDiffusion(22,int(mc['hidden_dim']),int(mc['attention_heads']),int(mc['attention_layers']),float(mc['dropout'])).cuda(); model.load_state_dict(ck['model']); model.eval()
    mean=ck['feature_mean']; std=ck['feature_std']; sigma=float(ck['sigma_r']); eta=float(ck['eta']); c=float(rc['clip_c']); tau=float(rc['tau'])
    dev=np.load(assets/'dev_top100.npz'); users=dev['users'].astype(np.int64); items=dev['items'].astype(np.int32); s0=dev['s0'].astype(np.float32); feat=((dev['features'][:,5:30,:].astype(np.float32)-mean)/std).astype(np.float32)
    labels=label_sets(users)
    per={s:[] for s in sampling}; one=[]
    with torch.no_grad():
        for st in range(0,len(users),512):
            en=min(st+512,len(users)); f=torch.as_tensor(feat[st:en],device='cuda'); u=torch.as_tensor(users[st:en],device='cuda')
            for s in sampling: per[s].append(ddim_sample(model,f,u,sigma,s,5,int(mc['diffusion_steps']),torch.device('cuda')).cpu().numpy())
            one.append(ddim_sample(model,f,u,sigma,sampling[0],1,int(mc['diffusion_steps']),torch.device('cuda')).cpu().numpy())
    per={s:np.concatenate(v) for s,v in per.items()}; one=np.concatenate(one); avg=np.mean(np.stack(list(per.values())),axis=0)
    ranked5=rerank(items,s0[:,5:30],avg,sigma,eta,c,tau); ranked1=rerank(items,s0[:,5:30],one,sigma,eta,c,tau)
    mbase=metrics_at(items,users,labels); m5=metrics_at(ranked5,users,labels); m1=metrics_at(ranked1,users,labels)
    seed_metrics={str(s):metrics_at(rerank(items,s0[:,5:30],x,sigma,eta,c,tau),users,labels) for s,x in per.items()}
    rms={}
    ss=sampling
    for i in range(len(ss)):
        for j in range(i+1,len(ss)):
            rms[f'{ss[i]}_{ss[j]}']=float(np.sqrt(np.mean((per[ss[i]]-per[ss[j]])**2)))
    probe=np.load(assets/'probe_top100.npz'); tgt=np.load(assets/'probe_targets.npz'); mask=tgt['internal'].astype(bool)&(tgt['target_rank']>=6)&(tgt['target_rank']<=30)
    pu=probe['users'][mask].astype(np.int64); pi=probe['items'][mask].astype(np.int32); ps=probe['s0'][mask].astype(np.float32); pf=((probe['features'][mask,5:30,:].astype(np.float32)-mean)/std).astype(np.float32); target=tgt['target_items'][mask].astype(np.int32)
    outs={s:[] for s in sampling}
    with torch.no_grad():
        for st in range(0,len(pu),512):
            en=min(st+512,len(pu)); f=torch.as_tensor(pf[st:en],device='cuda'); u=torch.as_tensor(pu[st:en],device='cuda')
            for s in sampling: outs[s].append(ddim_sample(model,f,u,sigma,s,5,int(mc['diffusion_steps']),torch.device('cuda')).cpu().numpy())
    xint=np.mean(np.stack([np.concatenate(outs[s]) for s in sampling]),axis=0); rint=rerank(pi,ps[:,5:30],xint,sigma,eta,c,tau); rb=target_rank(pi,target); rn=target_rank(rint,target); delta=rn-rb
    internal={'queries':int(len(rb)),'mean_rank_before':float(rb.mean()),'mean_rank_after':float(rn.mean()),'mean_rank_delta_after_minus_before':float(delta.mean()),'improved':int((delta<0).sum()),'worsened':int((delta>0).sum()),'unchanged':int((delta==0).sum()),'top10_corrected':int(((rb>10)&(rn<=10)).sum()),'top10_broken':int(((rb<=10)&(rn>10)).sum()),'top20_corrected':int(((rb>20)&(rn<=20)).sum()),'top20_broken':int(((rb<=20)&(rn>20)).sum())}
    return {'seed':result['seed'],'best_epoch':result['best']['epoch'],'eta':eta,'baseline_metrics':mbase,'five_step_four_sample_metrics':m5,'one_step_first_sample_metrics':m1,'five_step_single_seed_metrics':seed_metrics,'cutoff_changes':{'K10':cutoff_changes(items,ranked5,users,labels,10),'K20':cutoff_changes(items,ranked5,users,labels,20)},'sample_residual_pairwise_rms':rms,'one_vs_five_residual_rms_first_seed':float(np.sqrt(np.mean((one-per[sampling[0]])**2))),'internal':internal}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--assets',required=True); ap.add_argument('--runs',nargs='+',required=True); ap.add_argument('--out',required=True); a=ap.parse_args()
    if not torch.cuda.is_available() or '5090' not in torch.cuda.get_device_name(0): raise RuntimeError('RTX5090 required')
    payload={'status':'COMPLETE','gpu':torch.cuda.get_device_name(0),'runs':[run_one(Path(x),Path(a.assets)) for x in a.runs],'access':{'CONFIRM_ACCESSED':False,'TEST_ACCESSED':False}}
    Path(a.out).write_text(json.dumps(payload,indent=2)+'\n'); print(json.dumps(payload,sort_keys=True))
if __name__=='__main__': main()
