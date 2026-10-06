from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch,yaml
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'src'))
from diffusion_experiments.models.round4_joint_preference import JointSemanticPreference,rerank_boundary
from diffusion_experiments.scripts.round4_build_assets import build_supervision
from diffusion_experiments.scripts.round1_build_assets import dev_labels
from modules.ranking import metrics_at,metric_arrays,sha256_file
PRIMARY=('R10','N10','R20','N20'); ALL=PRIMARY+('R50','N50')

def sha(p): return sha256_file(Path(p))
def labels_probe(users,targets): return {int(u):{int(t)} for u,t in zip(users,targets)}
def result_vs(base,new):
    rel={k:(float(new[k])-float(base[k]))/max(float(base[k]),1e-12) for k in PRIMARY}
    return {'metrics':{k:float(new[k]) for k in ALL},'relative_primary':rel,'U':float(np.mean(list(rel.values()))),'delta_R50':float(new['R50']-base['R50']),'delta_N50':float(new['N50']-base['N50'])}
def load_model(run,update,zdim,udim,cdim,cfg,device):
    mcfg=cfg['model']; m=JointSemanticPreference(zdim,udim,cdim,int(mcfg['hidden_dim']),int(mcfg['time_dim']),int(mcfg['label_dim']),float(mcfg['dropout'])).to(device); ck=torch.load(Path(run)/'checkpoints'/f'update_{update:04d}.pt',map_location=device,weights_only=False); m.load_state_dict(ck['model']); m.eval(); return m
def infer_logits(model,items,A,cond,users,userctx,zitem,device,user_override=None,batch=256):
    out=np.zeros(items.shape,np.float32); ubase=userctx[users] if user_override is None else user_override
    with torch.no_grad():
      for st in range(0,len(users),batch):
        en=min(st+batch,len(users)); aa=A[st:en]; rr,cc=np.nonzero(aa)
        if not len(rr): continue
        it=items[st:en][rr,cc]; z=torch.as_tensor(zitem[it],device=device); u=torch.as_tensor(ubase[st:en][rr],device=device); c=torch.as_tensor(cond[st:en][rr,cc],device=device); r=model.deploy_logit(z,u,c).cpu().numpy(); out[st+rr,cc]=r
    return out
def eval_eta(items,s0,A,logits,eta,users,labels):
    base=metrics_at(items,users,labels); ranked,delta=rerank_boundary(items,s0,A,logits,float(eta)); res=result_vs(base,metrics_at(ranked,users,labels)); res['changed_users']=int(np.any(ranked!=items,axis=1).sum()); res['delta_abs_mean_A']=float(np.abs(delta[A]).mean()) if A.any() else 0.; res['delta_saturation_rate_A']=float((np.abs(np.tanh(logits[A]))>.95).mean()) if A.any() else 0.; return res,ranked,delta
def per_user(items,users,labels):
    _,_,hit,n=metric_arrays(items,users,labels,max_k=20); ranks=np.arange(1,21,dtype=np.float64)[None,:]; rec=np.cumsum(hit,1)/n[:,None]; dcg=np.cumsum(hit/np.log2(ranks+1),1); idcg=np.cumsum(np.ones_like(hit,dtype=float)/np.log2(ranks+1),1)
    for i,k in enumerate(n):
        cut=min(int(k),20)
        if cut<20: idcg[i,cut:]=idcg[i,cut-1]
    nd=dcg/idcg; return np.stack([rec[:,9],nd[:,9],rec[:,19],nd[:,19]],1)
def U_arr(base,new):
    b=base.mean(0); n=new.mean(0); return float(np.mean((n-b)/np.maximum(b,1e-12)))
def bootstrap(base,new,resamples,seed):
    rng=np.random.default_rng(seed); vals=[]; n=len(base)
    for _ in range(resamples):
        ix=rng.integers(0,n,n); vals.append(U_arr(base[ix],new[ix]))
    v=np.asarray(vals); return {'U':U_arr(base,new),'ci95':[float(np.quantile(v,.025)),float(np.quantile(v,.975))],'positive_fraction':float((v>0).mean()),'resamples':resamples,'seed':seed,'note':'conditional paired user bootstrap; positive_fraction is not a p-value'}
def transitions(base,new,users,labels,k):
    inc=dec=0
    for i,u in enumerate(users):
        lab=labels[int(u)]; b=sum(int(x) in lab for x in base[i,:k]); n=sum(int(x) in lab for x in new[i,:k]); inc+=n>b; dec+=n<b
    return {'increase':int(inc),'decrease':int(dec),'net':int(inc-dec)}

def pair_diag(logits,sup,probe_rows_to_local):
    vals=[]; dep=[]
    for qi,r0 in enumerate(sup['probe_rows']):
        li=int(probe_rows_to_local[int(r0)])
        if li<0 or int(sup['positive_b_rank'][qi])<=0: continue
        pp=int(sup['positive_b_rank'][qi])-1
        for k in range(sup['negative_positions'].shape[1]):
            if not sup['negative_mask'][qi,k]: continue
            nn=int(sup['negative_positions'][qi,k]); d=float(logits[li,pp]-logits[li,nn]); vals.append(d)
            if sup['deploy_pair_mask'][qi,k]: dep.append(d)
    def one(v):
        v=np.asarray(v,np.float64); return {'pairs':int(len(v)),'mean_margin':float(v.mean()) if len(v) else None,'win_rate':float(((v>0).sum()+.5*(v==0).sum())/len(v)) if len(v) else None}
    return {'all':one(vals),'deploy_A':one(dep)}

def choose_common(grid,cfg):
    viable=[]
    for x in grid:
        eta=float(x['eta']); protect=True
        if eta>0:
            for z in x['seed_results'].values():
                if z['relative_primary']['N10'] < -float(cfg['selection']['max_ndcg_relative_drop']) or z['relative_primary']['N20'] < -float(cfg['selection']['max_ndcg_relative_drop']): protect=False
        if protect: viable.append(x)
    positive=[x for x in viable if x['mean_U']>0]
    if not positive:
        return min([x for x in viable if float(x['eta'])==0.0],key=lambda x:int(x['update']))
    return sorted(positive,key=lambda x:(-float(x['mean_U']),float(x['eta']),int(x['update'])))[0]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--assets',required=True); ap.add_argument('--run-root',required=True); ap.add_argument('--out',required=True); a=ap.parse_args(); asset=Path(a.assets); runroot=Path(a.run_root); out=Path(a.out)
    if out.exists() and any(out.iterdir()): raise RuntimeError('refuse overwrite nonempty analysis dir')
    out.mkdir(parents=True,exist_ok=True); cfg=yaml.safe_load((ROOT/'diffusion_experiments/configs/round4_baby.yaml').read_text()); man=json.loads((asset/'manifest.json').read_text())
    if man['access']['CONFIRM_ACCESSED'] or man['access']['TEST_ACCESSED']: raise RuntimeError('closed-set invariant')
    device=torch.device('cuda:0');
    if not torch.cuda.is_available() or '5090' not in torch.cuda.get_device_name(0): raise RuntimeError('RTX5090 required')
    common=np.load(asset/'common.npz'); zitem=common['item_z_tv'].astype(np.float32); userctx=common['user_context'].astype(np.float32); fit_degree=common['fit_degree'].astype(np.int64); cf_norm2=common['cf_norm2'].astype(np.float32)
    seeds=[int(x) for x in cfg['training_seeds']]; depths=[int(x) for x in cfg['formal_depths']]; updates=[int(x) for x in cfg['model']['checkpoint_updates']]; etas=[float(x) for x in cfg['selection']['eta_candidates']]
    runs={}
    for L in depths:
        for seed in seeds:
            name=f'L{L}_seed{seed}'; r=json.loads((runroot/name/'result.json').read_text());
            if r['status']!='COMPLETE' or any(r['access'].values()): raise RuntimeError(f'invalid formal run {name}: {r.get("access")}')
            runs[name]=r
    devlabs=dev_labels(ROOT/'data/baby/baby.inter',np.load(asset/'L100_dev.npz')['users'].astype(np.int64)); selections={}; dev_grids={}
    for L in depths:
        d=np.load(asset/f'L{L}_dev.npz'); users=d['users'].astype(np.int64); items=d['items'].astype(np.int32); s0=d['s0'].astype(np.float32); A=d['A_mask'].astype(bool); cond=d['candidate_cond'].astype(np.float32); grid=[]
        logcache={}
        for upd in updates:
            for seed in seeds:
                m=load_model(runroot/f'L{L}_seed{seed}',upd,64,userctx.shape[1],cond.shape[-1],cfg,device); logcache[(upd,seed)]=infer_logits(m,items,A,cond,users,userctx,zitem,device)
            for eta in etas:
                sr={};
                for seed in seeds: sr[str(seed)]=eval_eta(items,s0,A,logcache[(upd,seed)],eta,users,devlabs)[0]
                grid.append({'update':upd,'eta':eta,'mean_U':float(np.mean([sr[str(s)]['U'] for s in seeds])),'seed_results':sr})
        sel=choose_common(grid,cfg); selections[str(L)]={'update':int(sel['update']),'eta':float(sel['eta']),'mean_DEV_U':float(sel['mean_U']),'DEV_seed_results':sel['seed_results']}; dev_grids[str(L)]=grid
    lock={'status':'LOCKED_BEFORE_INTERNAL_TEST','protocol_version':cfg['protocol_version'],'selection':selections,'asset_manifest_sha256':sha(asset/'manifest.json'),'formal_run_result_hashes':{k:sha(runroot/k/'result.json') for k in sorted(runs)},'access':{'INTERNAL_USED_FOR_SELECTION':False,'CONFIRM_ACCESSED':False,'TEST_ACCESSED':False,'TEST_USED_FOR_SELECTION':False}}
    (out/'selection_lock.json').write_text(json.dumps(lock,indent=2)+'\n'); lock_hash=sha(out/'selection_lock.json')
    fit=pd.read_csv(ROOT/cfg['protocol_dir']/'fit_edges.csv'); mon=pd.read_csv(ROOT/cfg['protocol_dir']/'monitor_edges.csv'); prdf=pd.read_csv(ROOT/cfg['protocol_dir']/'probe_edges.csv').sort_values('userID'); int_users=pd.read_csv(ROOT/cfg['protocol_dir']/'internal_users.csv').userID.to_numpy(np.int64); target_map=dict(zip(prdf.userID.astype(int),prdf.itemID.astype(int)))
    internal={}; rows_csv=[]; rng_shuffle=np.random.default_rng(int(cfg['bootstrap']['seed'])+17)
    for L in depths:
        p=np.load(asset/f'L{L}_probe.npz'); allusers=p['users'].astype(np.int64); im=p['internal'].astype(bool); ir=np.flatnonzero(im); users=allusers[im]; items=p['items'][im].astype(np.int32); s0=p['s0'][im].astype(np.float32); A=p['A_mask'][im].astype(bool); cond=p['candidate_cond'][im].astype(np.float32); targets=p['target_items'][im].astype(np.int32); labs=labels_probe(users,targets); inv=np.full(len(allusers),-1,np.int32); inv[ir]=np.arange(len(ir),dtype=np.int32)
        intsup,_=build_supervision(L,allusers,p['msca_items'].astype(np.int32),p['items'].astype(np.int32),p['s0'].astype(np.float32),p['A_mask'].astype(bool),fit,mon,target_map,fit_degree,cf_norm2,int_users,cfg)
        trmask=p['reranker_train'].astype(bool); trrows=np.flatnonzero(trmask); trusers=allusers[trmask]; trinv=np.full(len(allusers),-1,np.int32); trinv[trrows]=np.arange(len(trrows),dtype=np.int32); tritems=p['items'][trmask].astype(np.int32); trA=p['A_mask'][trmask].astype(bool); trcond=p['candidate_cond'][trmask].astype(np.float32); trainsup={k:np.load(asset/f'L{L}_supervision.npz')[k] for k in np.load(asset/f'L{L}_supervision.npz').files}
        sel=selections[str(L)]; upd=int(sel['update']); eta=float(sel['eta']); seedres={}; perm=rng_shuffle.permutation(len(users)); shufctx=userctx[users[perm]]
        for seed in seeds:
            model=load_model(runroot/f'L{L}_seed{seed}',upd,64,userctx.shape[1],cond.shape[-1],cfg,device); logits=infer_logits(model,items,A,cond,users,userctx,zitem,device); res,ranked,delta=eval_eta(items,s0,A,logits,eta,users,labs); shlog=infer_logits(model,items,A,cond,users,userctx,zitem,device,user_override=shufctx); shres,_,_=eval_eta(items,s0,A,shlog,eta,users,labs)
            bpu=per_user(items,users,labs); npu=per_user(ranked,users,labs); boot=bootstrap(bpu,npu,int(cfg['bootstrap']['resamples']),int(cfg['bootstrap']['seed'])+seed+L)
            trlog=infer_logits(model,tritems,trA,trcond,trusers,userctx,zitem,device); diag={'TRAIN':pair_diag(trlog,trainsup,trinv),'INTERNAL':pair_diag(logits,intsup,inv)}
            seedres[str(seed)]={'metrics':res,'shuffled_user_condition':shres,'bootstrap':boot,'transitions':{'K10':transitions(items,ranked,users,labs,10),'K20':transitions(items,ranked,users,labs,20)},'pair_diagnostics':diag}
            rows_csv.append({'L':L,'seed':seed,'update':upd,'eta':eta,'DEV_U':sel['DEV_seed_results'][str(seed)]['U'],'INTERNAL_U':res['U'],'INTERNAL_R10':res['metrics']['R10'],'INTERNAL_N10':res['metrics']['N10'],'INTERNAL_R20':res['metrics']['R20'],'INTERNAL_N20':res['metrics']['N20'],'INTERNAL_R50':res['metrics']['R50'],'INTERNAL_N50':res['metrics']['N50'],'SHUFFLED_INTERNAL_U':shres['U']})
        mean_int=float(np.mean([seedres[str(s)]['metrics']['U'] for s in seeds])); devpos=all(sel['DEV_seed_results'][str(s)]['U']>0 for s in seeds); intpos=all(seedres[str(s)]['metrics']['U']>0 for s in seeds); nprot=all(seedres[str(s)]['metrics']['relative_primary']['N10']>=-float(cfg['selection']['max_ndcg_relative_drop']) and seedres[str(s)]['metrics']['relative_primary']['N20']>=-float(cfg['selection']['max_ndcg_relative_drop']) for s in seeds)
        if sel['mean_DEV_U']>=.01 and mean_int>=.01 and devpos and intpos and nprot: cls='BABY_DEVELOPMENT_TARGET_MET'
        elif sel['mean_DEV_U']>0 and mean_int>0: cls='POSITIVE_BELOW_TARGET'
        else: cls='VALID_NO_INCREMENT'
        internal[str(L)]={'mean_INTERNAL_U':mean_int,'seed_results':seedres,'classification':cls,'DEV_mean_U':sel['mean_DEV_U'],'all_DEV_positive':devpos,'all_INTERNAL_positive':intpos,'INTERNAL_N_protection':nprot}
    summary={'status':'COMPLETE_LOCKED_INTERNAL_EVALUATED','protocol_version':cfg['protocol_version'],'selection_lock_sha256':lock_hash,'selection':selections,'DEV_grid':dev_grids,'INTERNAL':internal,'stage0':man['depths'],'formal_runs':runs,'historical_exposure_note':'DEV/INTERNAL are development evidence; old full Validation/Test were historically exposed. No fresh confirmation claim.','access':{'CONFIRM_ACCESSED':False,'TEST_ACCESSED':False,'TEST_USED_FOR_SELECTION':False}}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with open(out/'results.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows_csv[0])); w.writeheader(); w.writerows(rows_csv)
    print(json.dumps({'status':'COMPLETE','lock_sha256':lock_hash,'selection':selections,'internal':{L:{'mean_U':v['mean_INTERNAL_U'],'classification':v['classification']} for L,v in internal.items()}},sort_keys=True))

if __name__=='__main__': main()
