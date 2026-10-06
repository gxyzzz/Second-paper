from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch,yaml
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'src'))
from diffusion_experiments.scripts.round1_build_assets import histories_from_fit,colift_params
from diffusion_experiments.scripts.round4_build_assets import modality_values,score_depth,boundary_mask
from diffusion_experiments.scripts.round4_analyze import load_model,infer_logits,eval_eta,per_user,bootstrap,transitions
from modules.attribute import build_item_matrices,build_profiles
from modules.ranking import topk_from_embeddings,metrics_at,sha256_file
from pipelines.dataset_config import load_dataset_config

PRIMARY=('R10','N10','R20','N20'); ALL=PRIMARY+('R50','N50')
def sha(p): return sha256_file(Path(p))
def labels_by_x(inter_path,users,x):
    keep=set(map(int,users)); out={int(u):set() for u in users}
    for ch in pd.read_csv(inter_path,sep='\t',usecols=['userID','itemID','x_label'],chunksize=50000):
        z=ch[(ch.x_label==int(x)) & ch.userID.isin(keep)]
        for u,g in z.groupby('userID'): out[int(u)].update(g.itemID.astype(int).tolist())
    return out
def result_vs(base,new):
    rel={k:(float(new[k])-float(base[k]))/max(float(base[k]),1e-12) for k in PRIMARY}
    return {'metrics':{k:float(new[k]) for k in ALL},'relative_primary':rel,'U':float(np.mean(list(rel.values()))),'delta_R50':float(new['R50']-base['R50']),'delta_N50':float(new['N50']-base['N50'])}
def bg_from_npz(path):
    z=np.load(path); out={}
    for name in ['text','attribute','visual']:
        out[name]={'shrunk_mean':z['mu_'+name].astype(np.float32),'count':z['count_'+name].astype(np.int64),'global_mean':float(z['global_'+name]) if 'global_'+name in z.files else 0.0}
    return out
def load_eval_common(cfg,asset):
    c=np.load(asset/'common.npz'); fit=pd.read_csv(ROOT/cfg['protocol_dir']/'fit_edges.csv'); n_users=int(fit.userID.max())+1; histories=histories_from_fit(fit,n_users)
    return c,fit,histories
def prepare_candidates(cfg,asset,users,L,common,histories,item_matrices,profiles,device):
    fu=torch.as_tensor(common['final_user'],device=device); fi=torch.as_tensor(common['final_item'],device=device)
    raw_items,raw_scores=topk_from_embeddings(fu,fi,users,histories,top_l=L,batch_users=512)
    data_cfg=load_dataset_config('baby'); zt,za,zv=modality_values(data_cfg,histories,users,raw_items,item_matrices,profiles)
    if L==100:
        bg=bg_from_npz(ROOT/cfg['frozen_round1_assets']/'backgrounds.npz')
    else:
        bg=bg_from_npz(asset/f'background_L{L}.npz')
    items,s0,cond,_=score_depth(raw_items,raw_scores,zt,za,zv,bg,colift_params(data_cfg['coliftrec']),L,common['collab_user'].astype(np.float32),common['collab_item'].astype(np.float32),users,make_cond=True)
    ref=np.load(asset/f'L{L}_dev.npz'); cm=ref['cond_mean'].astype(np.float32); cs=ref['cond_std'].astype(np.float32); cond=((cond-cm)/cs).astype(np.float32); A=boundary_mask(items,s0,cfg)
    return raw_items.astype(np.int32),items,s0,cond,A

def dry_run(cfg,asset,runroot,analysis,device,n):
    common,fit,histories=load_eval_common(cfg,asset); du=pd.read_csv(ROOT/cfg['protocol_dir']/'dev_users.csv').userID.to_numpy(np.int64)[:n]; data_cfg=load_dataset_config('baby'); ni=int(common['final_item'].shape[0]); ac=data_cfg['coliftrec']['attribute']; mats,_=build_item_matrices(data_cfg['resolved_paths']['metadata'],ni,min_df=int(ac.get('tfidf_min_df',2)),max_df=float(ac.get('tfidf_max_df',.8)),description_len=int(ac.get('description_len',128)),weights=ac.get('weights')); prof=build_profiles(mats,histories,ni)
    rec={}
    for L in [100,500]:
        ms,bi,s0,cond,A=prepare_candidates(cfg,asset,du,L,common,histories,mats,prof,device); frozen=np.load(asset/f'L{L}_dev.npz'); idx=np.arange(n)
        set_mis=sum(set(ms[i].tolist())!=set(frozen['msca_items'][i].tolist()) for i in idx); bset_mis=sum(set(bi[i].tolist())!=set(frozen['items'][i].tolist()) for i in idx)
        rec[str(L)]={'users':n,'msca_set_mismatch':int(set_mis),'B_set_mismatch':int(bset_mis),'A_count_mean':float(A.sum(1).mean()),'s0_range':[float(s0.min()),float(s0.max())]}
    # Reproduce selected DEV results from already-frozen assets (does not read Test).
    lock=json.load(open(analysis/'selection_lock.json')); userctx=common['user_context'].astype(np.float32); zitem=common['item_z_tv'].astype(np.float32); labs=labels_by_x(data_cfg['resolved_paths']['interaction'],pd.read_csv(ROOT/cfg['protocol_dir']/'dev_users.csv').userID.to_numpy(np.int64),1)
    selected={}
    for L in [100,500]:
        d=np.load(asset/f'L{L}_dev.npz'); users=d['users'].astype(np.int64); items=d['items'].astype(np.int32); s0=d['s0'].astype(np.float32); A=d['A_mask'].astype(bool); cond=d['candidate_cond'].astype(np.float32); sel=lock['selection'][str(L)]; upd=int(sel['update']); eta=float(sel['eta']); sr={}
        for seed in [202610065,202610066]:
            m=load_model(runroot/f'L{L}_seed{seed}',upd,64,userctx.shape[1],cond.shape[-1],cfg,device); lo=infer_logits(m,items,A,cond,users,userctx,zitem,device); rr,_,_=eval_eta(items,s0,A,lo,eta,users,labs); sr[str(seed)]=rr['U']
        selected[str(L)]={'expected_mean_DEV_U':sel['mean_DEV_U'],'recomputed_mean_DEV_U':float(np.mean(list(sr.values()))),'seed_U':sr}
    return {'status':'DRY_RUN_COMPLETE_NO_TEST','candidate_rebuild':rec,'selected_DEV_replay':selected,'access':{'TEST_ACCESSED':False}}

def run_test(cfg,asset,runroot,analysis,out,device):
    lock_path=analysis/'selection_lock.json'; lock=json.load(open(lock_path)); lock_hash=sha(lock_path)
    if lock['access']['TEST_ACCESSED'] or lock['access']['TEST_USED_FOR_SELECTION']: raise RuntimeError('selection lock invalid')
    common,fit,histories=load_eval_common(cfg,asset); data_cfg=load_dataset_config('baby'); n_items=int(common['final_item'].shape[0]); n_users=int(common['final_user'].shape[0])
    users=np.arange(n_users,dtype=np.int64); labels=labels_by_x(data_cfg['resolved_paths']['interaction'],users,2)
    no_label=[int(u) for u in users if not labels[int(u)]]
    if no_label: raise RuntimeError(f'test label missing users={len(no_label)} first={no_label[:10]}')
    ac=data_cfg['coliftrec']['attribute']; mats,_=build_item_matrices(data_cfg['resolved_paths']['metadata'],n_items,min_df=int(ac.get('tfidf_min_df',2)),max_df=float(ac.get('tfidf_max_df',.8)),description_len=int(ac.get('description_len',128)),weights=ac.get('weights')); prof=build_profiles(mats,histories,n_items)
    userctx=common['user_context'].astype(np.float32); zitem=common['item_z_tv'].astype(np.float32); depths={}; strict_b100=None; t0=time.time()
    out.mkdir(parents=True,exist_ok=True)
    for L in [100,500]:
        msca,items,s0,cond,A=prepare_candidates(cfg,asset,users,L,common,histories,mats,prof,device); msca_m=metrics_at(msca,users,labels); b_m=metrics_at(items,users,labels); sel=lock['selection'][str(L)]; upd=int(sel['update']); eta=float(sel['eta']); seedres={}; save={'users':users,'msca_top50':msca[:,:50],'B_top50':items[:,:50],'A_count':A.sum(1).astype(np.int16)}
        if L==100: strict_b100={k:float(v) for k,v in b_m.items()}
        for seed in [202610065,202610066]:
            model=load_model(runroot/f'L{L}_seed{seed}',upd,64,userctx.shape[1],cond.shape[-1],cfg,device); logits=infer_logits(model,items,A,cond,users,userctx,zitem,device); res,ranked,delta=eval_eta(items,s0,A,logits,eta,users,labels)
            bpu=per_user(items,users,labels); npu=per_user(ranked,users,labels); boot=bootstrap(bpu,npu,int(cfg['bootstrap']['resamples']),int(cfg['bootstrap']['seed'])+700000+L+seed)
            seedres[str(seed)]={'vs_B_L':res,'bootstrap_vs_B_L':boot,'transitions':{'K10':transitions(items,ranked,users,labels,10),'K20':transitions(items,ranked,users,labels,20)}}
            save[f'diff{seed}_top50']=ranked[:,:50]
        np.savez_compressed(out/f'test_rankings_L{L}.npz',**save)
        mean_metrics={k:float(np.mean([seedres[str(s)]['vs_B_L']['metrics'][k] for s in [202610065,202610066]])) for k in ALL}; mean_U=float(np.mean([seedres[str(s)]['vs_B_L']['U'] for s in [202610065,202610066]]))
        depths[str(L)]={'selection':{'update':upd,'eta':eta,'lock_mean_DEV_U':sel['mean_DEV_U']},'MSCA':{k:float(v) for k,v in msca_m.items()},'B_L':{k:float(v) for k,v in b_m.items()},'B_L_vs_MSCA':result_vs(msca_m,b_m),'Diffusion_seeds':seedres,'Diffusion_seed_mean_metrics':mean_metrics,'mean_Diffusion_U_vs_B_L':mean_U,'A':{'mean_items':float(A.sum(1).mean()),'median_items':float(np.median(A.sum(1))),'max_items':int(A.sum(1).max())},'rankings_sha256':sha(out/f'test_rankings_L{L}.npz')}
    # Relative to strict B100 after both depths are known.
    for L in [100,500]:
        d=depths[str(L)]; d['B_L_vs_strict_B100']=result_vs(strict_b100,d['B_L'])
        for seed in [202610065,202610066]: d['Diffusion_seeds'][str(seed)]['vs_strict_B100']=result_vs(strict_b100,d['Diffusion_seeds'][str(seed)]['vs_B_L']['metrics'])
    hist_path=ROOT/'docs/evidence/archive/robustness/baby_multiseed/baby_seed1000_canonical_test.json'; historical=json.load(open(hist_path))
    result={'status':'COMPLETE_TEST_EVALUATED_AFTER_LOCK','protocol_version':cfg['protocol_version'],'selection_lock_sha256':lock_hash,'selection_lock':lock['selection'],'strict_fit_test_users':int(len(users)),'depths':depths,'historical_full_training_canonical_seed1000':{'source':str(hist_path.relative_to(ROOT)),'MSCA':historical['MSCA']['metrics'],'COLIFTREC':historical['COLIFTREC']['metrics'],'COLIFTREC_vs_MSCA':historical['COLIFTREC']['vs_MSCA']},'elapsed_seconds':time.time()-t0,'input_hashes':{'asset_manifest':sha(asset/'manifest.json'),'config':sha(ROOT/'diffusion_experiments/configs/round4_baby.yaml'),'interaction':sha(data_cfg['resolved_paths']['interaction'])},'access':{'DEV_USED_FOR_SELECTION':True,'INTERNAL_USED_FOR_SELECTION':False,'CONFIRM_ACCESSED':False,'TEST_ACCESSED':True,'TEST_USED_FOR_SELECTION':False},'note':'Strict-FIT Round4 Test is same-protocol prototype evidence. Historical full-training canonical metrics are listed separately and are not used as the Round4 increment denominator.'}
    (out/'test_results.json').write_text(json.dumps(result,indent=2)+'\n'); return result

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--assets',required=True); ap.add_argument('--run-root',required=True); ap.add_argument('--analysis-dir',required=True); ap.add_argument('--out',required=True); ap.add_argument('--dry-run-dev',type=int,default=0); ap.add_argument('--open-test',action='store_true'); a=ap.parse_args()
    if bool(a.dry_run_dev)==bool(a.open_test): raise RuntimeError('choose exactly one of --dry-run-dev N or --open-test')
    cfg=yaml.safe_load((ROOT/'diffusion_experiments/configs/round4_baby.yaml').read_text()); asset=Path(a.assets); runroot=Path(a.run_root); analysis=Path(a.analysis_dir); out=Path(a.out)
    if not torch.cuda.is_available() or '5090' not in torch.cuda.get_device_name(0): raise RuntimeError('RTX5090 required')
    device=torch.device('cuda:0')
    if a.dry_run_dev:
        res=dry_run(cfg,asset,runroot,analysis,device,int(a.dry_run_dev)); out.mkdir(parents=True,exist_ok=True); (out/'dry_run.json').write_text(json.dumps(res,indent=2)+'\n'); print(json.dumps(res,sort_keys=True)); return
    if not bool(cfg['access'].get('test_authorized_by_user_after_lock')): raise RuntimeError('user Test authorization not recorded in config')
    if not (analysis/'selection_lock.json').exists(): raise RuntimeError('selection lock missing')
    res=run_test(cfg,asset,runroot,analysis,out,device); print(json.dumps({'status':res['status'],'lock':res['selection_lock_sha256'],'depths':{L:{'mean_U':d['mean_Diffusion_U_vs_B_L'],'B_vs_MSCA_U':d['B_L_vs_MSCA']['U']} for L,d in res['depths'].items()}},sort_keys=True))
if __name__=='__main__': main()
