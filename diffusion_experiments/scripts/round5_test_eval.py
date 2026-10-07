from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from diffusion_experiments.modules.round5_common import *

def transitions(base,new,users,labels,k):
    inc=dec=0
    for r,u in enumerate(users):
        lab=labels[int(u)]; b=sum(int(x) in lab for x in base[r,:k]); n=sum(int(x) in lab for x in new[r,:k]); inc+=n>b; dec+=n<b
    return {'increase':int(inc),'decrease':int(dec),'net':int(inc-dec)}

def pair(base,new,users,labels,seed,cfg):
    bm=metrics_at(base,users,labels,ks=(10,20,50)); nm=metrics_at(new,users,labels,ks=(10,20,50)); return {'comparison':relative_result(bm,nm),'bootstrap':bootstrap_u(base,new,users,labels,int(cfg['bootstrap']['resamples']),seed),'transitions':{'K10':transitions(base,new,users,labels,10),'K20':transitions(base,new,users,labels,20)}}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--assets',required=True); ap.add_argument('--analysis',required=True); ap.add_argument('--out',required=True); ap.add_argument('--open-test',action='store_true'); a=ap.parse_args(); cfg=cfg_round5(); asset=Path(a.assets); analysis=Path(a.analysis); out=Path(a.out)
    if not a.open_test: raise RuntimeError('Test requires explicit --open-test')
    if not bool(cfg['access']['test_authorized_by_user_after_lock']): raise RuntimeError('user Test authorization missing')
    lockp=analysis/'selection_lock.json'; lock=json.load(open(lockp)); lock_hash=sha(lockp)
    if lock['access']['TEST_ACCESSED'] or lock['access']['TEST_USED_FOR_SELECTION']: raise RuntimeError('invalid pre-Test lock')
    # Bind exact rankings to pre-Test lock before reading labels.
    for key,x in lock['frozen_selected'].items():
        if sha(Path(x['ranking_file']))!=x['ranking_sha256'] or sha(Path(x['embeddings_file']))!=x['embeddings_sha256']: raise RuntimeError(f'frozen export hash mismatch {key}')
    t=np.load(asset/'teacher_L100.npz'); b0_raw=t['raw_items'].astype(np.int32); b0=t['items'].astype(np.int32); n_users=b0.shape[0]; users=np.arange(n_users,dtype=np.int64); data_cfg=load_dataset_config('baby'); labels=label_sets_x(data_cfg['resolved_paths']['interaction'],users,2)
    missing=[int(u) for u in users if not labels[int(u)]]
    if missing: raise RuntimeError(f'Test labels missing users={len(missing)}')
    msca=metrics_at(b0_raw,users,labels,ks=(10,20,50)); colift=metrics_at(b0,users,labels,ks=(10,20,50)); seeds=[int(x) for x in cfg['formal_seeds']]; sr={}
    for s in seeds:
        bz=np.load(analysis/'selected_exports'/f'B_CONT_seed{s}_L100.npz'); dz=np.load(analysis/'selected_exports'/f'D_CURR_seed{s}_L100.npz'); bi=bz['items'].astype(np.int32); di=dz['items'].astype(np.int32); bm=metrics_at(bi,users,labels,ks=(10,20,50)); dm=metrics_at(di,users,labels,ks=(10,20,50)); p=pair(bi,di,users,labels,int(cfg['bootstrap']['seed'])+200000+s,cfg)
        sr[str(s)]={'B_CONT':{k:float(v) for k,v in bm.items()},'D_CURR':{k:float(v) for k,v in dm.items()},'D_vs_B_CONT':p,'B_CONT_vs_B0_COLIFT':relative_result(colift,bm),'D_CURR_vs_B0_COLIFT':relative_result(colift,dm),'B_CONT_vs_B0_MSCA':relative_result(msca,bm),'D_CURR_vs_B0_MSCA':relative_result(msca,dm)}
    def mean_metrics(name): return {k:float(np.mean([sr[str(s)][name][k] for s in seeds])) for k in ALL}
    bmmean=mean_metrics('B_CONT'); dmmean=mean_metrics('D_CURR')
    histp=ROOT/'docs/evidence/archive/robustness/baby_multiseed/baby_seed1000_canonical_test.json'; historical=json.load(open(histp))
    result={'status':'COMPLETE_TEST_EVALUATED_AFTER_LOCK','protocol_version':cfg['protocol_version'],'selection_lock_sha256':lock_hash,'selection':lock['selection'],'test_users':int(n_users),'B0_MSCA':{k:float(v) for k,v in msca.items()},'B0_COLIFT':{k:float(v) for k,v in colift.items()},'B0_COLIFT_vs_MSCA':relative_result(msca,colift),'seeds':sr,'B_CONT_seed_mean_metrics':bmmean,'D_CURR_seed_mean_metrics':dmmean,'mean_D_vs_B_CONT_U':float(np.mean([sr[str(s)]['D_vs_B_CONT']['comparison']['U'] for s in seeds])),'mean_B_CONT_vs_B0_COLIFT_U':float(np.mean([sr[str(s)]['B_CONT_vs_B0_COLIFT']['U'] for s in seeds])),'mean_D_CURR_vs_B0_COLIFT_U':float(np.mean([sr[str(s)]['D_CURR_vs_B0_COLIFT']['U'] for s in seeds])),'historical_full_training_canonical_seed1000':{'source':str(histp.relative_to(ROOT)),'MSCA':historical['MSCA']['metrics'],'COLIFTREC':historical['COLIFTREC']['metrics'],'COLIFTREC_vs_MSCA':historical['COLIFTREC']['vs_MSCA']},'access':{'MONITOR_USED_FOR_SELECTION':True,'DEV_USED_FOR_SELECTION':False,'INTERNAL_USED_FOR_SELECTION':False,'CONFIRM_ACCESSED':False,'TEST_ACCESSED':True,'TEST_USED_FOR_SELECTION':False},'note':'Round5 Test opened once after monitor lock and development evaluation by explicit user authorization. Baby Test has historical exposure; this is not fresh external confirmation. Historical full-training metrics use a different training graph and are comparison context only.'}
    out.mkdir(parents=True,exist_ok=False); (out/'test_results.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({'status':result['status'],'lock':lock_hash,'D_vs_B_CONT_U':result['mean_D_vs_B_CONT_U'],'B_CONT_vs_B0_U':result['mean_B_CONT_vs_B0_COLIFT_U'],'D_CURR_vs_B0_U':result['mean_D_CURR_vs_B0_COLIFT_U']},sort_keys=True))
if __name__=='__main__': main()
