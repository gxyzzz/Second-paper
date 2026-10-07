from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'src'))
from diffusion_experiments.modules.round6r1_train_common import cfg_r1,sha
from diffusion_experiments.modules.round5_common import metrics_at,relative_result,bootstrap_u,label_sets_x,ALL
from pipelines.dataset_config import load_dataset_config

def transitions(base,new,users,labels,k):
    inc=dec=0
    for r,u in enumerate(users):
        lab=labels[int(u)]; b=sum(int(x) in lab for x in base[r,:k]); n=sum(int(x) in lab for x in new[r,:k]); inc+=n>b; dec+=n<b
    return {'increase':int(inc),'decrease':int(dec),'net':int(inc-dec)}

def pair(base,new,users,labels,seed,c):
    bm=metrics_at(base,users,labels,ks=(10,20,50)); dm=metrics_at(new,users,labels,ks=(10,20,50))
    return {'comparison':relative_result(bm,dm),'bootstrap':bootstrap_u(base,new,users,labels,int(c['bootstrap']['resamples']),int(seed)),'transitions':{'K10':transitions(base,new,users,labels,10),'K20':transitions(base,new,users,labels,20)}}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--analysis',required=True); ap.add_argument('--out',required=True); ap.add_argument('--open-test',action='store_true'); a=ap.parse_args(); analysis=Path(a.analysis); out=Path(a.out); c=cfg_r1()
    if not a.open_test: raise RuntimeError('Test requires explicit --open-test')
    if not bool(c['access']['test_authorized_by_user_after_lock']): raise RuntimeError('locked Test authorization missing')
    lockp=analysis/'selection_lock.json'; lock=json.load(open(lockp)); lock_hash=sha(lockp)
    if lock['access']['TEST_ACCESSED'] or lock['access']['TEST_USED_FOR_SELECTION']: raise RuntimeError('invalid pre-Test lock')
    for key,x in lock['frozen_selected'].items():
        if sha(Path(x['ranking_file']))!=x['ranking_sha256'] or sha(Path(x['embeddings_file']))!=x['embeddings_sha256']: raise RuntimeError(f'frozen export hash mismatch {key}')
    dcfg=load_dataset_config('baby'); teacher=np.load(ROOT/'diffusion_experiments/runs/round5/assets_formal_v4/teacher_L100.npz'); old_msca=teacher['raw_items'].astype(np.int32); old_colift=teacher['items'].astype(np.int32); n_users=old_colift.shape[0]; users=np.arange(n_users,dtype=np.int64); labels=label_sets_x(dcfg['resolved_paths']['interaction'],users,2)
    if any(not labels[int(u)] for u in users): raise RuntimeError('missing Test labels')
    om=metrics_at(old_msca,users,labels,ks=(10,20,50)); oc=metrics_at(old_colift,users,labels,ks=(10,20,50)); seeds=[int(x) for x in c['student_seeds']]; sr={}
    for s in seeds:
        bz=np.load(analysis/'selected_exports'/f'B_seed{s}_L100.npz'); dz=np.load(analysis/'selected_exports'/f'D_seed{s}_L100.npz'); bi=bz['items'].astype(np.int32); di=dz['items'].astype(np.int32); bm=metrics_at(bi,users,labels,ks=(10,20,50)); dm=metrics_at(di,users,labels,ks=(10,20,50)); p=pair(bi,di,users,labels,int(c['bootstrap']['seed'])+200000+s,c)
        sr[str(s)]={'B':{k:float(v) for k,v in bm.items()},'D':{k:float(v) for k,v in dm.items()},'D_vs_B':p,'B_vs_old_teacher':relative_result(oc,bm),'D_vs_old_teacher':relative_result(oc,dm),'B_vs_old_MSCA':relative_result(om,bm),'D_vs_old_MSCA':relative_result(om,dm)}
    mean_u=float(np.mean([sr[str(s)]['D_vs_B']['comparison']['U'] for s in seeds])); mean_b={k:float(np.mean([sr[str(s)]['B'][k] for s in seeds])) for k in ALL}; mean_d={k:float(np.mean([sr[str(s)]['D'][k] for s in seeds])) for k in ALL}
    histp=ROOT/'docs/evidence/archive/robustness/baby_multiseed/baby_seed1000_canonical_test.json'; hist=json.load(open(histp))
    result={'status':'COMPLETE_TEST_EVALUATED_AFTER_LOCK','protocol_version':c['protocol_version'],'selection_lock_sha256':lock_hash,'test_users':int(n_users),'OLD_TEACHER_MSCA':{k:float(v) for k,v in om.items()},'OLD_TEACHER_COLIFT':{k:float(v) for k,v in oc.items()},'OLD_TEACHER_COLIFT_vs_MSCA':relative_result(om,oc),'seeds':sr,'B_seed_mean_metrics':mean_b,'D_seed_mean_metrics':mean_d,'mean_D_vs_B_U':mean_u,'historical_full_training_canonical_seed1000':{'source':str(histp.relative_to(ROOT)),'MSCA':hist['MSCA']['metrics'],'COLIFTREC':hist['COLIFTREC']['metrics'],'COLIFTREC_vs_MSCA':hist['COLIFTREC']['vs_MSCA']},'access':{'MONITOR_USED_FOR_SELECTION':True,'DEV_USED_FOR_SELECTION':False,'INTERNAL_USED_FOR_SELECTION':False,'CONFIRM_ACCESSED':False,'TEST_ACCESSED':True,'TEST_USED_FOR_SELECTION':False},'authorization_note':c['access']['test_authorization_note'],'historical_exposure_note':'Baby Test was already exposed in earlier rounds; this locked evaluation is development evidence, not fresh external confirmation.'}
    out.mkdir(parents=True,exist_ok=False); (out/'test_results.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({'status':result['status'],'lock':lock_hash,'mean_D_vs_B_U':mean_u},sort_keys=True))
if __name__=='__main__': main()
