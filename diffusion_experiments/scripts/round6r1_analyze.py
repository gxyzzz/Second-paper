from __future__ import annotations
import argparse,csv,gc,json,subprocess,sys,yaml
from pathlib import Path
import numpy as np,pandas as pd,torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'src'))
from diffusion_experiments.modules.round6r1_train_common import cfg_r1,fresh_student,load_edges,histories_from_fit,sha
from diffusion_experiments.modules.round5_common import prepare_side,build_candidates_from_embeddings,export_embeddings,relative_result,bootstrap_u,label_sets_from_edges,label_sets_x,metrics_at,ALL
from pipelines.dataset_config import load_dataset_config

def load_model(seed,fit,ckpt,tmp):
    m,_,_=fresh_student(seed,fit,tmp); z=torch.load(ckpt,map_location='cpu',weights_only=False); m.load_state_dict(z['state_dict'],strict=True); m.eval(); return m,z

def transitions(base,new,users,labels,k):
    inc=dec=0
    for r,u in enumerate(users):
        lab=labels[int(u)]; b=sum(int(x) in lab for x in base[r,:k]); n=sum(int(x) in lab for x in new[r,:k]); inc+=n>b; dec+=n<b
    return {'increase':int(inc),'decrease':int(dec),'net':int(inc-dec)}

def eval_pair(base,new,users,labels,seed,c):
    bm=metrics_at(base,users,labels,ks=(10,20,50)); dm=metrics_at(new,users,labels,ks=(10,20,50))
    return {'comparison':relative_result(bm,dm),'bootstrap':bootstrap_u(base,new,users,labels,int(c['bootstrap']['resamples']),int(seed)),'transitions':{'K10':transitions(base,new,users,labels,10),'K20':transitions(base,new,users,labels,20)}}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run-root',required=True); ap.add_argument('--out',required=True); a=ap.parse_args(); rr=Path(a.run_root); out=Path(a.out)
    if out.exists() and any(out.iterdir()): raise RuntimeError('refuse overwrite nonempty analysis dir')
    out.mkdir(parents=True,exist_ok=True); (out/'selected_exports').mkdir(); c=cfg_r1(); pdx=ROOT/c['protocol_dir']; fit=load_edges(pdx/'fit_edges.csv'); n_users=int(fit.userID.max())+1; n_items=int(np.load(ROOT/'data/baby/text_feat.npy',mmap_mode='r').shape[0]); histories=histories_from_fit(fit,n_users); side=prepare_side(histories,n_items)
    seeds=[int(x) for x in c['student_seeds']]; runs={}; frozen={}
    for s in seeds:
        b=json.load(open(rr/f'B_seed{s}'/'result.json')); d=json.load(open(rr/f'D_seed{s}'/'result.json')); runs[f'B_seed{s}']=b; runs[f'D_seed{s}']=d
        if b['status']!='COMPLETE' or d['status']!='COMPLETE': raise RuntimeError(f'incomplete student run seed={s}')
        if b['base_plan_sha256']!=d['base_plan_sha256']: raise RuntimeError(f'base plan mismatch seed={s}')
        if b['initial_model_hash']!=d['initial_model_hash']: raise RuntimeError(f'initial model mismatch seed={s}')
        if d['aux_nonzero_events']<=0: raise RuntimeError(f'D auxiliary inactive seed={s}')
        if any(v for k,v in b['access'].items() if k!='MONITOR_USED_FOR_SELECTION') or any(v for k,v in d['access'].items() if k!='MONITOR_USED_FOR_SELECTION'): raise RuntimeError('student accessed locked labels')
    if not runs['B_seed999'].get('baseline_reproduction',{}).get('pass',False): raise RuntimeError('B999 reproduction not healthy')
    # Export exact selected checkpoint rankings before any DEV/INTERNAL label read.
    for s in seeds:
        for branch in ['B','D']:
            key=f'{branch}_seed{s}'; r=runs[key]; ck=Path(r['best_checkpoint'])
            if sha(ck)!=r['best_checkpoint_sha256']: raise RuntimeError(f'checkpoint hash mismatch {key}')
            m,z=load_model(s,fit,ck,out/'tmp'/key); emb=export_embeddings(m); cand=build_candidates_from_embeddings(emb,histories,side,100,n_users,n_items)
            ep=out/'selected_exports'/f'{key}_embeddings.npz'; rp=out/'selected_exports'/f'{key}_L100.npz'; np.savez_compressed(ep,**emb); np.savez_compressed(rp,items=cand['items'].astype(np.int32),raw_items=cand['raw_items'].astype(np.int32),s0=cand['s0'].astype(np.float32))
            frozen[key]={'seed':s,'branch':branch,'best_epoch':r['best_epoch'],'best_monitor_R20':r['best_monitor_R20'],'checkpoint':str(ck),'checkpoint_sha256':r['best_checkpoint_sha256'],'model_hash':r['best_model_hash'],'embeddings_file':str(ep.resolve()),'embeddings_sha256':sha(ep),'ranking_file':str(rp.resolve()),'ranking_sha256':sha(rp)}
            m.cpu(); del m,z,emb,cand; torch.cuda.empty_cache(); gc.collect()
    risks={}; gens={}
    for s,gseed in zip(seeds,[202610081,202610082]):
        pp=rr/f'risk_seed{gseed}'/'preflight.json'; p=json.load(open(pp));
        if not p['gate_pass']: raise RuntimeError(f'risk gate failed {gseed}')
        risks[str(gseed)]={'preflight_sha256':sha(pp),'risk_cache_sha256':p['risk_cache_sha256'],'status':p['status'],'personalization_status':p['personalization_status']}
        gp=ROOT/c['round6_generator081']/'generator.pt' if gseed==202610081 else rr/f'generator_seed{gseed}'/'generator.pt'; gens[str(gseed)]={'path':str(gp),'sha256':sha(gp)}
    lock={'status':'LOCKED_BEFORE_DEV_INTERNAL_TEST','protocol_version':c['protocol_version'],'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'config_sha256':sha(ROOT/'diffusion_experiments/configs/round6r1_baby.yaml'),'guide_sha256':sha(ROOT/'diffusion_experiments/ADVISOR_EXPERIMENT_GUIDE.md'),'reference_sha256':sha(ROOT/c['round6_assets']/'reference_L100.npz'),'frozen_selected':frozen,'risk':risks,'generators':gens,'training_result_hashes':{k:sha(rr/k/'result.json') for k in sorted(runs)},'access':{'MONITOR_USED_FOR_SELECTION':True,'DEV_USED_FOR_SELECTION':False,'INTERNAL_USED_FOR_SELECTION':False,'CONFIRM_ACCESSED':False,'TEST_ACCESSED':False,'TEST_USED_FOR_SELECTION':False},'test_authorization':c['access']}
    (out/'selection_lock.json').write_text(json.dumps(lock,indent=2)+'\n'); lock_hash=sha(out/'selection_lock.json')
    # Open development labels only after the lock exists.
    dev_users=pd.read_csv(pdx/'dev_users.csv').userID.to_numpy(np.int64); internal_users=pd.read_csv(pdx/'internal_users.csv').userID.to_numpy(np.int64); probe=load_edges(pdx/'probe_edges.csv'); dcfg=load_dataset_config('baby'); dev_labels=label_sets_x(dcfg['resolved_paths']['interaction'],dev_users,1); int_labels=label_sets_from_edges(probe,internal_users)
    teacher=np.load(ROOT/'diffusion_experiments/runs/round5/assets_formal_v4/teacher_L100.npz'); teacher_items=teacher['items'].astype(np.int32); teacher_raw=teacher['raw_items'].astype(np.int32)
    split_results={}
    for split,users,labels in [('DEV',dev_users,dev_labels),('INTERNAL',internal_users,int_labels)]:
        sr={'OLD_TEACHER_MSCA':{k:float(v) for k,v in metrics_at(teacher_raw[users],users,labels,ks=(10,20,50)).items()},'OLD_TEACHER_COLIFT':{k:float(v) for k,v in metrics_at(teacher_items[users],users,labels,ks=(10,20,50)).items()},'seeds':{}}
        for s in seeds:
            bz=np.load(out/'selected_exports'/f'B_seed{s}_L100.npz'); dz=np.load(out/'selected_exports'/f'D_seed{s}_L100.npz'); bi=bz['items'][users].astype(np.int32); di=dz['items'][users].astype(np.int32); bm=metrics_at(bi,users,labels,ks=(10,20,50)); dm=metrics_at(di,users,labels,ks=(10,20,50)); pair=eval_pair(bi,di,users,labels,int(c['bootstrap']['seed'])+s+(0 if split=='DEV' else 1000),c)
            sr['seeds'][str(s)]={'B':{k:float(v) for k,v in bm.items()},'D':{k:float(v) for k,v in dm.items()},'D_vs_B':pair,'B_vs_old_teacher':relative_result(sr['OLD_TEACHER_COLIFT'],bm),'D_vs_old_teacher':relative_result(sr['OLD_TEACHER_COLIFT'],dm)}
        us=[sr['seeds'][str(s)]['D_vs_B']['comparison']['U'] for s in seeds]; sr['mean_D_vs_B_U']=float(np.mean(us)); sr['all_seed_D_vs_B_positive']=all(x>0 for x in us); split_results[split]=sr
    dev=split_results['DEV']; inte=split_results['INTERNAL']; target=float(c['target_U']); seed_us=[dev['seeds'][str(s)]['D_vs_B']['comparison']['U'] for s in seeds]+[inte['seeds'][str(s)]['D_vs_B']['comparison']['U'] for s in seeds]
    if dev['mean_D_vs_B_U']>=target and inte['mean_D_vs_B_U']>=target and all(x>0 for x in seed_us): effect='BABY_DEVELOPMENT_TARGET_MET'
    elif np.sign(dev['mean_D_vs_B_U'])!=np.sign(inte['mean_D_vs_B_U']) or not (dev['all_seed_D_vs_B_positive']==inte['all_seed_D_vs_B_positive']): effect='UNSTABLE'
    elif dev['mean_D_vs_B_U']<=0 and inte['mean_D_vs_B_U']<=0: effect='NO_INCREMENT'
    elif (dev['mean_D_vs_B_U']+inte['mean_D_vs_B_U'])/2>0: effect='POSITIVE_BELOW_TARGET'
    else: effect='NO_INCREMENT'
    result={'status':'COMPLETE_LOCKED_DEV_INTERNAL_EVALUATED','protocol_version':c['protocol_version'],'selection_lock_sha256':lock_hash,'DEV':dev,'INTERNAL':inte,'effect_classification':effect,'risk_classification':'RISK_DIRECTION_PASS','personalization_classification':'PERSONALIZATION_UNRESOLVED' if any(x['personalization_status']=='PERSONALIZATION_UNRESOLVED' for x in risks.values()) else 'DIRECTIONAL_DIAGNOSTIC','runs':runs,'access':{'MONITOR_USED_FOR_SELECTION':True,'DEV_USED_FOR_SELECTION':False,'INTERNAL_USED_FOR_SELECTION':False,'CONFIRM_ACCESSED':False,'TEST_ACCESSED':False,'TEST_USED_FOR_SELECTION':False}}
    (out/'dev_internal_results.json').write_text(json.dumps(result,indent=2)+'\n')
    rows=[]
    for split in ['DEV','INTERNAL']:
        for s in seeds:
            x=split_results[split]['seeds'][str(s)]; row={'split':split,'seed':s,'B_best_epoch':runs[f'B_seed{s}']['best_epoch'],'D_best_epoch':runs[f'D_seed{s}']['best_epoch'],'D_vs_B_U':x['D_vs_B']['comparison']['U']}
            for name in ['B','D']:
                for k in ALL: row[f'{name}_{k}']=x[name][k]
            rows.append(row)
    with open(out/'dev_internal_results.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(json.dumps({'status':result['status'],'lock':lock_hash,'DEV_U':dev['mean_D_vs_B_U'],'INTERNAL_U':inte['mean_D_vs_B_U'],'effect':effect},sort_keys=True))
if __name__=='__main__': main()
