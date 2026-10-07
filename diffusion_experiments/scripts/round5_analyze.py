from __future__ import annotations
import argparse,csv,gc,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from diffusion_experiments.modules.round5_common import *

def load_ck_model(fit,ckpt):
    _,teacher_state,_,_=load_teacher_model(ROOT/cfg_round5()['teacher_training_json'],fit); m,_,_=load_student_from_state(teacher_state,fit); z=torch.load(ckpt,map_location='cpu',weights_only=False); m.load_state_dict(z['state_dict'],strict=True); m.eval(); return m,z

def transitions(base,new,users,labels,k):
    inc=dec=0
    for r,u in enumerate(users):
        lab=labels[int(u)]; b=sum(int(x) in lab for x in base[r,:k]); n=sum(int(x) in lab for x in new[r,:k]); inc+=n>b; dec+=n<b
    return {'increase':int(inc),'decrease':int(dec),'net':int(inc-dec)}

def eval_pair(base,new,users,labels,boot_seed,cfg):
    bm=metrics_at(base,users,labels,ks=(10,20,50)); nm=metrics_at(new,users,labels,ks=(10,20,50)); return {'comparison':relative_result(bm,nm),'bootstrap':bootstrap_u(base,new,users,labels,int(cfg['bootstrap']['resamples']),boot_seed),'transitions':{'K10':transitions(base,new,users,labels,10),'K20':transitions(base,new,users,labels,20)}}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--assets',required=True); ap.add_argument('--run-root',required=True); ap.add_argument('--out',required=True); a=ap.parse_args(); asset=Path(a.assets); rr=Path(a.run_root); out=Path(a.out)
    if out.exists() and any(out.iterdir()): raise RuntimeError('refuse overwrite nonempty analysis dir')
    out.mkdir(parents=True,exist_ok=True); (out/'selected_exports').mkdir(); cfg=cfg_round5(); pdx=ROOT/cfg['protocol_dir']; fit=load_edges(pdx/'fit_edges.csv'); mon=load_edges(pdx/'monitor_edges.csv'); n_users=int(fit.userID.max())+1; n_items=int(np.load(ROOT/'data/baby/text_feat.npy',mmap_mode='r').shape[0]); histories=histories_from_fit(fit,n_users); side=prepare_side(histories,n_items); seeds=[int(x) for x in cfg['formal_seeds']]; branches=['B_CONT','D_CURR']; epochs=[int(x) for x in cfg['selection']['checkpoints']]
    # Validate training runs and paired base plans before reading evaluation labels.
    runs={};
    for s in seeds:
        p0=np.load(rr/f'B_CONT_seed{s}'/'base_plan.npz'); p1=np.load(rr/f'D_CURR_seed{s}'/'base_plan.npz')
        for k in ['base_neg','route','order']:
            if not np.array_equal(p0[k],p1[k]): raise RuntimeError(f'paired base plan mismatch seed={s} key={k}')
        for b in branches:
            r=json.load(open(rr/f'{b}_seed{s}'/'result.json'))
            if r['status'] not in ['COMPLETE']: raise RuntimeError(f'invalid run {b} {s}: {r["status"]}')
            if any(r['access'].values()): raise RuntimeError('training accessed eval labels')
            runs[f'{b}_seed{s}']=r
    # B0 monitor anchor comes from frozen teacher snapshot built before any monitor selection.
    t100=np.load(asset/'teacher_L100.npz'); mon_users=np.sort(mon.userID.unique()).astype(np.int64); mon_labels=label_sets_from_edges(mon,mon_users); b0_mon=metrics_at(t100['items'][mon_users],mon_users,mon_labels,ks=(10,20,50)); msca_mon=metrics_at(t100['raw_items'][mon_users],mon_users,mon_labels,ks=(10,20,50))
    grid={b:[] for b in branches}
    # Monitor-only checkpoint selection. Epoch0 is the exact frozen B0 ranking for both branches/seeds.
    for b in branches:
        for ep in epochs:
            sr={}
            for s in seeds:
                if ep==0:
                    met={k:float(v) for k,v in b0_mon.items()}; mh=json.load(open(rr/f'{b}_seed{s}'/'result.json'))['initial_model_hash']; cksha=sha(rr/f'{b}_seed{s}'/'checkpoints'/'epoch_00.pt')
                else:
                    ck=rr/f'{b}_seed{s}'/'checkpoints'/f'epoch_{ep:02d}.pt'; model,z=load_ck_model(fit,ck); emb=export_embeddings(model); cand=build_candidates_from_embeddings(emb,histories,side,int(cfg['L_eval']),n_users,n_items); met=metrics_at(cand['items'][mon_users],mon_users,mon_labels,ks=(10,20,50)); mh=z['model_hash']; cksha=sha(ck); model.cpu(); del model,cand,emb,z; torch.cuda.empty_cache(); gc.collect()
                rel=relative_result(b0_mon,met); sr[str(s)]={'metrics':{k:float(v) for k,v in met.items()},'vs_B0':rel,'model_hash':mh,'checkpoint_sha256':cksha}
            grid[b].append({'epoch':ep,'seed_results':sr,'mean_U_vs_B0':float(np.mean([sr[str(s)]['vs_B0']['U'] for s in seeds]))})
    selected={}
    for b in branches:
        g=sorted(grid[b],key=lambda x:(-float(x['mean_U_vs_B0']),int(x['epoch'])))
        best=g[0]; e0=next(x for x in grid[b] if int(x['epoch'])==0)
        if float(best['mean_U_vs_B0'])<=float(e0['mean_U_vs_B0'])+1e-15: best=e0
        selected[b]={'epoch':int(best['epoch']),'mean_monitor_U_vs_B0':float(best['mean_U_vs_B0']),'seed_results':best['seed_results']}
    (out/'monitor_grid.json').write_text(json.dumps({'B0_MSCA':msca_mon,'B0_CoLift':b0_mon,'grid':grid,'selected':selected},indent=2)+'\n')
    # Freeze exact selected embedding exports and L100 rankings BEFORE opening DEV/INTERNAL labels.
    frozen={}
    for b in branches:
        ep=int(selected[b]['epoch'])
        for s in seeds:
            key=f'{b}_seed{s}'; exp_path=out/'selected_exports'/f'{key}_embeddings.npz'; rank_path=out/'selected_exports'/f'{key}_L100.npz'
            if ep==0:
                z=np.load(asset/'teacher_embeddings.npz'); emb={k:z[k].astype(np.float32) for k in z.files}; cand=None
                # Candidate ranking must be exactly B0 frozen L100, not a regenerated tie order.
                t=np.load(asset/'teacher_L100.npz'); np.savez_compressed(rank_path,items=t['items'].astype(np.int32),raw_items=t['raw_items'].astype(np.int32),s0=t['s0'].astype(np.float32)); mh=runs[key]['initial_model_hash']; cksha=sha(rr/key/'checkpoints'/'epoch_00.pt')
            else:
                ck=rr/key/'checkpoints'/f'epoch_{ep:02d}.pt'; model,zck=load_ck_model(fit,ck); emb=export_embeddings(model); cand=build_candidates_from_embeddings(emb,histories,side,int(cfg['L_eval']),n_users,n_items); np.savez_compressed(rank_path,items=cand['items'].astype(np.int32),raw_items=cand['raw_items'].astype(np.int32),s0=cand['s0'].astype(np.float32)); mh=zck['model_hash']; cksha=sha(ck); model.cpu(); del model,cand,zck; torch.cuda.empty_cache(); gc.collect()
            np.savez_compressed(exp_path,**emb)
            frozen[key]={'branch':b,'seed':s,'epoch':ep,'model_hash':mh,'checkpoint_sha256':cksha,'embeddings_sha256':sha(exp_path),'ranking_sha256':sha(rank_path),'embeddings_file':str(exp_path),'ranking_file':str(rank_path)}
    lock={'status':'LOCKED_BEFORE_DEV_INTERNAL_TEST','protocol_version':cfg['protocol_version'],'selection':selected,'frozen_selected':frozen,'monitor_grid_sha256':sha(out/'monitor_grid.json'),'asset_audit_sha256':sha(asset/'audit.json'),'config_sha256':sha(ROOT/'diffusion_experiments/configs/round5_baby.yaml'),'teacher_checkpoint_sha256':cfg['expected_teacher_sha256'],'generator_result_hashes':{str(s):sha(rr/f'generator_seed{s}'/'result.json') for s in seeds},'training_result_hashes':{k:sha(rr/k/'result.json') for k in sorted(runs)},'access':{'MONITOR_USED_FOR_SELECTION':True,'DEV_USED_FOR_SELECTION':False,'INTERNAL_USED_FOR_SELECTION':False,'CONFIRM_ACCESSED':False,'TEST_ACCESSED':False,'TEST_USED_FOR_SELECTION':False},'test_authorization_record':cfg['access']}
    (out/'selection_lock.json').write_text(json.dumps(lock,indent=2)+'\n'); lock_hash=sha(out/'selection_lock.json')
    # Only after lock: open development labels (DEV validation x=1 and INTERNAL probe targets).
    dev_users=pd.read_csv(pdx/'dev_users.csv').userID.to_numpy(np.int64); internal_users=pd.read_csv(pdx/'internal_users.csv').userID.to_numpy(np.int64); probe=load_edges(pdx/'probe_edges.csv'); dev_labels=label_sets_x(side['cfg']['resolved_paths']['interaction'],dev_users,1); int_labels=label_sets_from_edges(probe,internal_users)
    b0_items=t100['items'].astype(np.int32); b0_raw=t100['raw_items'].astype(np.int32)
    split_results={}
    for split,users,labels in [('DEV',dev_users,dev_labels),('INTERNAL',internal_users,int_labels)]:
        sr={'B0_MSCA':{k:float(v) for k,v in metrics_at(b0_raw[users],users,labels,ks=(10,20,50)).items()},'B0_COLIFT':{k:float(v) for k,v in metrics_at(b0_items[users],users,labels,ks=(10,20,50)).items()},'seeds':{}}
        for s in seeds:
            bz=np.load(out/'selected_exports'/f'B_CONT_seed{s}_L100.npz'); dz=np.load(out/'selected_exports'/f'D_CURR_seed{s}_L100.npz'); bi=bz['items'][users].astype(np.int32); di=dz['items'][users].astype(np.int32)
            bm=metrics_at(bi,users,labels,ks=(10,20,50)); dm=metrics_at(di,users,labels,ks=(10,20,50)); b0m=sr['B0_COLIFT']
            pair=eval_pair(bi,di,users,labels,int(cfg['bootstrap']['seed'])+s+(0 if split=='DEV' else 1000),cfg)
            sr['seeds'][str(s)]={'B_CONT':{k:float(v) for k,v in bm.items()},'D_CURR':{k:float(v) for k,v in dm.items()},'D_vs_B_CONT':pair,'B_CONT_vs_B0':relative_result(b0m,bm),'D_CURR_vs_B0':relative_result(b0m,dm)}
        sr['mean_D_vs_B_CONT_U']=float(np.mean([sr['seeds'][str(s)]['D_vs_B_CONT']['comparison']['U'] for s in seeds])); sr['mean_B_CONT_vs_B0_U']=float(np.mean([sr['seeds'][str(s)]['B_CONT_vs_B0']['U'] for s in seeds])); sr['mean_D_CURR_vs_B0_U']=float(np.mean([sr['seeds'][str(s)]['D_CURR_vs_B0']['U'] for s in seeds])); sr['all_seed_D_vs_B_positive']=all(sr['seeds'][str(s)]['D_vs_B_CONT']['comparison']['U']>0 for s in seeds); sr['all_seed_D_vs_B0_positive']=all(sr['seeds'][str(s)]['D_CURR_vs_B0']['U']>0 for s in seeds)
        split_results[split]=sr
    dev=split_results['DEV']; inte=split_results['INTERNAL']; target=float(cfg['target_U'])
    if dev['mean_D_vs_B_CONT_U']>=target and inte['mean_D_vs_B_CONT_U']>=target and dev['all_seed_D_vs_B_positive'] and inte['all_seed_D_vs_B_positive'] and dev['mean_D_CURR_vs_B0_U']>0 and inte['mean_D_CURR_vs_B0_U']>0: classification='BABY_DEVELOPMENT_TARGET_MET'
    elif dev['mean_D_vs_B_CONT_U']<=0 and inte['mean_D_vs_B_CONT_U']<=0: classification='NO_POSITIVE_INCREMENT'
    elif np.sign(dev['mean_D_vs_B_CONT_U'])!=np.sign(inte['mean_D_vs_B_CONT_U']) or dev['all_seed_D_vs_B_positive']!=inte['all_seed_D_vs_B_positive']: classification='SPLIT_OR_SEED_UNSTABLE'
    elif (dev['mean_D_vs_B_CONT_U']+inte['mean_D_vs_B_CONT_U'])/2>0: classification='POSITIVE_BELOW_TARGET'
    else: classification='NO_POSITIVE_INCREMENT'
    result={'status':'COMPLETE_LOCKED_DEV_INTERNAL_EVALUATED','protocol_version':cfg['protocol_version'],'selection_lock_sha256':lock_hash,'selection':selected,'monitor_grid':grid,'DEV':dev,'INTERNAL':inte,'classification':classification,'target_U':target,'training_runs':runs,'access':{'MONITOR_USED_FOR_SELECTION':True,'DEV_USED_FOR_SELECTION':False,'INTERNAL_USED_FOR_SELECTION':False,'CONFIRM_ACCESSED':False,'TEST_ACCESSED':False,'TEST_USED_FOR_SELECTION':False},'historical_exposure_note':'DEV/INTERNAL are development evidence. Baby Test has historical exposure but has not been read by Round5 yet.'}
    (out/'dev_internal_results.json').write_text(json.dumps(result,indent=2)+'\n')
    rows=[]
    for split in ['DEV','INTERNAL']:
        for s in seeds:
            x=split_results[split]['seeds'][str(s)]; row={'split':split,'seed':s,'B_CONT_epoch':selected['B_CONT']['epoch'],'D_CURR_epoch':selected['D_CURR']['epoch'],'D_vs_B_CONT_U':x['D_vs_B_CONT']['comparison']['U'],'B_CONT_vs_B0_U':x['B_CONT_vs_B0']['U'],'D_CURR_vs_B0_U':x['D_CURR_vs_B0']['U']}
            for name in ['B_CONT','D_CURR']:
                for k in ALL: row[f'{name}_{k}']=x[name][k]
            rows.append(row)
    with open(out/'dev_internal_results.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(json.dumps({'status':result['status'],'lock':lock_hash,'selection':{b:selected[b]['epoch'] for b in branches},'DEV_U':dev['mean_D_vs_B_CONT_U'],'INTERNAL_U':inte['mean_D_vs_B_CONT_U'],'classification':classification},sort_keys=True))
if __name__=='__main__': main()
