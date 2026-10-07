from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'src'))
from diffusion_experiments.modules.round7_common import (
    ALL_METRICS, PRIMARY, cfg_round7, label_sets, load_interactions, metrics_at,
    per_user_metric_arrays, relative_u, sha, transition_counts,
)


def avg_metrics(ds):
    return {k:float(np.mean([d[k] for d in ds])) for k in ALL_METRICS}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--lock-dir',required=True); ap.add_argument('--out',required=True); ap.add_argument('--dry-run-validation',action='store_true'); a=ap.parse_args()
    lockdir=Path(a.lock_dir); out=Path(a.out)
    if out.exists() and any(out.iterdir()): raise RuntimeError('refuse overwrite nonempty Test output')
    out.mkdir(parents=True,exist_ok=True)
    lockp=lockdir/'selection_lock.json'; lock=json.loads(lockp.read_text()); cfg=cfg_round7()
    if not lock.get('test_authorized'): raise RuntimeError('Test not authorized in lock')
    split_label=1 if a.dry_run_validation else 2
    df=load_interactions(); users,sets=label_sets(df,split_label)
    if len(users)!=19445: raise RuntimeError(f'expected 19445 evaluation users, got {len(users)}')
    cells={}; m0_by_b={}; m1_by_b={}; arrays_new=[]; arrays_base=[]
    for cell,rel in lock['ranking_files'].items():
        p=ROOT/rel
        if sha(p)!=lock['ranking_sha256'][cell]: raise RuntimeError(f'locked ranking hash mismatch {cell}')
        z=np.load(p); zu=z['users'].astype(np.int64)
        if not np.array_equal(zu,users): raise RuntimeError(f'user order mismatch {cell}')
        m0=z['m0'].astype(np.int32); m1=z['m1'].astype(np.int32); m2=z['m2'].astype(np.int32)
        b=int(lock['cells'][cell]['backbone_seed']); d=int(lock['cells'][cell]['diffusion_seed'])
        m0m=metrics_at(m0,users,sets); m1m=metrics_at(m1,users,sets); m2m=metrics_at(m2,users,sets)
        U=relative_u(m2m,m1m)
        cells[cell]={'backbone_seed':b,'diffusion_seed':d,'M0':m0m,'M1':m1m,'M2':m2m,'U_M2_vs_M1':U,'K10':transition_counts(m1,m2,users,sets,10),'K20':transition_counts(m1,m2,users,sets,20),'changed_users':int(np.sum(np.any(m2!=m1,axis=1))),'changed_slots':int(np.sum(m2!=m1))}
        if b not in m0_by_b: m0_by_b[b]=m0m; m1_by_b[b]=m1m
        else:
            if m0_by_b[b]!=m0m or m1_by_b[b]!=m1m: raise RuntimeError('baseline changed across D seeds')
        arrays_new.append(per_user_metric_arrays(m2,users,sets)); arrays_base.append(per_user_metric_arrays(m1,users,sets))
    main_m0=avg_metrics([m0_by_b[999],m0_by_b[1000]])
    main_m1=avg_metrics([m1_by_b[999],m1_by_b[1000]])
    # average two D per backbone, then two backbones == equal average of four cells
    main_m2=avg_metrics([x['M2'] for x in cells.values()])
    main_u=relative_u(main_m2,main_m1)
    resamples=int(cfg['bootstrap']['resamples']); seed=int(cfg['bootstrap']['seed'])+(1 if split_label==2 else 0)
    rng=np.random.default_rng(seed); vals=np.empty(resamples,np.float64); n=len(users)
    for r in range(resamples):
        ix=rng.integers(0,n,n); rels=[]
        for k in PRIMARY:
            nb=float(np.mean([a[k][ix].mean() for a in arrays_new])); bb=float(np.mean([a[k][ix].mean() for a in arrays_base])); rels.append((nb-bb)/bb)
        vals[r]=np.mean(rels)
    boot={'U':main_u,'ci95':[float(np.quantile(vals,.025)),float(np.quantile(vals,.975))],'positive_fraction':float(np.mean(vals>0)),'resamples':resamples,'seed':seed}
    cell_us=[x['U_M2_vs_M1'] for x in cells.values()]
    primary_dirs={k:int(np.sign(main_m2[k]-main_m1[k])) for k in PRIMARY}; all_dirs={k:int(np.sign(main_m2[k]-main_m1[k])) for k in ALL_METRICS}
    target=bool(main_u>=float(cfg['target_U']) and all(np.mean([cells[f'b{b}_d202610101']['U_M2_vs_M1'],cells[f'b{b}_d202610102']['U_M2_vs_M1']])>0 for b in [999,1000]) and sum(u>0 for u in cell_us)>=3)
    status='DRY_RUN_VALIDATION' if a.dry_run_validation else 'COMPLETE_LOCKED_TEST'
    result={'status':status,'protocol_version':cfg['protocol_version'],'selection_lock_sha256':sha(lockp),'selected_eta':lock['selected_eta'],'split':'VALIDATION' if split_label==1 else 'TEST','users':len(users),'main':{'M0':main_m0,'M1':main_m1,'M2':main_m2,'U_M2_vs_M1':main_u,'bootstrap':boot,'primary_direction':primary_dirs,'all6_direction':all_dirs,'primary_positive_count':int(sum(v>0 for v in primary_dirs.values())),'all6_positive_count':int(sum(v>0 for v in all_dirs.values())),'target_1pct_met':target},'cells':cells,'cell_positive_count':int(sum(u>0 for u in cell_us)),'expansion_decision_before_test':lock['expansion_decision_before_test'],'TEST_USED_FOR_SELECTION':False,'NO_POST_TEST_TUNING':True}
    (out/'test_results.json' if split_label==2 else out/'validation_dryrun.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':status,'eta':lock['selected_eta'],'main_U':main_u,'ci95':boot['ci95'],'positive_cells':result['cell_positive_count'],'target':target},sort_keys=True))
if __name__=='__main__': main()
