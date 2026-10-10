from __future__ import annotations
import json
import numpy as np
import torch
from diffusion_experiments.round23_hardness_matched_diffusion import round23_core as c

LOG_EPOCHS={9,10,19,20,29,30,59}

def load_result(seed,variant): return json.loads((c.EVID/f'ROUND23_{variant}_SEED{seed}.json').read_text())

def save_best(model,net,seed,variant,epoch,score):
    p=c.OUT/f'seed{seed}/{variant}/best.pt';p.parent.mkdir(parents=True,exist_ok=True)
    obj={'protocol':c.PROTOCOL,'seed':seed,'variant':variant,'epoch':int(epoch),'full_colift_R20':float(score),'model_state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'TEST_ACCESSED':False}
    if net is not None: obj['diffusion_state_dict']={k:v.detach().cpu() for k,v in net.state_dict().items()}
    torch.save(obj,p);return p

def fairness(seed):
    r={v:load_result(seed,v) for v in c.VARIANTS};epochs=range(c.FORMAL_EPOCHS)
    normal_bad=[e for e in epochs if len({r[v]['normal_plan_hashes'][str(e)] for v in r})!=1]
    pair_bad=[e for e in epochs if len({r[v]['aux_pair_hashes'][str(e)] for v in r})!=1]
    lr_equal=len({tuple(r[v]['lr_trajectory']) for v in r})==1;init_equal=len({r[v]['initial_state_hash'] for v in r})==1;steps={v:r[v]['normal_optimizer_steps_total'] for v in r}
    warm_metrics={v:r[v]['trajectory'][9]['evaluation']['colift'] for v in r};warm_metric_exact=len({tuple((k,float(m[k])) for k in sorted(m)) for m in warm_metrics.values()})==1
    dig={v:r[v]['real_negative_plan_digest'] for v in r};plan_equal=len(set(dig.values()))==1
    out={'protocol':c.PROTOCOL,'seed':seed,'initial_state_exact':init_equal,'warmup_validation_metrics_exact':warm_metric_exact,'normal_plan_mismatch_epochs':normal_bad,'aux_pair_mismatch_epochs':pair_bad,'lr_trajectory_exact':lr_equal,'normal_optimizer_steps':steps,'warmup_epochs':{v:r[v]['warmup_epochs'] for v in r},'aux_onset_epoch':{v:r[v]['aux_onset_epoch'] for v in r},'lambda_HN':{v:r[v]['lambda_HN'] for v in r},'real_negative_plan_digests':dig,'C1FULL_C1DETACH_D1HM_real_negative_ids_exact':plan_equal,'D1HM_stale_synthetic_reuse_count':r['D1HM']['stale_synthetic_reuse_count'],'PASS':bool(init_equal and warm_metric_exact and not normal_bad and not pair_bad and lr_equal and len(set(steps.values()))==1 and plan_equal and r['D1HM']['stale_synthetic_reuse_count']==0),'TEST_ACCESSED':False}
    c.json_write(c.EVID/f'ROUND23_FAIRNESS_SEED{seed}.json',out);return out

def validation_summary():
    keys=[('C1FULL','B0'),('C1DETACH','B0'),('C1FULL','C1DETACH'),('D1HM','C1DETACH'),('D1HM','B0')]
    out={'protocol':c.PROTOCOL,'seeds':{},'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False};agg={f'{a}_vs_{b}':[] for a,b in keys}
    for seed in c.SEEDS:
        r={v:load_result(seed,v) for v in c.VARIANTS};row={'best_epochs':{v:r[v]['best_epoch'] for v in r},'best_metrics':{v:r[v]['best_evaluation']['colift'] for v in r}}
        for a,b in keys:
            k=f'{a}_vs_{b}';row[k]=c.delta_pack(row['best_metrics'][a],row['best_metrics'][b]);agg[k].append(row[k]['U'])
        out['seeds'][str(seed)]=row
    for k,v in agg.items(): out[f'mean_U_{k}']=float(np.mean(v))
    out['hard_negative_direction_confirmed']=bool(all(out['seeds'][str(s)]['C1FULL_vs_B0']['U']>0 for s in c.SEEDS))
    out['detach_hard_negative_positive_both']=bool(all(out['seeds'][str(s)]['C1DETACH_vs_B0']['U']>0 for s in c.SEEDS))
    out['diffusion_specific_positive_both']=bool(all(out['seeds'][str(s)]['D1HM_vs_C1DETACH']['U']>0 for s in c.SEEDS))
    out['diffusion_specific_success']=bool(out['diffusion_specific_positive_both'] and out['mean_U_D1HM_vs_C1DETACH']>0)
    out['diffusion_specific_strong']=bool(out['diffusion_specific_success'] and out['mean_U_D1HM_vs_C1DETACH']>=.005)
    c.json_write(c.EVID/'ROUND23_VALIDATION_SUMMARY.json',out);return out
