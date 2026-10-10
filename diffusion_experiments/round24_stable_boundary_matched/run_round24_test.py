from __future__ import annotations
import argparse,gc,json,subprocess
from pathlib import Path
import numpy as np
import torch
from diffusion_experiments.round20_positive_anchored_hardneg.run_round20_test import TestEvaluator,evaluate_detail
from diffusion_experiments.round24_stable_boundary_matched import round24_core as c
from diffusion_experiments.round24_stable_boundary_matched import round24_formal as f


def git_out(*args):return subprocess.check_output(['git',*args],cwd=c.ROOT,text=True).strip()

def freeze_validation():
    s=json.loads((c.EVID/'ROUND24_VALIDATION_SUMMARY.json').read_text());fair={str(seed):json.loads((c.EVID/f'ROUND24_FAIRNESS_SEED{seed}.json').read_text()) for seed in c.SEEDS}
    if not all(x['PASS'] for x in fair.values()):raise RuntimeError('fairness not PASS')
    rows={}
    for seed in c.SEEDS:
        rows[str(seed)]={}
        for v in c.VARIANTS:
            r=f.load_result(seed,v);p=Path(r['best_checkpoint']);got=c.sha256_file(p)
            if got!=r['checkpoint_sha256']:raise RuntimeError(f'checkpoint SHA mismatch before freeze {seed} {v}')
            rows[str(seed)][v]={'best_epoch':r['best_epoch'],'checkpoint':str(p),'checkpoint_sha256':got,'best_metrics':r['best_evaluation']['colift']}
    out={'protocol':c.PROTOCOL,'protocol_hash':c.protocol_hash(),'branch':git_out('branch','--show-current'),'validation_source_HEAD':git_out('rev-parse','HEAD'),'parameters':{'dataset':'Baby','seeds':[999,1000],'warmup_epochs':10,'formal_epochs':60,'lambda_HN':.20,'T':25,'beta':'linear 1e-4 -> 0.02','p_text_drop':.05,'p_visual_drop':.05,'s_text':1.1,'s_visual':1.1,'bands':{'Easy':'ranks61-100','Medium':'ranks31-60','Hard':'ranks11-30'},'conditional_candidate_t':'1..23','pure_noise_diagnostic_t':24,'norm_epsilon':1e-8,'checkpoint_selection':'Full CoLiftRec Validation R20'},'validation_summary':s,'fairness':fair,'rows':rows,'status':'FROZEN_BEFORE_TEST','TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
    c.json_write(c.EVID/'ROUND24_VALIDATION_FREEZE.json',out);return out

def load_frozen_model(seed,variant,lock):
    row=lock['rows'][str(seed)][variant];p=Path(row['checkpoint']);got=c.sha256_file(p)
    if got!=row['checkpoint_sha256']:raise RuntimeError(f'checkpoint SHA mismatch {seed} {variant}')
    model,config,td,vd,pack=c.instantiate_shared(seed);cp=torch.load(p,map_location='cpu',weights_only=False)
    if int(cp['epoch'])!=int(row['best_epoch']):raise RuntimeError(f'checkpoint epoch mismatch {seed} {variant}')
    model.load_state_dict(cp['model_state_dict'],strict=True);model.eval();return model

def run_test(seed):
    lock=json.loads((c.EVID/'ROUND24_VALIDATION_FREEZE.json').read_text())
    if lock['status']!='FROZEN_BEFORE_TEST' or lock['TEST_ACCESSED'] is not False:raise RuntimeError('validation not frozen before Test')
    ev=TestEvaluator(seed);out={'phase':'ROUND24_ONE_TIME_BABY_TEST','seed':seed,'validation_source_HEAD':lock['validation_source_HEAD'],'protocol_hash':lock['protocol_hash'],'test_used_for_selection':False,'no_post_test_tuning':True,'variants':{},'TEST_ACCESSED':True,'BABY_TEST_STATUS':'CONSUMED','SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
    for v in c.VARIANTS:
        m=load_frozen_model(seed,v,lock);agg,_=evaluate_detail(ev,m);out['variants'][v]=agg;del m;torch.cuda.empty_cache();gc.collect()
    keys=[('C1FULL','B0'),('C1DETACH','B0'),('C1FULL','C1DETACH'),('D1BHM','B0'),('D1BHM','C1DETACH')]
    for a,b in keys:out[f'{a}_vs_{b}']=c.delta_pack(out['variants'][a]['colift'],out['variants'][b]['colift'])
    c.json_write(c.EVID/f'ROUND24_TEST_SEED{seed}.json',out);return out

def test_summary():
    keys=[('C1FULL','B0'),('C1DETACH','B0'),('C1FULL','C1DETACH'),('D1BHM','B0'),('D1BHM','C1DETACH')]
    rows={str(seed):json.loads((c.EVID/f'ROUND24_TEST_SEED{seed}.json').read_text()) for seed in c.SEEDS};out={'protocol':c.PROTOCOL,'seeds':{},'BABY_TEST_STATUS':'CONSUMED_CLOSED_FOR_FUTURE_TUNING','TEST_ACCESSED':True,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
    for seed in c.SEEDS:
        r=rows[str(seed)];out['seeds'][str(seed)]={'metrics':{v:r['variants'][v]['colift'] for v in c.VARIANTS}}
        for a,b in keys:out['seeds'][str(seed)][f'{a}_vs_{b}']=r[f'{a}_vs_{b}']
    for a,b in keys:out[f'mean_U_{a}_vs_{b}']=float(np.mean([out['seeds'][str(s)][f'{a}_vs_{b}']['U'] for s in c.SEEDS]))
    c.json_write(c.EVID/'ROUND24_TEST_SUMMARY.json',out);return out

def write_report():
    pre=json.loads((c.EVID/'ROUND24_PREFLIGHT.json').read_text());gate=json.loads((c.EVID/'ROUND24_GATE_STABILITY_AUDIT.json').read_text());val=json.loads((c.EVID/'ROUND24_VALIDATION_SUMMARY.json').read_text());test=json.loads((c.EVID/'ROUND24_TEST_SUMMARY.json').read_text())
    cov={str(seed):json.loads((c.EVID/f'ROUND24_TRAJECTORY_COVERAGE_SEED{seed}.json').read_text()) for seed in c.SEEDS}
    lines=['# Round24 Final Report','','## Protocol verdict','','Round24 completed the stable current-boundary / BPR-hardness-matched protocol and consumed the one-time Baby Test only after Validation freeze.','','## Gate stability','','Round23 relative-error blocking was unstable because its denominator |target margin|+1e-6 could approach zero. Round24 removed that ratio from blocking and uses bounded BPR gradient hardness sigmoid(-margin).',f"Gate audit: NO_NEAR_ZERO_DENOMINATOR_BLOCKING_METRIC = {gate['NO_NEAR_ZERO_DENOMINATOR_BLOCKING_METRIC']}.",'','## Preflight']
    for m in c.MODES:
        x=pre['BPR_hardness_match'][m];lines+=['',f"- {m}: hardness ratio median {x['hardness_ratio']['median']:.6f}; coverage {x['coverage_fraction']*100:.2f}%; too-easy {x['too_easy_fraction']*100:.2f}%; pure-noise preferred {x['pure_noise_preferred_fraction']*100:.2f}%."]
    lines+=['','## Validation']
    for seed in c.SEEDS:
        r=val['seeds'][str(seed)];lines+=['',f'### seed{seed}']
        for k in ('C1FULL_vs_B0','C1DETACH_vs_B0','C1FULL_vs_C1DETACH','D1BHM_vs_B0','D1BHM_vs_C1DETACH'):lines.append(f"- {k}: U {r[k]['U']*100:+.4f}% ({r[k]['primary_positive_count']}/4 primary, {r[k]['overall_positive_count']}/6 overall)")
    lines+=['','## One-time Baby Test']
    for seed in c.SEEDS:
        r=test['seeds'][str(seed)];lines+=['',f'### seed{seed}']
        for k in ('C1FULL_vs_B0','C1DETACH_vs_B0','C1FULL_vs_C1DETACH','D1BHM_vs_B0','D1BHM_vs_C1DETACH'):lines.append(f"- {k}: U {r[k]['U']*100:+.4f}% ({r[k]['primary_positive_count']}/4 primary, {r[k]['overall_positive_count']}/6 overall)")
    lines+=['','## Required answers','','1. Round23 relative-error gate was unstable because the target-margin denominator could be near zero.','2. Round24 BPR hardness is mathematically bounded in (0,1) and directly equals BPR margin-gradient magnitude.','3. Pure-noise preference is reported per mode/epoch but t=24 never enters training loss.','4. Conditional trajectory coverage is reported in preflight and every D1BHM active epoch.','5. Uncovered samples are split into too-easy versus too-hard.','6. C1-FULL reproduction is given by Validation/Test C1FULL_vs_B0.','7. C1-DETACH value is given by C1DETACH_vs_B0.','8. Negative-item gradient contribution is isolated by C1FULL_vs_C1DETACH.','9. D1-BHM BPR hardness matching is reported by ratio/error/loss diagnostics.','10. Diffusion-specific value is D1BHM_vs_C1DETACH.','11. Validation/Test consistency is shown above; Test was not used for any selection.','12. Hard-negative direction paper value is judged from both C1 controls across Validation/Test.','13. Diffusion hard-negative line closure follows the preregistered stop rule and is stated below.']
    healthy=all(pre['BPR_hardness_match'][m]['hardness_ratio']['median']>=.5 and pre['BPR_hardness_match'][m]['hardness_ratio']['median']<=1.5 for m in c.MODES)
    close=bool(healthy and all(val['seeds'][str(s)]['D1BHM_vs_C1DETACH']['U']<=0 for s in c.SEEDS))
    lines+=['','## Stop rule','',f"- Healthy BPR hardness matching: {healthy}",f"- D1BHM <= C1DETACH on both Validation seeds: {all(val['seeds'][str(s)]['D1BHM_vs_C1DETACH']['U']<=0 for s in c.SEEDS)}",f"- DIFFUSION_HARD_NEGATIVE_LINE = {'CLOSED' if close else 'NOT_CLOSED_BY_RULE'}",'','## Test discipline','','- BABY_TEST = CONSUMED / CLOSED FOR FUTURE TUNING','- Sports/Electronics were not accessed.','- No post-Test rescue or checkpoint reselection was performed.']
    p=c.EVID/'ROUND24_REPORT.md';p.write_text('\n'.join(lines)+'\n');return {'path':str(p),'diffusion_line_closed':close}

def main():
    ap=argparse.ArgumentParser();sp=ap.add_subparsers(dest='cmd',required=True);sp.add_parser('freeze');p=sp.add_parser('test');p.add_argument('--seed',type=int,choices=[999,1000],required=True);sp.add_parser('summary');sp.add_parser('report');a=ap.parse_args()
    out=freeze_validation() if a.cmd=='freeze' else run_test(a.seed) if a.cmd=='test' else test_summary() if a.cmd=='summary' else write_report();print(json.dumps(out,indent=2))
if __name__=='__main__':main()
