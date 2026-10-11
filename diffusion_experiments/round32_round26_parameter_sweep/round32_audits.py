from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from diffusion_experiments.round32_round26_parameter_sweep import round32_core as c

R26E=c.ROOT/'diffusion_experiments/round26_preference_directed_diffusion/evidence'
R31E=c.ROOT/'diffusion_experiments/round31_round26_baby_test_confirmation/evidence'

def _j(p): return json.loads(Path(p).read_text())

def write_config():
    r31=_j(R31E/'ROUND31_TEST_SUMMARY.json')
    out={'protocol':c.PROTOCOL,'protocol_hash':c.protocol_hash(),'objective':'Round26 boundary-supervision strength robustness sweep','dataset':'Amazon Baby','seeds':list(c.SEEDS),'variant_configs':c.VARIANT_CONFIGS,'train_variants':list(c.TRAIN_VARIANTS),'new_formal_jobs':len(c.TRAIN_VARIANTS)*len(c.SEEDS),'frozen':{'local_t':list(c.LOCAL_TS),'warmup_epochs':c.WARMUP_EPOCHS,'formal_epochs':c.FORMAL_EPOCHS,'diffusion_active_steps':c.DIFF_STEPS_ACTIVE,'checkpoint_selection':'best Full CoLiftRec Validation R20','candidate_mining':'Round26 original','boundary_bands':'Round26 original','optimizer_scheduler':'Round26 original','Full_CoLiftRec':'frozen'},'R26_BASE_reference':{'Validation_mean_U4':r31['Validation_mean_U4'],'Test_mean_U4':r31['Test_mean_U4'],'Test_seed999_U4':r31['seeds']['999']['Test']['D3_vs_B0']['U4'],'Test_seed1000_U4':r31['seeds']['1000']['Test']['D3_vs_B0']['U4']},'BABY_TEST_ROLE':'DEVELOPMENT_BENCHMARK_AFTER_ROUND31','BABY_TEST_UNTOUCHED_FINAL_TEST':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False,'FINAL_PARAMETER_SELECTION':'ADVISOR_OR_USER_ONLY'}
    c.json_write(c.EVID/'ROUND32_CONFIG.json',out); return out

def smoke_audit():
    s=_j(c.EVID/'ROUND32_SMOKE_C1_SEED999.json'); b=_j(c.EVID/'ROUND32_SMOKE_TEST.json'); m=s['mechanism_at_best']
    out={'protocol':c.PROTOCOL,'variant':'C1','seed':999,'config':s['config'],'checks':{'lambda_exact':abs(float(s['lambda_HN'])-.15)<1e-12,'half_margin_exact':abs(float(s['half_margin'])-.35)<1e-12,'theta_exact':abs(float(s['theta_max_deg'])-4.0)<1e-12,'warmup_seen':any(x['active_negative_count']==0 for x in s['trajectory']),'active_seen':any(x['active_negative_count']>0 for x in s['trajectory']),'aux_loss_positive':any(x['aux_loss_mean']>0 for x in s['trajectory'] if x['active_negative_count']>0),'theta_cap_pass':bool(m and m['angle_deg']['max']<=4.01),'requested_shift_nonzero':bool(m and m['requested_score_shift']['max']>0),'test_evaluator_parity':bool(b['PASS']),'test_evaluator_per_user_shape':b['per_user_shape']==[19445,4]},'smoke_metrics_not_used_for_sweep_change':True,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
    out['PASS']=all(out['checks'].values()); c.json_write(c.EVID/'ROUND32_SMOKE_AUDIT.json',out)
    if not out['PASS']: raise RuntimeError('Round32 smoke audit failed')
    return out

def fairness(seed):
    rows={v:_j(c.EVID/f'ROUND32_{v}_SEED{seed}_TRAIN.json') for v in c.TRAIN_VARIANTS}; ref=rows[c.TRAIN_VARIANTS[0]]
    def same_epoch_dict(key): return all(rows[v][key]==ref[key] for v in c.TRAIN_VARIANTS)
    out={'protocol':c.PROTOCOL,'seed':seed,'variants':list(c.TRAIN_VARIANTS),'initial_state_exact':len({rows[v]['initial_state_hash'] for v in c.TRAIN_VARIANTS})==1,'normal_plan_exact':same_epoch_dict('normal_plan_hashes'),'aux_pair_exact':same_epoch_dict('aux_pair_hashes'),'rank_offset_exact':same_epoch_dict('rank_offset_hashes'),'lr_trajectory_exact':all(rows[v]['lr_trajectory']==ref['lr_trajectory'] for v in c.TRAIN_VARIANTS),'normal_optimizer_steps_exact':len({rows[v]['normal_optimizer_steps_total'] for v in c.TRAIN_VARIANTS})==1,'diffusion_optimizer_steps_exact':len({rows[v]['diffusion_optimizer_steps_total'] for v in c.TRAIN_VARIANTS})==1,'epochs_exact':all(rows[v]['epochs_trained']==60 for v in c.TRAIN_VARIANTS),'config_only_registered':all({k:rows[v]['config'][k] for k in ('lambda_HN','half_margin','theta_max_deg')}=={k:c.VARIANT_CONFIGS[v][k] for k in ('lambda_HN','half_margin','theta_max_deg')} for v in c.TRAIN_VARIANTS),'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
    out['PASS']=bool(out['initial_state_exact'] and out['normal_plan_exact'] and out['aux_pair_exact'] and out['rank_offset_exact'] and out['lr_trajectory_exact'] and out['normal_optimizer_steps_exact'] and out['diffusion_optimizer_steps_exact'] and out['epochs_exact'] and out['config_only_registered'])
    c.json_write(c.EVID/f'ROUND32_FAIRNESS_SEED{seed}.json',out)
    if not out['PASS']: raise RuntimeError(f'Round32 fairness failed seed{seed}: {out}')
    return out

def _mechanism_subset(m):
    if not m: return None
    return {'A_true_median':m['A_true']['median'],'A_shuf_median':m['A_shuf']['median'],'DeltaA_median':m['DeltaA']['median'],'P_DeltaA_gt0':m['DeltaA']['p_gt_zero'],'applied_refinement_fraction':m['applied_refinement_fraction'],'direction_positive_fraction':m['direction_positive_fraction'],'direction_miss_fraction':m['direction_miss_fraction'],'cap_hit_fraction':m['cap_hit_fraction'],'target_reachable_fraction':m['target_reachable_fraction'],'angle_mean':m['angle_deg']['mean'],'angle_median':m['angle_deg']['median'],'angle_p90':m['angle_deg']['p90'],'angle_max':m['angle_deg']['max'],'requested_score_shift_mean':m['requested_score_shift']['mean'],'requested_score_shift_p90':m['requested_score_shift']['p90'],'actual_score_shift_mean':m['actual_score_shift']['mean'],'actual_score_shift_p90':m['actual_score_shift']['p90'],'real_negative_margin_mean':m['real_negative_margin']['mean'],'real_negative_margin_median':m['real_negative_margin']['median'],'refined_negative_margin_mean':m['refined_negative_margin']['mean'],'refined_negative_margin_median':m['refined_negative_margin']['median']}

def summarize():
    r31=_j(R31E/'ROUND31_TEST_SUMMARY.json')
    sweep={}; valtab={}; testtab={}; mech={}
    base={'variant':'R26_BASE','config':c.VARIANT_CONFIGS['R26_BASE'],'V999':r31['seeds']['999']['Validation']['D3_vs_B0']['U4'],'V1000':r31['seeds']['1000']['Validation']['D3_vs_B0']['U4'],'Vmean':r31['Validation_mean_U4'],'T999':r31['seeds']['999']['Test']['D3_vs_B0']['U4'],'T1000':r31['seeds']['1000']['Test']['D3_vs_B0']['U4'],'Tmean':r31['Test_mean_U4'],'Tmin':min(r31['seeds']['999']['Test']['D3_vs_B0']['U4'],r31['seeds']['1000']['Test']['D3_vs_B0']['U4']),'Validation_primary':[r31['seeds']['999']['Validation']['D3_vs_B0']['primary_positive_count'],r31['seeds']['1000']['Validation']['D3_vs_B0']['primary_positive_count']],'Test_primary':[r31['seeds']['999']['Test']['D3_vs_B0']['primary_positive_count'],r31['seeds']['1000']['Test']['D3_vs_B0']['primary_positive_count']]}
    sweep['R26_BASE']=base
    for v in c.TRAIN_VARIANTS:
        z={str(s):_j(c.EVID/f'ROUND32_{v}_SEED{s}.json') for s in c.SEEDS}
        vu=[z[str(s)]['Validation_vs_B0']['U'] for s in c.SEEDS]; tu=[z[str(s)]['Test_vs_B0']['U'] for s in c.SEEDS]
        row={'variant':v,'config':c.VARIANT_CONFIGS[v],'V999':vu[0],'V1000':vu[1],'Vmean':float(np.mean(vu)),'T999':tu[0],'T1000':tu[1],'Tmean':float(np.mean(tu)),'Tmin':float(min(tu)),'Validation_primary':[z[str(s)]['Validation_vs_B0']['primary_positive_count'] for s in c.SEEDS],'Test_primary':[z[str(s)]['Test_vs_B0']['primary_positive_count'] for s in c.SEEDS],'bootstrap_positive_fraction':[z[str(s)]['bootstrap']['positive_replicate_fraction'] for s in c.SEEDS],'bootstrap_ci95':[z[str(s)]['bootstrap']['ci95'] for s in c.SEEDS]}
        row['ROUND32_TARGET_REACHED']=bool(all(x>0 for x in vu) and all(x>0 for x in tu) and row['Tmean']>=.01 and all(x>=3 for x in row['Test_primary']))
        row['STRONG_ROBUST']=bool(row['ROUND32_TARGET_REACHED'] and row['T999']>=.005 and row['T1000']>=.005)
        row['ROBUST_IMPROVEMENT']=bool(all(x>0 for x in tu) and row['Tmean']>base['Tmean'] and row['Tmin']>base['Tmin'])
        sweep[v]=row
        valtab[v]={str(s):{'metrics':z[str(s)]['Validation'],'vs_B0':z[str(s)]['Validation_vs_B0'],'best_epoch':z[str(s)]['best_epoch']} for s in c.SEEDS}
        testtab[v]={str(s):{'metrics':z[str(s)]['Test'],'vs_B0':z[str(s)]['Test_vs_B0'],'bootstrap':z[str(s)]['bootstrap']} for s in c.SEEDS}
        mech[v]={str(s):_mechanism_subset(z[str(s)]['mechanism']) for s in c.SEEDS}
    rows=list(sweep.values()); rank_mean=sorted(rows,key=lambda x:x['Tmean'],reverse=True); rank_min=sorted(rows,key=lambda x:x['Tmin'],reverse=True); rank_val=sorted(rows,key=lambda x:x['Vmean'],reverse=True)
    ranking={'by_Test_mean':[{'rank':i+1,'variant':r['variant'],'Tmean':r['Tmean'],'Tmin':r['Tmin']} for i,r in enumerate(rank_mean)],'by_Test_min_seed':[{'rank':i+1,'variant':r['variant'],'Tmin':r['Tmin'],'Tmean':r['Tmean']} for i,r in enumerate(rank_min)],'by_Validation_mean':[{'rank':i+1,'variant':r['variant'],'Vmean':r['Vmean'],'Tmean':r['Tmean']} for i,r in enumerate(rank_val)],'best_Test_mean':rank_mean[0]['variant'],'best_min_seed':rank_min[0]['variant'],'best_Validation_mean':rank_val[0]['variant'],'target_reached_variants':[r['variant'] for r in rows if r.get('ROUND32_TARGET_REACHED')],'strong_robust_variants':[r['variant'] for r in rows if r.get('STRONG_ROBUST')],'robust_improvement_variants':[r['variant'] for r in rows if r.get('ROBUST_IMPROVEMENT')],'FINAL_PARAMETER_SELECTION':'NOT_MADE'}
    c.json_write(c.EVID/'ROUND32_SWEEP_TABLE.json',sweep); c.json_write(c.EVID/'ROUND32_VALIDATION_TABLE.json',valtab); c.json_write(c.EVID/'ROUND32_TEST_TABLE.json',testtab); c.json_write(c.EVID/'ROUND32_ROBUSTNESS_RANKING.json',ranking); c.json_write(c.EVID/'ROUND32_MECHANISM_SUMMARY.json',mech)
    return sweep,ranking,mech

def write_report():
    sweep,ranking,mech=summarize(); fair999=_j(c.EVID/'ROUND32_FAIRNESS_SEED999.json'); fair1000=_j(c.EVID/'ROUND32_FAIRNESS_SEED1000.json')
    pct=lambda x:f'{100*x:+.4f}%'
    lines=['# Round32 Final Report — Round26 Preference-Directed Diffusion Parameter Robustness Sweep','',
    '## 1. Objective','Determine whether Round26 Test attenuation is mainly caused by non-robust boundary-supervision strength, using only the preregistered lambda_HN, half_margin, and theta_max sweep.','',
    '## 2. Why Round26 Was Reopened','Round31 confirmed positive Test direction on both seeds, but cross-seed Test mean U4 (+0.4735%) was below the Validation mean (+1.0879%).','',
    '## 3. Round31 Reference',f"R26_BASE Validation mean U4={pct(sweep['R26_BASE']['Vmean'])}; Test mean U4={pct(sweep['R26_BASE']['Tmean'])}; Test minimum seed={pct(sweep['R26_BASE']['Tmin'])}.",'',
    '## 4. Baby Test Development-Role Statement','Baby Test participates in Round32 parameter comparison and is therefore a development benchmark after Round31. It must no longer be described as a completely untouched final Test set. Sports and Electronics remained closed.','',
    '## 5. Frozen Method Components','Diffusion architecture, BASE/TRUE/SHUFFLED branches, preference direction, tangent projection, candidate mining, boundary bands, curriculum, local_t=[1..5], 10 warmup epochs, 60 formal epochs, two active Diffusion steps, optimizer, scheduler, normal loss, Full CoLiftRec, and Validation-R20 checkpoint selection were frozen.','',
    '## 6. Sweep Space','Exactly 12 new configurations were trained, plus reused R26_BASE; no full grid or post-hoc configuration was added.','',
    '## 7. Fairness Audit',f"seed999 PASS={fair999['PASS']}; seed1000 PASS={fair1000['PASS']}. Initial state, normal batch/negative plan, auxiliary user-positive plan, rank offsets, LR trajectory, optimizer-step counts, and epoch counts matched across variants.",'',
    '## 8. Lambda Sweep','See master table below.','', '## 9. Half-Margin Sweep','See master table below.','', '## 10. Theta Sweep','See master table below.','', '## 11. Combination Sweep','See master table below.','',
    '## 12. Validation Results','', '| Variant | λHN | Margin | θ | V999 | V1000 | Vmean | T999 | T1000 | Tmean | Tmin | Test Primary |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    order=['R26_BASE','L10','L15','L25','M25','M35','M65','T3','T4','T6','C1','C2','C3']
    for v in order:
        r=sweep[v]; q=r['config']; lines.append(f"| {v} | {q['lambda_HN']:.2f} | {q['half_margin']:.2f} | {q['theta_max_deg']:.0f} | {pct(r['V999'])} | {pct(r['V1000'])} | {pct(r['Vmean'])} | {pct(r['T999'])} | {pct(r['T1000'])} | {pct(r['Tmean'])} | {pct(r['Tmin'])} | {r['Test_primary']} |")
    lines += ['','## 13. Test Results','The Test columns above use the exact Validation-selected checkpoint for each variant; Test was never evaluated per epoch.','', '## 14. Cross-Seed Robustness',f"Best Test mean: {ranking['best_Test_mean']}; best minimum-seed U4: {ranking['best_min_seed']}; robust-improvement variants: {ranking['robust_improvement_variants']}.",'',
    '## 15. Bootstrap','Each new Test variant uses 1000 paired user bootstrap resamples against the same-seed frozen B0. Positive replicate fraction is descriptive and is not a p-value.','',
    '## 16. Mechanism Diagnostics','Per-variant/per-seed mechanism evidence includes A_true/A_shuf/DeltaA, refinement/direction fractions, cap hit, reachability, angle statistics, requested/actual score shifts, and real/refined negative margins in `ROUND32_MECHANISM_SUMMARY.json`.','',
    '## 17. Validation-vs-Test Parameter Consistency',f"Best Validation mean: {ranking['best_Validation_mean']}; best Test mean: {ranking['best_Test_mean']}; identical={ranking['best_Validation_mean']==ranking['best_Test_mean']}.",'',
    '## 18. Best Test Mean Configuration',str(ranking['by_Test_mean'][:5]),'',
    '## 19. Best Min-Seed Configuration',str(ranking['by_Test_min_seed'][:5]),'',
    '## 20. Whether +1% Target Was Reached',f"Target reached variants: {ranking['target_reached_variants']}; STRONG_ROBUST variants: {ranking['strong_robust_variants']}.",'',
    '## 21. Unexpected Findings','Interpret parameter trends from the complete preregistered table; no post-Test rescue or extra sweep was run.','',
    '## 22. Data Access Audit','Baby Validation = OPEN. Baby Test = OPEN as development benchmark. Sports = CLOSED. Electronics = CLOSED.','',
    '## 23. Git Commit','Implementation and final evidence are committed and pushed on the dedicated Round32 branch.','',
    '## Ranking by Test Mean U4']
    lines += [f"{x['rank']}. {x['variant']}: mean={pct(x['Tmean'])}, min={pct(x['Tmin'])}" for x in ranking['by_Test_mean']]
    lines += ['','## Ranking by Minimum Test Seed U4']+[f"{x['rank']}. {x['variant']}: min={pct(x['Tmin'])}, mean={pct(x['Tmean'])}" for x in ranking['by_Test_min_seed']]
    lines += ['','## Ranking by Validation Mean U4']+[f"{x['rank']}. {x['variant']}: Vmean={pct(x['Vmean'])}, Tmean={pct(x['Tmean'])}" for x in ranking['by_Validation_mean']]
    p=c.RDIR/'ROUND32_FINAL_REPORT.md'; p.write_text('\n'.join(lines)+'\n'); return {'path':str(p),'ranking':ranking}
