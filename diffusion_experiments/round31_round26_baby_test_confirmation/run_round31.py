from __future__ import annotations
import argparse,json
from diffusion_experiments.round31_round26_baby_test_confirmation import round31_core as c
from diffusion_experiments.round31_round26_baby_test_confirmation.round31_audits import checkpoint_audit,write_config
from diffusion_experiments.round31_round26_baby_test_confirmation.round31_test_eval import run_checkpoint,protocol_audit,summarize

def write_report():
    ck=json.loads((c.EVID/'ROUND31_CHECKPOINT_AUDIT.json').read_text())
    pa=json.loads((c.EVID/'ROUND31_TEST_PROTOCOL_AUDIT.json').read_text())
    s=json.loads((c.EVID/'ROUND31_TEST_SUMMARY.json').read_text())
    pct=lambda x:f'{100*x:+.4f}%'
    lines=['# Round31 Final Report — Frozen Round26 Baby Test Confirmation','',
      '## 1. Objective','Confirm whether the frozen Round26 D3 Validation gain over B0 survives on Baby Test, without retraining, reselection, or Test-driven tuning.','',
      '## 2. Frozen Round26 Protocol',f"Source protocol: `{c.SOURCE_PROTOCOL}`. Round26 settings remain frozen: 10 warmup epochs, 60 formal epochs, lambda_HN=0.20, theta_max=5°, half_margin=0.50, local_t=[1,2,3,4,5], Diffusion active steps=2.",'',
      '## 3. Checkpoint Integrity',f"Checkpoint audit PASS = {ck['PASS']}."]
    for seed in c.SEEDS:
        for v in c.VARIANTS:
            r=ck['rows'][str(seed)][v]; lines.append(f"- seed{seed} {v}: epoch {r['best_epoch']}; SHA256 `{r['actual_checkpoint_sha256']}`; PASS={r['PASS']}.")
    lines+=['','## 4. Test Protocol Audit',f"Protocol audit PASS = {pa['PASS']}; same Test users={pa['same_test_users_across_all_checkpoints']}; same ground truth={pa['same_ground_truth_across_all_checkpoints']}; same Full CoLift parameters={pa['same_full_colift_parameters_across_all_checkpoints']}; all checkpoint repeats exact={pa['all_checkpoint_repeats_exact']}; Diffusion inference calls all zero={pa['diffusion_inference_calls_all_zero']}.",'',
      '## 5. No-Test-Tuning Statement','Baby Test was opened only for frozen Round31 confirmation. Test was not used for parameter, epoch, threshold, architecture, angle, lambda, margin, or curriculum selection. No post-Test rescue was performed.','']
    for seed in c.SEEDS:
        z=s['seeds'][str(seed)]; val=z['Validation']; test=z['Test']; cmp=test['D3_vs_B0']
        lines += [f"## {6 if seed==999 else 8}. seed{seed} Validation Recap",f"B0: R10={val['B0']['R10']:.8f}, N10={val['B0']['N10']:.8f}, R20={val['B0']['R20']:.8f}, N20={val['B0']['N20']:.8f}, R50={val['B0']['R50']:.8f}, N50={val['B0']['N50']:.8f}.",f"D3: R10={val['D3']['R10']:.8f}, N10={val['D3']['N10']:.8f}, R20={val['D3']['R20']:.8f}, N20={val['D3']['N20']:.8f}, R50={val['D3']['R50']:.8f}, N50={val['D3']['N50']:.8f}.",f"Validation U4={pct(val['D3_vs_B0']['U4'])}; primary positives={val['D3_vs_B0']['primary_positive_count']}/4; overall positives={val['D3_vs_B0']['overall_positive_count']}/6.",'',f"## {7 if seed==999 else 9}. seed{seed} Test",f"B0: R10={test['B0']['R10']:.8f}, N10={test['B0']['N10']:.8f}, R20={test['B0']['R20']:.8f}, N20={test['B0']['N20']:.8f}, R50={test['B0']['R50']:.8f}, N50={test['B0']['N50']:.8f}.",f"D3: R10={test['D3']['R10']:.8f}, N10={test['D3']['N10']:.8f}, R20={test['D3']['R20']:.8f}, N20={test['D3']['N20']:.8f}, R50={test['D3']['R50']:.8f}, N50={test['D3']['N50']:.8f}.",f"Test U4={pct(cmp['U4'])}; primary positives={cmp['primary_positive_count']}/4; overall positives={cmp['overall_positive_count']}/6.",f"Relative deltas: R10={pct(cmp['relative_delta']['R10'])}, N10={pct(cmp['relative_delta']['N10'])}, R20={pct(cmp['relative_delta']['R20'])}, N20={pct(cmp['relative_delta']['N20'])}, R50={pct(cmp['relative_delta']['R50'])}, N50={pct(cmp['relative_delta']['N50'])}.",'']
    lines += ['## 10. Validation vs Test Comparison',f"Validation cross-seed mean U4={pct(s['Validation_mean_U4'])}; Test cross-seed mean U4={pct(s['Test_mean_U4'])}; change={pct(s['validation_to_test_mean_U4_delta'])}.",'',
      '## 11. Cross-Seed Mean Utility',f"seed999 Test U4={pct(s['seeds']['999']['Test']['D3_vs_B0']['U4'])}; seed1000 Test U4={pct(s['seeds']['1000']['Test']['D3_vs_B0']['U4'])}; mean={pct(s['Test_mean_U4'])}.",'',
      '## 12. Metric-Level Analysis']
    for seed in c.SEEDS:
        cmp=s['seeds'][str(seed)]['Test']['D3_vs_B0']; neg=[k for k in c.ALL if cmp['relative_delta'][k]<=0]; lines.append(f"- seed{seed}: non-positive metrics = {neg if neg else 'none'}; primary positives {cmp['primary_positive_count']}/4; overall positives {cmp['overall_positive_count']}/6.")
    lines += ['','## 13. STRONG_PASS / PASS / MIXED / FAIL',f"`ROUND31_STATUS = {s['ROUND31_STATUS']}`.",'',
      '## 14. Data Access Audit','- Baby Validation: historically used for Round26 checkpoint selection.','- Baby Test: opened only in Round31 for frozen confirmation.','- Sports: NOT ACCESSED.','- Electronics: NOT ACCESSED.','',
      '## 15. Git Commit','Round31 evaluator/audits and Test evidence are committed separately on the dedicated Round31 branch.','',
      '## Scientific Interpretation','If D3 improves Test, the correct interpretation is training-time Diffusion-guided boundary refinement: Diffusion is not called during Test inference; the trained recommender has absorbed the auxiliary boundary supervision.']
    p=c.RDIR/'ROUND31_FINAL_REPORT.md'; p.write_text('\n'.join(lines)+'\n'); return {'path':str(p),'ROUND31_STATUS':s['ROUND31_STATUS']}

def main():
    ap=argparse.ArgumentParser(); sp=ap.add_subparsers(dest='cmd',required=True)
    sp.add_parser('audit'); sp.add_parser('config')
    p=sp.add_parser('test'); p.add_argument('--seed',type=int,choices=c.SEEDS,required=True); p.add_argument('--variant',choices=c.VARIANTS,required=True)
    sp.add_parser('protocol'); sp.add_parser('summary'); sp.add_parser('report')
    a=ap.parse_args()
    if a.cmd=='audit': out=checkpoint_audit()
    elif a.cmd=='config': out=write_config()
    elif a.cmd=='test': out=run_checkpoint(a.seed,a.variant)
    elif a.cmd=='protocol': out=protocol_audit()
    elif a.cmd=='summary': out=summarize()
    else: out=write_report()
    print(json.dumps(out,indent=2,default=str))
if __name__=='__main__': main()
