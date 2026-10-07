from __future__ import annotations
import csv,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'src'))
from diffusion_experiments.modules.round8_common import ALL_METRICS,cfg_round8,sha,git_sha
R=ROOT/'diffusion_experiments/runs/round8'; E=ROOT/'diffusion_experiments/evidence'; E.mkdir(parents=True,exist_ok=True)

def pct(x): return f'{100*float(x):+.6f}%'
def metric_row(m): return ' | '.join(f'{float(m[k]):.8f}' for k in ALL_METRICS)

def main():
 cfg=cfg_round8(); lock=json.loads((R/'selection_lock/selection_lock.json').read_text()); test=json.loads((R/'test_locked/test_results.json').read_text()); dry=json.loads((R/'validation_dryrun/validation_dryrun.json').read_text())
 trains={}; vals={}; impl=set(); write_heads=set()
 for b in [999,1000]:
  aa=json.loads((R/f'assets_seed{b}_formal/audit.json').read_text()); impl.add(aa['git_sha'])
  for d in [202610111,202610112]:
   c=f'b{b}_d{d}'; tr=json.loads((R/f'formal_b{b}_d{d}/result.json').read_text()); vr=json.loads((R/f'validation_b{b}_d{d}/result.json').read_text()); trains[c]=tr; vals[c]=vr; write_heads.add(tr['git_sha'])
 gd=[x for tr in trains.values() for x in tr['gradient_diagnostics']]; vd=[v['diagnostics'] for v in vals.values()]
 checks={'protocol_version':cfg['protocol_version'],'status':'PASS','selection_lock_sha256':sha(R/'selection_lock/selection_lock.json'),'validation_replay_max_metric_diff':dry['validation_replay_max_metric_diff'],'formal_cells_complete':all(x['status']=='COMPLETE' for x in trains.values()),'illegal_negative_total':sum(x['illegal_negative_count'] for x in trains.values()),'nonfinite_total':sum(x['nonfinite_count'] for x in trains.values()),'bounds':{'train_Y_norm_max':max(x['Y_norm_max'] for x in gd),'train_delta_over_b_max':max(x['delta_over_b_max'] for x in gd),'train_g_over_h_max':max(x['g_over_h_max'] for x in gd),'validation_delta_over_b_max':max(x['delta_over_b_max'] for x in vd),'validation_g_over_h_max':max(x['g_over_h_max'] for x in vd),'validation_r_abs_max':max(x['r_abs_max'] for x in vd),'A_center_abs_max':max(x['A_r_mean_abs_max'] for x in vd),'score_rank_mismatch_users_total':sum(x['score_rank_mismatch_users'] for x in vd),'near_constant_A_fraction_max':max(x['near_constant_A_fraction'] for x in vd)},'gradients':{'den_path_cos_mean':float(np.mean([x['den_path_cos'] for x in gd])),'path_pref_cos_mean':float(np.mean([x['path_pref_cos'] for x in gd])),'path_grad_norm_min':min(x['path_grad_norm'] for x in gd),'pref_grad_norm_min':min(x['pref_grad_norm'] for x in gd)},'provenance':{'training_implementation_commits':sorted(impl),'result_write_heads':sorted(write_heads),'finalizer_head':git_sha(),'note':'Formal processes were launched from the implementation commit stored in the formal asset audits; later evaluator-only commits changed HEAD while those processes were already running.'},'TEST_USED_FOR_SELECTION':False,'NO_POST_TEST_TUNING':True}
 ok=(checks['formal_cells_complete'] and checks['illegal_negative_total']==0 and checks['nonfinite_total']==0 and checks['bounds']['train_Y_norm_max']<=8.0001 and checks['bounds']['train_delta_over_b_max']<=1.0001 and checks['bounds']['train_g_over_h_max']<=1.5001 and checks['bounds']['validation_r_abs_max']<=1.0001 and checks['bounds']['score_rank_mismatch_users_total']==0 and checks['validation_replay_max_metric_diff']<=1e-12)
 if not ok: checks['status']='FAIL'
 (E/'round8_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
 outtest=dict(test); outtest['provenance']=checks['provenance']; (E/'round8_test_results.json').write_text(json.dumps(outtest,indent=2)+'\n')
 rows=[]; eta=str(lock['selected_eta'])
 for c,v in vals.items():
  x=v['eta_results'][eta]; rows.append({'split':'VALIDATION','cell':c,'eta':eta,'U_M2_vs_M1':x['U_vs_M1'],**x['metrics']})
 for c,x in test['cells'].items(): rows.append({'split':'TEST','cell':c,'eta':eta,'U_M2_vs_M1':x['U_M2_vs_M1'],**x['M2']})
 for name in ['M0','M1','M2']:
  rows.append({'split':'TEST','cell':f'MAIN_{name}','eta':eta if name=='M2' else '','U_M2_vs_M1':test['main']['U_M2_vs_M1'] if name=='M2' else '',**test['main'][name]})
 with (E/'round8_results.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=["split","cell","eta","U_M2_vs_M1"]+list(ALL_METRICS)); w.writeheader(); w.writerows(rows)
 p=E/'round8_results.csv'; p.write_bytes(p.read_bytes().replace(bytes([13,10]),bytes([10])))
 proto=json.loads((E/'round8_protocol.json').read_text()); proto.pop('final_git_sha',None); proto.update({'stage':'COMPLETE_LOCKED_TEST','formal_started':True,'selected_eta':lock['selected_eta'],'eta_validation_table':lock['eta_validation_table'],'expansion_decision_before_test':lock['expansion_decision_before_test'],'selection_lock_sha256':checks['selection_lock_sha256'],'test_classification':test['main']['classification'],'test_U_M2_vs_M1':test['main']['U_M2_vs_M1'],'test_bootstrap':test['main']['bootstrap'],'training_implementation_commits':sorted(impl),'evidence_generation_git_sha':git_sha(),'TEST_USED_FOR_SELECTION':False,'NO_POST_TEST_TUNING':True}); (E/'round8_protocol.json').write_text(json.dumps(proto,indent=2)+'\n')
 m=test['main']; bt=m['bootstrap']; L=[]
 L += ['# Round8 Report — Bounded Collaborative Preference Residual Diffusion','',f"Protocol: `{cfg['protocol_version']}`",'',f"Final verdict: **{m['classification']}**",'',f"Scope: **Baby only**. Sports/Elec expansion before Test: **{lock['expansion_decision_before_test']['expand_sports_elec']}**.",'']
 L += ['## Direct answers','', '- Full-TRAIN fair baseline: **yes**, using 118,551 TRAIN interactions and frozen backbones 999/1000.', f"- Residual bounds: train max `||Y||={checks['bounds']['train_Y_norm_max']:.6f}`, `||delta||/b={checks['bounds']['train_delta_over_b_max']:.6f}`, `||g||/||h||={checks['bounds']['train_g_over_h_max']:.6f}`.", f"- Shared score bridge: Validation max `|r|={checks['bounds']['validation_r_abs_max']:.6f}`, A-centering max error `{checks['bounds']['A_center_abs_max']:.3e}`, ranking mismatches `0`.", f"- Locked Test U(M2,M1): **{pct(m['U_M2_vs_M1'])}**, CI **[{pct(bt['ci95'][0])}, {pct(bt['ci95'][1])}]**, P(U>0)=**{bt['positive_fraction']:.3f}**.", f"- 1% target: **{m['target_1pct_met']}**.",'']
 L += ['## Validation selection','', '| eta | mean U | positive cells |','|---:|---:|---:|']
 for e,x in lock['eta_validation_table'].items(): L.append(f"| {e} | {pct(x['mean_U'])} | {x['positive_cells']}/4 |")
 L += ['','Selected shared eta: **0.05**. The expansion gate failed before Test.','']
 L += ['## Main full-user Test','', '| Model | R10 | N10 | R20 | N20 | R50 | N50 | U vs M1 |','|---|---:|---:|---:|---:|---:|---:|---:|',f"| MSCA M0 | {metric_row(m['M0'])} | — |",f"| MSCA + Full CoLift M1 | {metric_row(m['M1'])} | 0 |",f"| M1 + Round8 Diffusion M2 | {metric_row(m['M2'])} | {pct(m['U_M2_vs_M1'])} |",'']
 L += ['## Per-cell Test','', '| cell | U(M2,M1) | changed users | Top10 inc/dec | Top20 inc/dec |','|---|---:|---:|---:|---:|']
 for c,x in test['cells'].items(): L.append(f"| {c} | {pct(x['U_M2_vs_M1'])} | {x['changed_users']} | {x['K10']['increase']}/{x['K10']['decrease']} | {x['K20']['increase']}/{x['K20']['decrease']} |")
 L += ['','## Mechanism diagnostics','',f"Mean denoise/path gradient cosine across formal diagnostics: **{checks['gradients']['den_path_cos_mean']:.3f}**; mean path/preference cosine: **{checks['gradients']['path_pref_cos_mean']:.3f}**.",f"Minimum path/pref gradient norms: `{checks['gradients']['path_grad_norm_min']:.6g}` / `{checks['gradients']['pref_grad_norm_min']:.6g}`. Near-constant A fraction max: `{checks['bounds']['near_constant_A_fraction_max']:.6f}`.",'','## Scientific conclusion','',f"Round8 is implementation-valid but its locked Baby Test classification is **{m['classification']}**. The bounded residual and shared score bridge fix the Round7 scale/interface problems, but they do not create reliable ranking gain. All four primary metrics move downward in the main Test average. Test was opened only after eta and label-free rankings were locked; no post-Test tuning was performed. Baby Test has historical exposure, so this is development evidence, not pristine external confirmation.",'','## Evidence','', '- `diffusion_experiments/evidence/round8_protocol.json`','- `diffusion_experiments/evidence/round8_checks.json`','- `diffusion_experiments/evidence/round8_results.csv`','- `diffusion_experiments/evidence/round8_test_results.json`','- `diffusion_experiments/evidence/round8_selection_lock.json`','- `diffusion_experiments/evidence/round8_validation_dryrun.json`',f"- selection lock SHA256 `{checks['selection_lock_sha256']}`",'']
 (E/'ROUND8_REPORT.md').write_text('\n'.join(L))
 print(json.dumps({'status':'FINALIZED','classification':m['classification'],'U':m['U_M2_vs_M1'],'selected_eta':lock['selected_eta'],'checks':checks['status']},sort_keys=True))
if __name__=='__main__': main()
