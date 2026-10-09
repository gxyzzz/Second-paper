import json, hashlib
from pathlib import Path
R=Path('diffusion_experiments/round17_decision_guided'); E=R/'evidence'; O=R/'outputs'
SEEDS=(999,1000); VARS=('D1','D2','D3')
def load(p): return json.load(open(p))
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
rows={s:{v:load(O/f'seed{s}/{v}/result.json') for v in VARS} for s in SEEDS}
d0={s:load(E/f'ROUND17_D0_SEED{s}.json') for s in SEEDS}
orc={s:load(E/f'ROUND17_ORACLE_SEED{s}.json') for s in SEEDS}
rnd={s:{v:load(E/f'ROUND17_RANDOM_BASIS_{v}_SEED{s}.json') for v in ('D2','D3')} for s in SEEDS}
param=load(E/'ROUND17_PARAMETER_AUDIT.json'); hist=load(E/'ROUND17_HISTORY_AUDIT.json'); basis=load(E/'ROUND17_DIFFUSION_BASIS_AUDIT.json')
(E/'ROUND17_ORACLE_DIAGNOSTIC.json').write_text(json.dumps({'protocol':'ROUND17_TADGD_V1','seeds':{str(s):orc[s] for s in SEEDS},'TEST_ACCESSED':False},indent=2)+'\n')
(E/'ROUND17_RANDOM_BASIS_CONTROL.json').write_text(json.dumps({'protocol':'ROUND17_TADGD_V1','variants':{v:{str(s):rnd[s][v] for s in SEEDS} for v in ('D2','D3')},'TEST_ACCESSED':False},indent=2)+'\n')
hctrl={'protocol':'ROUND17_TADGD_V1','variants':{},'TEST_ACCESSED':False}
for v in ('D2','D3'):
 hctrl['variants'][v]={}
 for s in SEEDS:
  d=rows[s][v];m=d['hard_shell_mechanism'];hctrl['variants'][v][str(s)]={'U_TRUE':d['best_vs_C0']['U'],'U_HISTORY_SHUFFLED':d['history_control']['vs_C0']['U'],'history_control_label':d['history_control']['label'],'mean_true_minus_shuffled':m['history_advantage']['mean'],'fraction_true_gt_shuffled':m['fraction_true_gt_shuffled']}
(E/'ROUND17_HISTORY_SHUFFLE_CONTROL.json').write_text(json.dumps(hctrl,indent=2)+'\n')
def gates_for(v,s):
 d=rows[s][v];m=d['hard_shell_mechanism'];rel=d['best_vs_C0']['relative_delta']
 current_clean=sha(f'diffusion_experiments/round16r_cleantrain/assets/seed{s}_train_clean.npz')
 current_ctx=sha(f'diffusion_experiments/round16_fcbrd/assets/seed{s}_validation.npz')
 f=bool(d['frozen_reference']['hashes_unchanged'] and current_clean==param['seeds'][str(s)]['clean_asset_sha256'] and current_ctx==param['seeds'][str(s)]['validation_context_sha256'])
 return {'C':bool(hist['seeds'][str(s)]['PASS']),'F':f,
  'G':bool(m['Delta_margin']['mean']>0 and m['fraction_Delta_margin_gt0']>=.55),
  'H':bool(m['history_advantage']['mean']>0 and m['fraction_true_gt_shuffled']>=.55 and d['best_vs_C0']['U']>d['history_control']['vs_C0']['U']),
  'U_seed':bool(d['best_vs_C0']['U']>.001 and min(rel[k] for k in ('R10','N10','R20','N20'))>=-.005),
  'B':bool(d['bound']['max_angle']<=5.01),'RB':bool(rnd[s][v]['PASS'])}
summary={'protocol':'ROUND17_TADGD_V1','TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False,'validation_verdict':'ROUND17_FAIL','variants':{}}
for v in ('D2','D3'):
 per={str(s):gates_for(v,s) for s in SEEDS}; us=[rows[s][v]['best_vs_C0']['U'] for s in SEEDS]
 two={k:all(per[str(s)][k] for s in SEEDS) for k in ('C','F','G','H','B','RB')}
 two['U']=all(per[str(s)]['U_seed'] for s in SEEDS) and sum(us)/2>=.002
 summary['variants'][v]={'per_seed':per,'U_true':{str(s):rows[s][v]['best_vs_C0']['U'] for s in SEEDS},'mean_U_true':sum(us)/2,'two_seed_gates':two,'all_required_pass':all(two.values())}
summary['D1_U']={str(s):rows[s]['D1']['best_vs_C0']['U'] for s in SEEDS}
summary['D0_U']={str(s):d0[s]['vs_C0']['U'] for s in SEEDS}
summary['expansion_open']=False;summary['seed1001_not_opened']=True;summary['seed1002_not_opened']=True
summary['comparison']={'D2_gt_D1_both_seeds':all(rows[s]['D2']['best_vs_C0']['U']>rows[s]['D1']['best_vs_C0']['U'] for s in SEEDS),'D3_gt_D2_both_seeds':all(rows[s]['D3']['best_vs_C0']['U']>rows[s]['D2']['best_vs_C0']['U'] for s in SEEDS)}
summary['test_lock']={'authorized_by_user':True,'status':'LOCKED_BEFORE_TEST','purpose':'EXPLORATORY_BABY_TEST_AFTER_ROUND17_VALIDATION_FAIL','variant':'D2','reason':'D2 is the preregistered primary target-aware variant; D3 is lower-U on both Validation seeds and fails RB on both, while D2 passes RB on both. Selection remains FAIL/ineligible for method PASS.','seed999_checkpoint_epoch':rows[999]['D2']['best_epoch'],'seed1000_checkpoint_epoch':rows[1000]['D2']['best_epoch'],'test_must_not_change_variant_checkpoint_or_hyperparameters':True}
(E/'ROUND17_GATE_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({'D2':summary['variants']['D2'],'D3':summary['variants']['D3'],'lock':summary['test_lock']},indent=2))
