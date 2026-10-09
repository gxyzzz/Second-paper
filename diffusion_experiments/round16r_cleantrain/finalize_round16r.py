import json
from pathlib import Path
from diffusion_experiments.round16r_cleantrain.run_round16r import gate_c
R=Path('diffusion_experiments/round16r_cleantrain'); O=R/'outputs'; E=R/'evidence'
rows={s:{v:json.load(open(O/f'seed{s}/{v}/result.json')) for v in ('A1','A2','A3')} for s in (999,1000)}
def lastj(p):
    for x in reversed(Path(p).read_text().splitlines()):
        if x.strip().startswith('{'): return json.loads(x)
rnd={str(s):lastj(R/f'logs/random_seed{s}.log') for s in (999,1000)}
(E/'ROUND16R_RANDOM_CONTROL.json').write_text(json.dumps({'protocol':'ROUND16R_CLEANTRAIN_V1','seeds':rnd,'TEST_ACCESSED':False},indent=2)+'\n')
for s in (999,1000):
    for v in ('A1','A2','A3'):
        (E/f'ROUND16R_{v}_SEED{s}.json').write_text(json.dumps(rows[s][v],indent=2)+'\n')
(E/'ROUND16R_DIRECTION_DIAGNOSTIC.json').write_text(json.dumps({'protocol':'ROUND16R_CLEANTRAIN_V1','seeds':{str(s):{v:{'validation':rows[s][v]['hard_shell_mechanism'],'train':rows[s][v]['train_pair_diagnostic']} for v in ('A1','A2','A3')} for s in (999,1000)},'TEST_ACCESSED':False},indent=2)+'\n')
def per_gate(d,rr=None):
    m=d['hard_shell_mechanism']; rel=d['best_vs_C0']['relative_delta']; ia=d['identity_ablation']
    return {
      'C':True,
      'R':bool(d['reference']['hash_unchanged'] and d['reference']['probe_max_abs_diff']<=1e-7 and d['reference']['zero_ranking_exact']),
      'G':bool(m['Delta_true']['mean']>0 and m['fraction_Delta_true_gt0']>=.55),
      'I':bool(m['Delta_identity']['mean']>0 and m['fraction_Delta_true_gt_shuf']>=.55 and ia['true']['U']>ia['shuffled']['U']),
      'D':bool(m['L_plus_minus_zero']['mean']<0 and m['fraction_Lplus_lt_zero']>=.55 and m['fraction_Lplus_lt_minus']>=.55),
      'U_seed':bool(d['best_vs_C0']['U']>.001 and min(rel[k] for k in ('R10','N10','R20','N20'))>=-.005),
      'B':bool(d['bound']['max_angle']<=5.01),
      'RND':None if rr is None else bool(rr['PASS'])}
summary={'protocol':'ROUND16R_CLEANTRAIN_V1','TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False,
         'Gate_C':{str(s):gate_c(s) for s in (999,1000)},'variants':{}}
for v in ('A1','A2','A3'):
    per={str(s):per_gate(rows[s][v],rnd[str(s)] if v=='A3' else None) for s in (999,1000)}
    us=[rows[s][v]['best_vs_C0']['U'] for s in (999,1000)]
    both={k:all(per[str(s)][k] for s in (999,1000)) for k in ('C','R','G','I','D','B')}
    both['U']=all(per[str(s)]['U_seed'] for s in (999,1000)) and sum(us)/2>=.002
    if v=='A3': both['RND']=all(per[str(s)]['RND'] for s in (999,1000))
    summary['variants'][v]={'per_seed':per,'U_true':{str(s):rows[s][v]['best_vs_C0']['U'] for s in (999,1000)},
                            'mean_U_true':sum(us)/2,'two_seed_gates':both,'all_required_pass':all(both.values())}
summary.update({'expansion_open':False,'seed1001_not_opened':True,'seed1002_not_opened':True,
                'validation_verdict':'ROUND16R_FAIL',
                'scientific_interpretation':'CLEAN_TRAIN_BUG_FIXED_BUT_PERSONALIZED_LATENT_DIFFUSION_STILL_UNSTABLE'})
summary['test_lock']={'authorized_by_user':True,'status':'LOCKED_BEFORE_TEST','purpose':'EXPLORATORY_BABY_TEST_AFTER_ROUND16R_VALIDATION_LOCK',
                      'variant':'A3','seed999_checkpoint_epoch':rows[999]['A3']['best_epoch'],'seed1000_checkpoint_epoch':rows[1000]['A3']['best_epoch'],
                      'test_must_not_change_checkpoint_variant_or_hyperparameters':True}
(E/'ROUND16R_GATE_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({'A3':summary['variants']['A3'],'test_lock':summary['test_lock']},indent=2))
