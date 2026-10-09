import json
from pathlib import Path
E=Path('diffusion_experiments/round18_diffusion_hard_negative/evidence')
S=(999,1000); V=('F1','F2','F3','F4'); P=('R10','N10','R20','N20')
def L(p): return json.load(open(p))
f0={s:L(E/f'ROUND18_F0_SEED{s}.json') for s in S}
r={s:{v:L(E/f'ROUND18_{v}_SEED{s}.json') for v in V} for s in S}
q=L(E/'ROUND18_DIFFUSION_GENERATOR_QUALITY.json')
def U(s,v): return r[s][v]['final_colift_vs_F0']['U']
def UB(s,v): return r[s][v]['final_backbone_vs_F0']['U']
inc={
 'F1_minus_F0':{str(s):U(s,'F1') for s in S},
 'F2_minus_F1':{str(s):U(s,'F2')-U(s,'F1') for s in S},
 'F3_minus_F2':{str(s):U(s,'F3')-U(s,'F2') for s in S},
 'F4_minus_F3':{str(s):U(s,'F4')-U(s,'F3') for s in S},
}
for x in inc.values(): x['mean']=sum(x[str(s)] for s in S)/2
u1=sum(U(s,'F1') for s in S)/2
hc=inc['F2_minus_F1']['mean']>0
dif=all(U(s,'F4')>U(s,'F3') for s in S) and inc['F4_minus_F3']['mean']>=.001
abs_seed={str(s):bool(U(s,'F4')>.001 and min(r[s]['F4']['final_colift_vs_F0']['relative_delta'][k] for k in P)>=-.005) for s in S}
abs_gate=all(abs_seed.values()) and sum(U(s,'F4') for s in S)/2>=.002
stab=all(U(s,'F4')>0 for s in S)
inf=all(r[s][v]['diffusion_inference_calls']==0 for s in S for v in V)
cl_seed={}
for s in S:
 b=r[s]['F4']['final_backbone_metrics'];c=r[s]['F4']['final_colift_metrics'];cl_seed[str(s)]={'primary_positive_count':sum(c[k]>b[k] for k in P),'PASS':sum(c[k]>b[k] for k in P)>=3}
cl=all(x['PASS'] for x in cl_seed.values())
summary={'protocol':'ROUND18_HCD_HNC_V1','validation_verdict':'ROUND18_FAIL','TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False,
 'U_colift':{v:{str(s):U(s,v) for s in S}|{'mean':sum(U(s,v) for s in S)/2} for v in V},
 'U_backbone':{v:{str(s):UB(s,v) for s in S}|{'mean':sum(UB(s,v) for s in S)/2} for v in V},
 'incremental_decomposition':inc,
 'gates':{'DG':bool(q['Gate_DG_PASS']),'INF':inf,'U0':u1>-.002,'HC':hc,'SIM_delta_mean':inc['F3_minus_F2']['mean'],'DIF':dif,'ABS':abs_gate,'ABS_per_seed':abs_seed,'STAB':stab,'CL':cl,'CL_per_seed':cl_seed},
 'round18_pass':False,'expansion_open':False,'seed1001_not_opened':True,'seed1002_not_opened':True,
 'test_lock':{'authorized_by_user':True,'status':'LOCKED_BEFORE_TEST','validation_verdict':'ROUND18_FAIL','method_checkpoint':'F4 epoch4','matched_control_checkpoint':'F3 epoch4','seeds':[999,1000],'purpose':'exploratory Baby Test after Validation lock; Test cannot change method, checkpoint, bands, or hyperparameters','no_post_test_tuning':True}}
summary['round18_pass']=all(summary['gates'][k] for k in ('DG','INF','DIF','ABS','STAB','CL'))
(E/'ROUND18_GATE_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
