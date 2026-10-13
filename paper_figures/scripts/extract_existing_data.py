from __future__ import annotations
import csv, json, subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
RAW=ROOT/'paper_figures'/'raw_data'
MAIN_BRANCH='main'
R26_BRANCH='exp/round26-preference-directed-diffusion-20261010'
R31_BRANCH='exp/round31-round26-baby-test-confirmation-20261011'
R32_BRANCH='exp/round32-round26-parameter-sweep-20261011'
R33_BRANCH='exp/round33-seed1000-cross-domain-20261012'
BRANCHES=[MAIN_BRANCH,R26_BRANCH,R31_BRANCH,R32_BRANCH,R33_BRANCH]
COMMITS={b:subprocess.check_output(['git','rev-parse',b],cwd=ROOT,text=True).strip() for b in BRANCHES}
NOW=datetime.now(timezone.utc).isoformat()
PROV=[]

def git_json(branch,path):
    raw=subprocess.check_output(['git','show',f'{COMMITS[branch]}:{path}'],cwd=ROOT)
    return json.loads(raw)

def local_json(path): return json.loads((ROOT/path).read_text())

def write_csv(rel, rows, fields):
    path=RAW/rel; path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    return str(Path('paper_figures/raw_data')/rel)

def provenance(fig,data_file,dataset,split,method,branch,path,key,notes='',seed=1000):
    PROV.append(dict(figure_id=fig,data_file=data_file,dataset=dataset,seed=seed,split=split,
      method_or_variant=method,source_branch=branch,source_commit=COMMITS[branch],source_path=path,
      source_key=key,extraction_script='paper_figures/scripts/extract_existing_data.py',generated_at=NOW,notes=notes))

def rel_u(new,base,keys=('R10','N10','R20','N20')):
    return sum((new[k]-base[k])/base[k] for k in keys)/len(keys)

# current main results
cur_path='docs/evidence/current/THREE_DOMAIN_SEED1000_RESULTS.json'
current=local_json(cur_path)
name={'baby':'Baby','sports':'Sports','elec':'Electronics'}
rows=[]
for ds in ('baby','sports','elec'):
    u=current[ds]['U4']
    rows.append(dict(dataset=name[ds],colift_vs_msca_u4=100*u['colift_vs_msca'],pdbr_vs_colift_u4=100*u['diffusion_vs_colift'],full_vs_msca_u4=100*u['full_vs_msca']))
f=write_csv(Path('performance/fig01_three_domain_module_gain.csv'),rows,['dataset','colift_vs_msca_u4','pdbr_vs_colift_u4','full_vs_msca_u4'])
for ds in ('baby','sports','elec'): provenance('Fig01',f,name[ds],'Test','MSCA/CoLiftRec/Full',MAIN_BRANCH,cur_path,f'{ds}.U4')

for fig,ds in [('Fig02','baby'),('Fig03','sports'),('Fig04','elec')]:
    rows=[]
    for metric,label in [('R10','R@10'),('N10','N@10'),('R20','R@20'),('N20','N@20')]:
        a=current[ds]['MSCA'][metric]; b=current[ds]['CoLiftRec'][metric]; c=current[ds]['Full'][metric]
        rows.append(dict(metric=label,colift_vs_msca=100*(b-a)/a,pdbr_vs_colift=100*(c-b)/b))
    rel=Path(f'performance/{fig.lower()}_{ds}_primary_metric_gain.csv')
    f=write_csv(rel,rows,['metric','colift_vs_msca','pdbr_vs_colift'])
    provenance(fig,f,name[ds],'Test','primary metric gains',MAIN_BRANCH,cur_path,f'{ds}.MSCA/CoLiftRec/Full')

# bootstrap CIs
r32_m25_path='diffusion_experiments/round32_round26_parameter_sweep/evidence/ROUND32_M25_SEED1000.json'
r32_m25=git_json(R32_BRANCH,r32_m25_path)
r33_test_paths={'sports':'diffusion_experiments/round33_seed1000_cross_domain/evidence/ROUND33_SPORTS_TEST.json','elec':'diffusion_experiments/round33_seed1000_cross_domain/evidence/ROUND33_ELEC_TEST.json'}
ci_rows=[dict(dataset='Baby',point_u4=100*r32_m25['bootstrap']['point_U'],ci95_low=100*r32_m25['bootstrap']['ci95'][0],ci95_high=100*r32_m25['bootstrap']['ci95'][1],positive_replicate_fraction=r32_m25['bootstrap']['positive_replicate_fraction'])]
provenance('support', 'paper_figures/raw_data/performance/pdbr_bootstrap_ci.csv','Baby','Test','M25',R32_BRANCH,r32_m25_path,'bootstrap')
for ds in ('sports','elec'):
    d=git_json(R33_BRANCH,r33_test_paths[ds]); b=d['bootstrap_Diffusion_vs_CoLiftRec']
    ci_rows.append(dict(dataset=name[ds],point_u4=100*b['point_U'],ci95_low=100*b['ci95'][0],ci95_high=100*b['ci95'][1],positive_replicate_fraction=b['positive_replicate_fraction']))
    provenance('support','paper_figures/raw_data/performance/pdbr_bootstrap_ci.csv',name[ds],'Test','M25',R33_BRANCH,r33_test_paths[ds],'bootstrap_Diffusion_vs_CoLiftRec')
write_csv(Path('performance/pdbr_bootstrap_ci.csv'),ci_rows,['dataset','point_u4','ci95_low','ci95_high','positive_replicate_fraction'])

# Round26 base for sensitivity
r26_b0_path='diffusion_experiments/round26_preference_directed_diffusion/evidence/ROUND26_B0_SEED1000.json'
r26_d3_path='diffusion_experiments/round26_preference_directed_diffusion/evidence/ROUND26_D3_SEED1000.json'
r31_b0_path='diffusion_experiments/round31_round26_baby_test_confirmation/evidence/ROUND31_B0_SEED1000_TEST.json'
r31_d3_path='diffusion_experiments/round31_round26_baby_test_confirmation/evidence/ROUND31_D3_SEED1000_TEST.json'
r26_b0=git_json(R26_BRANCH,r26_b0_path); r26_d3=git_json(R26_BRANCH,r26_d3_path)
r31_b0=git_json(R31_BRANCH,r31_b0_path); r31_d3=git_json(R31_BRANCH,r31_d3_path)
base_val_u=rel_u(r26_d3['best_evaluation']['colift'],r26_b0['best_evaluation']['colift'])
base_test_u=rel_u(r31_d3['canonical_metrics']['colift'],r31_b0['canonical_metrics']['colift'])

def sensitivity(fig,filename,variants,base_value):
    rows=[]
    for variant,value in variants:
        if variant=='R26_BASE':
            row=dict(variant=variant,parameter_value=value,validation_u4=100*base_val_u,test_u4=100*base_test_u,
                     validation_primary_positive=sum(r26_d3['best_evaluation']['colift'][k]>r26_b0['best_evaluation']['colift'][k] for k in ('R10','N10','R20','N20')),
                     test_primary_positive=sum(r31_d3['canonical_metrics']['colift'][k]>r31_b0['canonical_metrics']['colift'][k] for k in ('R10','N10','R20','N20')),best_epoch=r26_d3['best_epoch'])
            rows.append(row)
            provenance(fig,f'paper_figures/raw_data/sensitivity/{filename}','Baby','Validation','R26_BASE',R26_BRANCH,r26_d3_path,'best_evaluation.colift; compared to ROUND26_B0')
            provenance(fig,f'paper_figures/raw_data/sensitivity/{filename}','Baby','Test','R26_BASE',R31_BRANCH,r31_d3_path,'canonical_metrics.colift; compared to ROUND31_B0')
        else:
            p=f'diffusion_experiments/round32_round26_parameter_sweep/evidence/ROUND32_{variant}_SEED1000.json'
            d=git_json(R32_BRANCH,p)
            rows.append(dict(variant=variant,parameter_value=value,validation_u4=100*d['Validation_vs_B0']['U'],test_u4=100*d['Test_vs_B0']['U'],
                             validation_primary_positive=d['Validation_vs_B0']['primary_positive_count'],test_primary_positive=d['Test_vs_B0']['primary_positive_count'],best_epoch=d['best_epoch']))
            provenance(fig,f'paper_figures/raw_data/sensitivity/{filename}','Baby','Validation/Test',variant,R32_BRANCH,p,'Validation_vs_B0; Test_vs_B0')
    write_csv(Path('sensitivity')/filename,rows,['variant','parameter_value','validation_u4','test_u4','validation_primary_positive','test_primary_positive','best_epoch'])

sensitivity('Fig05','fig05_lambda_hn_sensitivity.csv',[('L10',.10),('L15',.15),('R26_BASE',.20),('L25',.25)],.20)
sensitivity('Fig06','fig06_half_margin_sensitivity.csv',[('M25',.25),('M35',.35),('R26_BASE',.50),('M65',.65)],.50)
sensitivity('Fig07','fig07_theta_sensitivity.csv',[('T3',3),('T4',4),('R26_BASE',5),('T6',6)],5)

# training curves
r32_train_path='diffusion_experiments/round32_round26_parameter_sweep/evidence/ROUND32_M25_SEED1000_TRAIN.json'
r32_train=git_json(R32_BRANCH,r32_train_path)
for fig,ds,bbranch,bpath,fbranch,fpath in [
 ('Fig08','baby',R26_BRANCH,r26_b0_path,R32_BRANCH,r32_train_path),
 ('Fig09','sports',R33_BRANCH,'diffusion_experiments/round33_seed1000_cross_domain/evidence/ROUND33_SPORTS_B0_TRAIN.json',R33_BRANCH,'diffusion_experiments/round33_seed1000_cross_domain/evidence/ROUND33_SPORTS_M25_TRAIN.json'),
 ('Fig10','elec',R33_BRANCH,'diffusion_experiments/round33_seed1000_cross_domain/evidence/ROUND33_ELEC_B0_TRAIN.json',R33_BRANCH,'diffusion_experiments/round33_seed1000_cross_domain/evidence/ROUND33_ELEC_M25_TRAIN.json')]:
    b=r26_b0 if ds=='baby' else git_json(bbranch,bpath); q=r32_train if ds=='baby' else git_json(fbranch,fpath)
    bm={int(x['epoch']):float(x['evaluation']['colift']['R20']) for x in b['trajectory']}
    qm={int(x['epoch']):float(x['evaluation']['colift']['R20']) for x in q['trajectory']}
    epochs=sorted(set(bm)&set(qm)); rows=[dict(epoch=e,baseline_colift_r20=bm[e],pdbr_colift_r20=qm[e]) for e in epochs]
    filename=f'{fig.lower()}_{ds}_validation_r20_curve.csv'; f=write_csv(Path('training')/filename,rows,['epoch','baseline_colift_r20','pdbr_colift_r20'])
    provenance(fig,f,name[ds],'Validation','Baseline',bbranch,bpath,'trajectory[*].evaluation.colift.R20')
    provenance(fig,f,name[ds],'Validation','PDBR',fbranch,fpath,'trajectory[*].evaluation.colift.R20')

# Baby mechanism dynamics
rows11=[]; rows12=[]
for rec in r32_train['trajectory']:
    if 'mechanism' not in rec: continue
    m=rec['mechanism']; e=int(rec['epoch'])
    rows11.append(dict(epoch=e,a_true_median=m['A_true']['median'],a_shuf_median=m['A_shuf']['median']))
    rows12.append(dict(epoch=e,p_delta_positive=m['DeltaA']['p_gt_zero']))
f=write_csv(Path('mechanism/fig11_preference_alignment.csv'),rows11,['epoch','a_true_median','a_shuf_median'])
provenance('Fig11',f,'Baby','Validation/Train diagnostic','M25',R32_BRANCH,r32_train_path,'trajectory[*].mechanism.A_true.median/A_shuf.median')
f=write_csv(Path('mechanism/fig12_deltaA_positive_fraction.csv'),rows12,['epoch','p_delta_positive'])
provenance('Fig12',f,'Baby','Validation/Train diagnostic','M25',R32_BRANCH,r32_train_path,'trajectory[*].mechanism.DeltaA.p_gt_zero')

# best-epoch geometry across domains
geom=[]
for ds,branch,path,d in [
 ('baby',R32_BRANCH,r32_train_path,r32_train),
 ('sports',R33_BRANCH,'diffusion_experiments/round33_seed1000_cross_domain/evidence/ROUND33_SPORTS_M25_TRAIN.json',git_json(R33_BRANCH,'diffusion_experiments/round33_seed1000_cross_domain/evidence/ROUND33_SPORTS_M25_TRAIN.json')),
 ('elec',R33_BRANCH,'diffusion_experiments/round33_seed1000_cross_domain/evidence/ROUND33_ELEC_M25_TRAIN.json',git_json(R33_BRANCH,'diffusion_experiments/round33_seed1000_cross_domain/evidence/ROUND33_ELEC_M25_TRAIN.json'))]:
    m=d['mechanism_at_best']
    geom.append(dict(dataset=name[ds],best_epoch=d['best_epoch'],applied_refinement_fraction=m['applied_refinement_fraction'],cap_hit_fraction=m['cap_hit_fraction'],target_reachable_fraction=m['target_reachable_fraction'],angle_deg_mean=m['angle_deg']['mean'],angle_deg_p90=m['angle_deg']['p90'],actual_score_shift_mean=m['actual_score_shift']['mean'],actual_score_shift_p90=m['actual_score_shift']['p90']))
    provenance('Fig13','paper_figures/raw_data/mechanism/fig13_refinement_geometry.csv',name[ds],'Validation/Train diagnostic','M25',branch,path,'mechanism_at_best')
write_csv(Path('mechanism/fig13_refinement_geometry.csv'),geom,['dataset','best_epoch','applied_refinement_fraction','cap_hit_fraction','target_reachable_fraction','angle_deg_mean','angle_deg_p90','actual_score_shift_mean','actual_score_shift_p90'])

# early stopping efficiency
es_path='docs/evidence/current/EARLY_STOPPING_RETRO_AUDIT.json'; es=local_json(es_path); erows=[]
for ds in ('Baby','Sports','Electronics'):
 for br,label in [('B0','Baseline'),('M25','PDBR')]:
  x=es['historical_60_epoch_replay'][ds][br]; erows.append(dict(dataset=ds,branch=label,epochs_run=x['epochs_run'],best_epoch=x['global_best_epoch'],max_epochs=60))
f=write_csv(Path('efficiency/fig14_early_stopping_efficiency.csv'),erows,['dataset','branch','epochs_run','best_epoch','max_epochs'])
provenance('Fig14',f,'Baby/Sports/Electronics','Validation replay','Baseline/PDBR',MAIN_BRANCH,es_path,'historical_60_epoch_replay')

# optional status marker
(RAW/'README.md').write_text('''# Raw data\n\nAll current paper-facing data use seed1000 unless explicitly stated.\n\n- Performance, sensitivity, training, mechanism and efficiency CSVs are generated by `scripts/extract_existing_data.py`.\n- `PROVENANCE.csv` maps every extracted dataset to branch/commit/path/key.\n- CoLiftRec seed1000 modality ablation: `OPTIONAL_DATA_NOT_YET_FROZEN`.\n- Case study: `OPTIONAL_DATA_NOT_YET_FROZEN`.\n- Fig15/16 rank-band data are produced only by deterministic TRAIN-side diagnostic replay if historical per-band evidence is unavailable.\n''')

fields=['figure_id','data_file','dataset','seed','split','method_or_variant','source_branch','source_commit','source_path','source_key','extraction_script','generated_at','notes']
write_csv(Path('PROVENANCE.csv'),PROV,fields)
print(f'generated {len(PROV)} provenance rows')
