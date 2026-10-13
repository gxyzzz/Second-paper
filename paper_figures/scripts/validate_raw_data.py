from __future__ import annotations
import csv, json, math, subprocess
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[2]
RAW=ROOT/'paper_figures'/'raw_data'
errors=[]
def fail(msg): errors.append(msg)
def rows(path):
    with path.open(newline='') as f: return list(csv.DictReader(f))
for path in RAW.rglob('*.csv'):
    try: rr=rows(path)
    except Exception as e: fail(f'unreadable {path}: {e}'); continue
    if not rr: fail(f'empty csv {path}')
    for i,row in enumerate(rr,2):
        for k,v in row.items():
            if v in ('',None): continue
            try: x=float(v)
            except ValueError: continue
            if not math.isfinite(x): fail(f'nonfinite {path}:{i}:{k}={v}')
prov=rows(RAW/'PROVENANCE.csv')
valid_ds={'Baby','Sports','Electronics','Baby/Sports/Electronics'}
for r in prov:
    if r['seed']!='1000': fail(f"non-seed1000 provenance: {r}")
    if r['dataset'] not in valid_ds: fail(f"invalid dataset {r['dataset']}")
    p=ROOT/r['source_path']
    if not p.exists():
        ref=f"{r['source_commit']}:{r['source_path']}"
        if subprocess.run(['git','cat-file','-e',ref],cwd=ROOT).returncode!=0: fail(f'unresolvable source {ref}')
cur=json.loads((ROOT/'docs/evidence/current/THREE_DOMAIN_SEED1000_RESULTS.json').read_text())
fig1=rows(RAW/'performance/fig01_three_domain_module_gain.csv')
for r in fig1:
    key={'Baby':'baby','Sports':'sports','Electronics':'elec'}[r['dataset']]; u=cur[key]['U4']
    expected=[100*u['colift_vs_msca'],100*u['diffusion_vs_colift'],100*u['full_vs_msca']]
    actual=[float(r['colift_vs_msca_u4']),float(r['pdbr_vs_colift_u4']),float(r['full_vs_msca_u4'])]
    if max(abs(a-b) for a,b in zip(actual,expected))>1e-10: fail(f'Fig01 mismatch {r["dataset"]}')
checks=[('fig05_lambda_hn_sensitivity.csv','R26_BASE',.20),('fig06_half_margin_sensitivity.csv','M25',.25),('fig07_theta_sensitivity.csv','R26_BASE',5.0)]
for fn,var,val in checks:
    rr=rows(RAW/'sensitivity'/fn); hit=[r for r in rr if r['variant']==var]
    if len(hit)!=1 or abs(float(hit[0]['parameter_value'])-val)>1e-12: fail(f'parameter marker mismatch {fn}')
    vals=[float(r['parameter_value']) for r in rr]
    if len(vals)!=len(set(vals)): fail(f'duplicate sensitivity values {fn}')
for path in (RAW/'training').glob('*.csv'):
    ep=[int(r['epoch']) for r in rows(path)]
    if ep!=sorted(ep) or len(ep)!=len(set(ep)): fail(f'non-monotonic epochs {path}')

# optional rank-band diagnostic integrity
band_path=RAW/'mechanism/fig15_16_band_diagnostics.csv'
if band_path.exists():
    br=rows(band_path)
    if [r['band'] for r in br] != ['Easy','Medium','Hard']:
        fail('rank-band order/content mismatch')
    for r in br:
        if int(float(r['sample_count'])) <= 0: fail(f"empty band diagnostic {r['band']}")
        for key in ('P_DeltaA_positive','applied_refinement_fraction','cap_hit_fraction'):
            x=float(r[key])
            if not (0.0 <= x <= 1.0): fail(f"invalid fraction {r['band']}:{key}={x}")
    ids={r['figure_id'] for r in prov}
    if not {'Fig15','Fig16'} <= ids: fail('missing Fig15/Fig16 provenance rows')

cfg=yaml.safe_load((ROOT/'src/configs/model/CoLiftRecDiffusion.yaml').read_text())['common']['diffusion']
if not (float(cfg['lambda_hn'])==.20 and float(cfg['half_margin'])==.25 and float(cfg['theta_max_deg'])==5.0): fail('current M25 config mismatch')
if errors:
    print('VALIDATION FAIL')
    for e in errors: print('-',e)
    raise SystemExit(1)
print(f'VALIDATION PASS: {len(list(RAW.rglob("*.csv")))} CSV files; {len(prov)} provenance rows')
