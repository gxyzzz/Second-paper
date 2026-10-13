from __future__ import annotations
import inspect, json, sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from modules import preference_diffusion as pd
from pipelines.dataset_config import load_dataset_config
from pipelines.preference_pipeline import DomainEvaluator

EXPECTED_COLIFT = {
    'baby': {'text': (1.0,.25), 'attribute': (.75,.15), 'visual': (.25,.025)},
    'sports': {'text': (0.0,.25), 'attribute': (1.0,.20), 'visual': (1.0,.025)},
    'elec': {'text': (1.0,.20), 'attribute': (1.0,.20), 'visual': (.25,.025)},
}
EXPECTED_MSCA = {
    'baby': {'n_layers':[2], 'fusion_coeff':[.4], 'cl_weight':[.005], 'reg_weight':[3e-7]},
    'sports': {'n_layers':[3], 'fusion_coeff':[.3], 'cl_weight':[.005], 'reg_weight':[5e-8]},
    'elec': {'n_layers':[4], 'fusion_coeff':[.2], 'cl_weight':[.01], 'reg_weight':[5e-10]},
}
RESULTS = {
 'baby': {'MSCA': {'R10':.067720,'N10':.037031,'R20':.103784,'N20':.046318,'R50':.174483,'N50':.060638},
          'CoLiftRec': {'R10':.071226,'N10':.038945,'R20':.107293,'N20':.048269,'R50':.179539,'N50':.062955},
          'Full': {'R10':.072726,'N10':.039750,'R20':.110479,'N20':.049464,'R50':.179375,'N50':.063351},
          'U4': {'colift_vs_msca':.0448,'diffusion_vs_colift':.0240,'full_vs_msca':.0699}, 'primary':'4/4','overall':'5/6'},
 'sports': {'MSCA': {'R10':.078633,'N10':.042836,'R20':.117032,'N20':.052734,'R50':.189671,'N50':.067531},
            'CoLiftRec': {'R10':.083048,'N10':.044891,'R20':.124388,'N20':.055527,'R50':.195963,'N50':.070075},
            'Full': {'R10':.086167,'N10':.046832,'R20':.126438,'N20':.057239,'R50':.197375,'N50':.071669},
            'U4': {'colift_vs_msca':.0550,'diffusion_vs_colift':.0320,'full_vs_msca':.0887}, 'primary':'4/4','overall':'6/6'},
 'elec': {'MSCA': {'R10':.049395,'N10':.027998,'R20':.072552,'N20':.033973,'R50':.116521,'N50':.042912},
          'CoLiftRec': {'R10':.050878,'N10':.028767,'R20':.074946,'N20':.034986,'R50':.119535,'N50':.044051},
          'Full': {'R10':.053055,'N10':.029549,'R20':.078976,'N20':.036249,'R50':.123687,'N50':.045357},
          'U4': {'colift_vs_msca':.0301,'diffusion_vs_colift':.0400,'full_vs_msca':.0713}, 'primary':'4/4','overall':'6/6'},
}

def close(a,b,tol=1e-12):
    if isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)):
        return len(a)==len(b) and all(close(x,y,tol) for x,y in zip(a,b))
    try:
        return abs(float(a)-float(b)) <= tol
    except (TypeError,ValueError):
        return a == b

def main():
    checks = {}
    overall = yaml.safe_load((ROOT/'src/configs/overall.yaml').read_text())
    checks['seed1000'] = overall['seed'] == [1000]
    for ds in ('baby','sports','elec'):
        cfg = load_dataset_config(ds); d = cfg['diffusion']; c = cfg['coliftrec']
        checks[f'{ds}_m25'] = (d['lambda_hn']==.20 and d['half_margin']==.25 and d['theta_max_deg']==5.0 and
            d['local_t']==[1,2,3,4,5] and d['warmup_epochs']==10 and d['formal_epochs']==60 and
            d['active_steps']==2 and d['inference_enabled'] is False)
        checks[f'{ds}_colift'] = all(close(c[m]['lambda'],EXPECTED_COLIFT[ds][m][0]) and
            close(c[m]['alpha'],EXPECTED_COLIFT[ds][m][1]) for m in EXPECTED_COLIFT[ds])
        msca = yaml.safe_load((ROOT/f'src/configs/model/MSCA/{ds}.yaml').read_text())
        checks[f'{ds}_msca_override'] = all(close(msca[k], v) for k,v in EXPECTED_MSCA[ds].items())
    checks['bands'] = pd.BANDS == ((60,100),(30,60),(10,30))
    checks['curriculum'] = [pd.curriculum_count(x) for x in (0,9,10,19,20,29,30,59)] == [0,0,1,1,2,2,3,3]
    source = inspect.getsource(DomainEvaluator.evaluate)
    checks['test_diffusion_calls_zero'] = ('diffusion_inference_calls' in source and
        'predict_directions' not in source and 'warmup_step' not in source)
    runtime = '\n'.join(p.read_text(errors='ignore') for p in (ROOT/'src').rglob('*.py'))
    checks['no_experiment_runtime_dependency'] = 'diffusion_experiments' not in runtime
    out = {'status':'PASS' if all(checks.values()) else 'FAIL', 'checks':checks, 'frozen_results':RESULTS,
           'notes':['Implementation parity audit; no new parameter search or Test-driven selection.',
                    'Formal Test evaluator does not instantiate or call Diffusion.']}
    (ROOT/'docs/evidence/current/MAIN_RUNTIME_PARITY_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n')
    (ROOT/'docs/evidence/current/THREE_DOMAIN_SEED1000_RESULTS.json').write_text(json.dumps(RESULTS,indent=2)+'\n')
    print(json.dumps(out,indent=2))
    if out['status'] != 'PASS': raise SystemExit(1)

if __name__ == '__main__': main()
