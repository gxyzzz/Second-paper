from __future__ import annotations
import csv,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,str(ROOT))
from paper_experiments.common import ALL, PRIMARY, checkpoint_info, relative_u4

ROUND33='exp/round33-seed1000-cross-domain-20261012'
ROUND32='exp/round32-round26-parameter-sweep-20261011'

def git_json(ref,path):
    raw=subprocess.check_output(['git','show',f'{ref}:{path}'],cwd=ROOT,text=True); return json.loads(raw)

def c5(dataset):
    x=json.loads((ROOT/f'paper_experiments/coliftrec_ablation/evidence/COLIFTREC_ABLATION_{dataset.upper()}.json').read_text())
    return next(r for r in x['variants'] if r['variant']=='C5')

def p1(dataset):
    if dataset=='baby':
        x=git_json(ROUND32,'diffusion_experiments/round32_round26_parameter_sweep/evidence/ROUND32_M25_SEED1000.json')
        info=checkpoint_info(dataset,'p1')
        return x['Test'],int(info['epoch']),info['sha256']
    name='SPORTS' if dataset=='sports' else 'ELEC'
    x=git_json(ROUND33,f'diffusion_experiments/round33_seed1000_cross_domain/evidence/ROUND33_{name}_TEST.json')
    info=checkpoint_info(dataset,'p1')
    return x['M25_result']['colift'],int(info['epoch']),info['sha256']

def row(dataset,variant,name,m,best_epoch=None,sha=None):
    base=c5(dataset); bm={k:float(base[k]) for k in ALL}
    return {'dataset':dataset,'variant':variant,'variant_name':name,'best_epoch':best_epoch,
            **{k:float(m[k]) for k in ALL},'u4_vs_colift_baseline':relative_u4(m,bm),
            'primary_positive_count':sum(float(m[k])>bm[k] for k in PRIMARY),
            'overall_positive_count':sum(float(m[k])>bm[k] for k in ALL),
            'diffusion_inference_calls':0,'checkpoint_sha':sha or ''}

def main():
    rows=[]
    for d in ['baby','sports','elec']:
        base=c5(d); binfo=checkpoint_info(d,'b0'); rows.append(row(d,'P0','CoLiftRec baseline',base,best_epoch=binfo['epoch'],sha=binfo['sha256']))
        m,e,s=p1(d); rows.append(row(d,'P1','Full PDBR (M25)',m,e,s))
        variants=['P2','P3','P4']+(['P5','P6'] if d=='baby' else [])
        for v in variants:
            p=ROOT/f'paper_experiments/pdbr_ablation/evidence/{d.upper()}_{v}.json'
            if not p.exists(): continue
            x=json.loads(p.read_text()); rows.append(x['test'])
    out=ROOT/'paper_experiments/pdbr_ablation/evidence/PDBR_ABLATION_SUMMARY.csv'
    fields=['dataset','variant','variant_name','best_epoch',*ALL,'u4_vs_colift_baseline','primary_positive_count','overall_positive_count','diffusion_inference_calls','checkpoint_sha']
    with out.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    print('wrote',out,'rows',len(rows))
if __name__=='__main__':main()
