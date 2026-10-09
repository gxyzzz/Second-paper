from __future__ import annotations
import argparse,json
from pathlib import Path
from diffusion_experiments.round21_gdnsm_curriculum_hardneg import round21_core as c


def load_ev(seed,variant):
    return json.loads((c.RDIR/'evidence'/f'ROUND21_{variant}_SEED{seed}.json').read_text())

def cmd_report():
    sp=c.RDIR/'evidence'/'ROUND21_VALIDATION_SUMMARY.json'
    if not sp.exists():raise SystemExit('validation summary missing')
    s=json.loads(sp.read_text());lines=['# Round21 Validation Report','','## Protocol','',f'Protocol: `{c.PROTOCOL}`. Baby seed999/1000 only. Test/Sports/Electronics remained closed.','','## Full-CoLift Validation']
    for seed in ('999','1000'):
        x=s['seeds'][seed];lines+=['',f'### seed{seed}','',f"- C1 vs B0 U: {x['C1_vs_B0']['U']*100:+.4f}%",f"- D1 vs B0 U: {x['D1_vs_B0']['U']*100:+.4f}%",f"- D1 vs C1 U: {x['D1_vs_C1']['U']*100:+.4f}%",f"- Best epochs (Full-CoLift R20): {x['best_epochs']}"]
    lines+=['','## Cross-seed', '',f"- mean U(D1 vs C1): {s['mean_U_D1_vs_C1']*100:+.4f}%",f"- mean U(D1 vs B0): {s['mean_U_D1_vs_B0']*100:+.4f}%",f"- D1>C1 on both seeds: {s['D1_vs_C1_positive_both']}",f"- Diffusion-specific >=+0.5% target: {s['target_diffusion_specific_0_5pct']}",f"- Total >=+1% target: {s['target_total_1pct']}",'','No Test result was accessed or used.']
    (c.RDIR/'evidence'/'ROUND21_REPORT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))

def main():
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='cmd',required=True)
    p=sub.add_parser('init');p.add_argument('--seed',type=int,required=True)
    sub.add_parser('smoke')
    p=sub.add_parser('mechanism');p.add_argument('--seed',type=int,required=True)
    p=sub.add_parser('formal');p.add_argument('--seed',type=int,required=True);p.add_argument('--variant',choices=['B0','C1','D1'],required=True)
    p=sub.add_parser('fairness');p.add_argument('--seed',type=int,required=True)
    sub.add_parser('summary');sub.add_parser('report')
    a=ap.parse_args()
    if a.cmd=='init':out=c.create_shared_init(a.seed)
    elif a.cmd=='smoke':out=c.run_smoke()
    elif a.cmd=='mechanism':out=c.run_mechanism(a.seed)
    elif a.cmd=='formal':out=c.train_variant(a.seed,a.variant)
    elif a.cmd=='fairness':
        r={v:load_ev(a.seed,v) for v in ('B0','C1','D1')};out=c.fairness(a.seed,r)
    elif a.cmd=='summary':
        r={seed:{v:load_ev(seed,v) for v in ('B0','C1','D1')} for seed in c.SEEDS};out=c.summarize_formal(r)
    elif a.cmd=='report':cmd_report();return
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
