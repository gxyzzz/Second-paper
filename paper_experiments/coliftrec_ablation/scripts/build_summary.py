from pathlib import Path
import csv, json
ROOT = Path(__file__).resolve().parents[3]
ev = ROOT / 'paper_experiments/coliftrec_ablation/evidence'
ALL = ('R10','N10','R20','N20','R50','N50')
rows = []
for dataset in ('BABY','SPORTS','ELEC'):
    data = json.loads((ev / f'COLIFTREC_ABLATION_{dataset}.json').read_text())
    rows.extend(data['variants'])
fields = ['dataset','variant','variant_name',*ALL,'delta_u4_vs_backbone','delta_u4_vs_full_colift','checkpoint_sha','seed']
out = ev / 'COLIFTREC_ABLATION_SUMMARY.csv'
with out.open('w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore', lineterminator=chr(10))
    w.writeheader(); w.writerows(rows)
print(f'wrote {out} rows={len(rows)}')
