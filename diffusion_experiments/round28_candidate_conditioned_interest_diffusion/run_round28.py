from __future__ import annotations
import argparse, json

from diffusion_experiments.round28_candidate_conditioned_interest_diffusion import round28_core as c
from diffusion_experiments.round28_candidate_conditioned_interest_diffusion.round28_anchor_bank import build_anchor_audit
from diffusion_experiments.round28_candidate_conditioned_interest_diffusion import round28_audits as a
from diffusion_experiments.round28_candidate_conditioned_interest_diffusion.round28_train import train_variant

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('anchor', 'target', 'smoke', 'preflight', 'summary', 'report'):
        p = sub.add_parser(name)
        if name == 'preflight':
            p.add_argument('--epochs', type=int, default=12)
    f = sub.add_parser('formal')
    f.add_argument('--seed', type=int, required=True, choices=c.SEEDS)
    f.add_argument('--variant', required=True, choices=c.VARIANTS)
    f.add_argument('--epochs', type=int, default=c.FORMAL_EPOCHS)
    sv = sub.add_parser('seed-summary')
    sv.add_argument('--seed', type=int, required=True, choices=c.SEEDS)
    z = ap.parse_args()

    if z.cmd == 'anchor':
        out = build_anchor_audit(999)
    elif z.cmd == 'target':
        out = a.target_stability()
    elif z.cmd == 'smoke':
        out = a.smoke_and_gradient()
    elif z.cmd == 'preflight':
        out = a.preflight(z.epochs)
    elif z.cmd == 'formal':
        pf = json.loads((c.EVID / 'ROUND28_PREFLIGHT_SUMMARY.json').read_text())
        if not pf.get('FORMAL_ALLOWED', False):
            raise RuntimeError(f"Round28 formal blocked: {pf.get('BLOCK_REASONS')}")
        out = train_variant(z.seed, z.variant, epochs=z.epochs, smoke=False)
    elif z.cmd == 'seed-summary':
        out = a.seed_validation(z.seed)
    elif z.cmd == 'summary':
        out = c.validation_summary()
    else:
        out = {'report': a.final_report()}
    print(json.dumps(out, indent=2, default=str))

if __name__ == '__main__':
    main()
