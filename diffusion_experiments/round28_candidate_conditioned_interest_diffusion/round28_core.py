from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import torch

from diffusion_experiments.round26_preference_directed_diffusion import round26_core as r26
from diffusion_experiments.round24_stable_boundary_matched import round24_core as r24
from diffusion_experiments.round22_corrected_online_gdnsm import round22_core as r22

ROOT = r26.ROOT
RDIR = ROOT / 'diffusion_experiments/round28_candidate_conditioned_interest_diffusion'
EVID = RDIR / 'evidence'
LOGS = RDIR / 'logs'
OUT = RDIR / 'outputs'
ASSETS = RDIR / 'assets'
for p in (EVID, LOGS, OUT, ASSETS):
    p.mkdir(parents=True, exist_ok=True)

PROTOCOL = 'ROUND28_CANDIDATE_CONDITIONED_COHERENT_INTEREST_DIFFUSION_V1'
SEEDS = (999, 1000)
VARIANTS = ('B0', 'D5')
WARMUP_EPOCHS = 10
FORMAL_EPOCHS = 60
LAMBDA_HN = 0.20
THETA_MAX_DEG = 5.0
HALF_MARGIN = 0.5
EPS = 1e-8
K_HISTORY = 8
MIN_HISTORY = 3
MIN_COHERENT = 2
N_ANCHORS = 4
ENSEMBLE_NOISE_SEEDS = (20261011, 20261012, 20261013)
LOCAL_TS = (1, 2, 3, 4, 5)
DIFF_STEPS_WARMUP = 1
DIFF_STEPS_ACTIVE = 2
SEP_MARGIN = 0.20
L_DIR = 0.50
L_SEP = 0.25
L_BOUND = 0.05
ALL = r24.ALL
PRIMARY = r24.PRIMARY

instantiate_shared = r26.instantiate_shared
TrainEvents = r26.TrainEvents
forward_bundle = r26.forward_bundle
msca_loss_from_bundle = r26.msca_loss_from_bundle
latent_stats = r26.latent_stats
aux_bpr = r26.aux_bpr
state_hash = r26.state_hash
stats = r26.stats
delta_pack = r26.delta_pack
CachedEvaluator = r26.CachedEvaluator
sha256_file = r26.sha256_file
make_diff = r26.make_diff
z = r26.z
inv_item = r26.inv_item
capture_rng = r22.capture_rng
restore_rng = r22.restore_rng

def json_write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=lambda x: float(x) if isinstance(x, np.generic) else str(x)) + '\n')

def protocol_hash():
    h = hashlib.sha256()
    for name in (
        'round28_core.py', 'round28_anchor_bank.py', 'round28_interest_field.py',
        'round28_diffusion.py', 'round28_train.py', 'round28_audits.py', 'run_round28.py'
    ):
        p = RDIR / name
        if p.exists():
            h.update(name.encode())
            h.update(p.read_bytes())
    h.update(json.dumps({
        'K_HISTORY': K_HISTORY, 'MIN_HISTORY': MIN_HISTORY, 'MIN_COHERENT': MIN_COHERENT,
        'anchors': N_ANCHORS, 'anchor_bands': [[11, 20], [21, 30]],
        'theta': THETA_MAX_DEG, 'half_margin': HALF_MARGIN, 'lambda_HN': LAMBDA_HN,
        'warmup': WARMUP_EPOCHS, 'epochs': FORMAL_EPOCHS, 'local_t': LOCAL_TS,
        'diff_active': DIFF_STEPS_ACTIVE, 'ensemble_noise_seeds': ENSEMBLE_NOISE_SEEDS,
        'loss': ['L_base', '0.50*L_dir', '0.25*L_sep', '0.05*L_bound'],
        'sep_margin': SEP_MARGIN,
    }, sort_keys=True).encode())
    return h.hexdigest()

def save_best(model, net, seed, variant, epoch, score):
    p = OUT / f'seed{seed}' / variant / 'best.pt'
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'protocol': PROTOCOL,
        'protocol_hash': protocol_hash(),
        'seed': int(seed),
        'variant': variant,
        'epoch': int(epoch),
        'full_colift_R20': float(score),
        'model_state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
        'diffusion_state_dict': None if net is None else {k: v.detach().cpu() for k, v in net.state_dict().items()},
        'TEST_ACCESSED': False,
        'SPORTS_ACCESSED': False,
        'ELECTRONICS_ACCESSED': False,
    }, p)
    return p

def u4(metrics, baseline):
    return float(np.mean([(metrics[k] - baseline[k]) / baseline[k] for k in PRIMARY]))

def comparison(metrics, baseline):
    return {
        'U4': u4(metrics, baseline),
        'absolute_delta': {k: float(metrics[k] - baseline[k]) for k in ALL},
        'relative_delta': {k: float((metrics[k] - baseline[k]) / baseline[k]) for k in ALL},
        'primary_positive_count': int(sum(metrics[k] > baseline[k] for k in PRIMARY)),
        'overall_positive_count': int(sum(metrics[k] > baseline[k] for k in ALL)),
    }

def validation_summary():
    rows = {
        str(seed): {
            variant: json.loads((EVID / f'ROUND28_{variant}_SEED{seed}.json').read_text())
            for variant in VARIANTS
        } for seed in SEEDS
    }
    out = {
        'protocol': PROTOCOL,
        'protocol_hash': protocol_hash(),
        'seeds': {},
        'TEST_ACCESSED': False,
        'SPORTS_ACCESSED': False,
        'ELECTRONICS_ACCESSED': False,
    }
    for seed in SEEDS:
        r = rows[str(seed)]
        b = r['B0']['best_evaluation']['colift']
        d = r['D5']['best_evaluation']['colift']
        cmp = comparison(d, b)
        out['seeds'][str(seed)] = {
            'best_epochs': {'B0': r['B0']['best_epoch'], 'D5': r['D5']['best_epoch']},
            'best_metrics': {'B0': b, 'D5': d},
            'D5_vs_B0': cmp,
        }
    us = [out['seeds'][str(s)]['D5_vs_B0']['U4'] for s in SEEDS]
    out['mean_U4_D5_vs_B0'] = float(np.mean(us))
    out['minimum_success'] = bool(all(x > 0 for x in us) and out['mean_U4_D5_vs_B0'] >= 0.01)
    no_clear_single_degradation = all(
        out['seeds'][str(s)]['D5_vs_B0']['primary_positive_count'] >= 3 for s in SEEDS
    )
    out['strong_pass'] = bool(
        all(x >= 0.005 for x in us)
        and out['mean_U4_D5_vs_B0'] >= 0.01
        and no_clear_single_degradation
    )
    out['ROUND28_STATUS'] = 'STRONG_PASS' if out['strong_pass'] else ('PASS' if out['minimum_success'] else 'FAIL')
    out['BABY_TEST_ALLOWED_AUTOMATICALLY'] = False
    json_write(EVID / 'ROUND28_VALIDATION_SUMMARY.json', out)
    return out
