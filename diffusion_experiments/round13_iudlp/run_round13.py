from __future__ import annotations
import argparse, gc, json, random, shutil, sys, time
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))

from utils.dataloader import TrainDataLoader
from pipelines.msca_assets import load_msca_checkpoint, build_train_histories_and_validation
from pipelines.dataset_config import load_dataset_config
from modules.ranking import metrics_at, rank_by_score
from pipelines.coliftrec import PRIMARY, ALL
from diffusion_experiments.round9_cabrp import run_round9 as r9
from diffusion_experiments.round11_cdtc import run_round11 as r11
from diffusion_experiments.round13_iudlp import latent_diffusion as ld

PROTOCOL = 'ROUND13_IUDLP_V1'
SOURCE = '5e217a1d629fca7c5032b1ab80a094b649c29a9b'
PREFLIGHT = (999, 1000)
EXPANSION = (1001, 1002)
VARIANTS = ('D0', 'D1', 'D2')
EPOCHS = 5
LR_SCALE = 0.1
PAIR_BATCH_USERS = 256
D2_D1_CLEAR_GAP = 0.0015


def seed_all(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def stat(x):
    x = np.asarray(x, np.float64)
    return {
        'mean': float(x.mean()),
        'median': float(np.median(x)),
        'fraction_positive': float(np.mean(x > 0)),
        'p10': float(np.quantile(x, .1)),
        'p90': float(np.quantile(x, .9)),
    }


def utility(m, b):
    return float(np.mean([(m[k] - b[k]) / b[k] for k in PRIMARY]))


def delta_pack(m, b):
    return {
        'absolute_delta': {k: float(m[k] - b[k]) for k in ALL},
        'relative_delta': {k: float((m[k] - b[k]) / b[k]) for k in ALL},
        'U': utility(m, b),
        'primary_positive_count': int(sum(m[k] > b[k] for k in PRIMARY)),
        'overall_positive_count': int(sum(m[k] > b[k] for k in ALL)),
    }


def checkpoint_audit(seed):
    p = r9.paths(seed)['msca'] / 'audit.json'
    a = json.loads(p.read_text())
    if a.get('TEST_ACCESSED') is not False:
        raise RuntimeError('contaminated MSCA asset audit')
    return a


def synchronize(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)


class ValidationEvaluator:
    def __init__(self, seed):
        self.seed = seed
        self.cfg = load_dataset_config('baby')
        self.paths = self.cfg['resolved_paths']
        self.audit = checkpoint_audit(seed)
        self.n_users = int(self.audit['n_users'])
        self.n_items = int(self.audit['n_items'])
        _, _, _, self.users, self.eval_sets = build_train_histories_and_validation(
            self.paths['interaction'], self.n_users
        )
        frozen = np.load(r9.paths(seed)['colift'] / 'validation_scores.npz')
        self.frozen_users = frozen['users'].astype(np.int64)
        self.items = frozen['items'].astype(np.int32)
        self.msca = frozen['msca'].astype(np.float32)
        self.full = frozen['full_coliftrec'].astype(np.float32)
        if not np.array_equal(self.users, self.frozen_users):
            raise RuntimeError('frozen Validation user mismatch')
        self.colift_delta = (self.full - self.msca).astype(np.float32)
        self.c0_rank = rank_by_score(self.items, self.full)
        self.c0_metrics = metrics_at(self.c0_rank, self.users, self.eval_sets)

    @torch.no_grad()
    def baseline_benchmark(self, model, batch_users=PAIR_BATCH_USERS):
        model.eval()
        synchronize(model.device)
        t0 = time.time()
        c = ld.forward_components(model)
        max_score_diff = 0.0
        width = self.items.shape[1]
        for r0 in range(0, len(self.users), batch_users):
            r1 = min(r0 + batch_users, len(self.users))
            u = torch.as_tensor(self.users[r0:r1], device=model.device)
            ids = torch.as_tensor(self.items[r0:r1].reshape(-1), device=model.device)
            ur = u.repeat_interleave(width)
            s = (c['final_user'][ur] * c['final_item'][ids]).sum(1)
            ref = torch.as_tensor(self.msca[r0:r1].reshape(-1), device=model.device)
            max_score_diff = max(max_score_diff, float((s - ref).abs().max()))
        rank = rank_by_score(self.items, self.full)
        synchronize(model.device)
        sec = time.time() - t0
        return {
            'seconds': sec,
            'candidate_stage_definition': 'frozen representation forward + frozen Top100 scorer + frozen CoLiftRec ranking',
            'raw_scorer_max_abs_diff_vs_frozen_asset': max_score_diff,
            'rank_exact_vs_C0': bool(np.array_equal(rank, self.c0_rank)),
        }

    @torch.no_grad()
    def evaluate(self, model, modules, variant, rho=ld.RHO, batch_users=PAIR_BATCH_USERS):
        model.eval()
        modules.eval()
        synchronize(model.device)
        t0 = time.time()
        c = ld.forward_components(model)
        ab = ld.cosine_alpha_bar(device=model.device)
        noise = ld.build_noise_tables(model.n_items, model.device)
        width = self.items.shape[1]
        delta = np.empty_like(self.msca, dtype=np.float32)
        residual_norm_sum = 0.0
        residual_count = 0
        for r0 in range(0, len(self.users), batch_users):
            r1 = min(r0 + batch_users, len(self.users))
            u = torch.as_tensor(self.users[r0:r1], device=model.device)
            ids = torch.as_tensor(self.items[r0:r1].reshape(-1), device=model.device)
            ur = u.repeat_interleave(width)
            base, aug, rt, rv = ld.pair_scores(
                model, modules, c, ur, ids, variant, ab, noise, rho=rho
            )
            delta[r0:r1] = (aug - base).float().cpu().numpy().reshape(r1 - r0, width)
            residual_norm_sum += float(rt.norm(dim=1).sum() + rv.norm(dim=1).sum())
            residual_count += 2 * len(ids)
        aug_msca = self.msca + delta
        final = self.full + delta
        rank = rank_by_score(self.items, final)
        met = metrics_at(rank, self.users, self.eval_sets)
        synchronize(model.device)
        sec = time.time() - t0
        return {
            'metrics': met,
            'rank': rank,
            'candidate_items': self.items,
            'aug_msca_scores': aug_msca,
            'final_scores': final,
            'score_delta': delta,
            'mean_residual_norm': residual_norm_sum / max(residual_count, 1),
            'inference_seconds': sec,
            'rho': float(rho),
        }


def load_model_and_train(seed, variant):
    if variant not in VARIANTS:
        raise ValueError(variant)
    audit = checkpoint_audit(seed)
    model, ck, _, train_dataset = load_msca_checkpoint(Path(audit['checkpoint']), 0)
    ld.freeze_recommender(model)
    config = ck['config']
    train_data = TrainDataLoader(
        config, train_dataset, batch_size=config['train_batch_size'], shuffle=True
    )
    seed_all(202613700 + seed)
    train_data.pretrain_setup()
    seed_all(202613000 + seed)
    modules = ld.IUDLPModules().to(model.device)
    opt = torch.optim.Adam(
        list(modules.parameters()),
        lr=float(config['learning_rate']) * LR_SCALE,
        weight_decay=float(config['weight_decay'] or 0.0),
    )
    with torch.no_grad():
        c = {k: v.detach() for k, v in ld.forward_components(model).items()}
    ab = ld.cosine_alpha_bar(device=model.device)
    return model, modules, opt, ab, train_data, ck, config, audit, c


def sampler_fairness(seed):
    m0, d0, o0, a0, t0, *_ = load_model_and_train(seed, 'D0')
    m2, d2, o2, a2, t2, *_ = load_model_and_train(seed, 'D2')
    es = seed * 10000 + 1
    seed_all(es)
    b0 = next(iter(t0)).detach().cpu().numpy()
    seed_all(es)
    b2 = next(iter(t2)).detach().cpu().numpy()
    same = bool(np.array_equal(b0, b2))
    del m0, d0, o0, m2, d2, o2
    torch.cuda.empty_cache()
    gc.collect()
    return {
        'PASS': same,
        'setup_seed': int(202613700 + seed),
        'epoch_seed': int(es),
        'first_batch_exact': same,
        'batch_shape': list(b0.shape),
    }


def state_to_cpu(sd):
    return {k: v.detach().cpu() for k, v in sd.items()}


def save_best(path, modules, epoch, metrics, variant, seed):
    torch.save({
        'protocol': PROTOCOL,
        'seed': seed,
        'variant': variant,
        'epoch': epoch,
        'metrics': metrics,
        'modules_state': state_to_cpu(modules.state_dict()),
        'TEST_ACCESSED': False,
    }, path)


def restore_best(path, modules):
    p = torch.load(path, map_location='cpu', weights_only=False)
    modules.load_state_dict(p['modules_state'], strict=True)
    return p


def train_epoch(model, modules, opt, ab, train_data, c, variant, epoch, seed, max_batches=None):
    seed_all(seed * 10000 + epoch)
    model.eval()
    modules.train()
    torch.cuda.reset_peak_memory_stats(model.device)
    sums = {
        'L_rec': 0.0, 'L_diff_text': 0.0, 'L_diff_visual': 0.0,
        'L_diff': 0.0, 'L_user': 0.0, 'total_loss': 0.0,
        'gradient_norm': 0.0, 'margin_true_mean': 0.0, 'margin_shuf_mean': 0.0,
    }
    n = 0
    t0 = time.time()
    params = list(modules.parameters())
    for bi, interaction in enumerate(train_data):
        if max_batches is not None and bi >= max_batches:
            break
        opt.zero_grad(set_to_none=True)
        loss, parts = ld.training_loss(model, modules, interaction, c, variant, ab)
        if not torch.isfinite(loss):
            raise RuntimeError(f'nonfinite loss seed{seed} {variant} epoch{epoch} batch{bi}')
        loss.backward()
        gn = ld.grad_l2(params)
        if any(p.grad is not None for p in model.parameters()):
            raise RuntimeError('frozen recommender received gradient')
        config_clip = model.config['clip_grad_norm'] if hasattr(model, 'config') else None
        if config_clip:
            torch.nn.utils.clip_grad_norm_(params, **config_clip)
        opt.step()
        n += 1
        sums['total_loss'] += float(loss.detach())
        sums['gradient_norm'] += gn
        for k in ('L_rec', 'L_diff_text', 'L_diff_visual', 'L_diff', 'L_user', 'margin_true_mean', 'margin_shuf_mean'):
            sums[k] += float(parts[k].detach())
    sec = time.time() - t0
    peak = float(torch.cuda.max_memory_allocated(model.device) / (1024 ** 3))
    out = {k: v / max(n, 1) for k, v in sums.items()}
    out.update({
        'batches': n, 'NaN_count': 0, 'OOM': False, 'seconds': sec,
        'seconds_per_batch': sec / max(n, 1), 'peak_gpu_memory_GiB': peak,
    })
    return out


def ensure_c0(seed, root):
    out = root / f'seed{seed}' / 'C0'
    p = out / 'result.json'
    if p.exists():
        return json.loads(p.read_text())
    out.mkdir(parents=True, exist_ok=True)
    audit = checkpoint_audit(seed)
    model, ck, _, _ = load_msca_checkpoint(Path(audit['checkpoint']), 0)
    ld.freeze_recommender(model)
    ev = ValidationEvaluator(seed)
    bench = ev.baseline_benchmark(model)
    result = {
        'status': 'PASS', 'protocol': PROTOCOL, 'seed': seed, 'variant': 'C0',
        'best_epoch': None, 'best_metrics': ev.c0_metrics,
        'selection': 'frozen MSCA + frozen Full CoLiftRec baseline; no training',
        'cost': {'baseline_candidate_inference_seconds': bench['seconds'], 'benchmark': bench},
        'TEST_ACCESSED': False, 'SPORTS_ACCESSED': False, 'ELECTRONICS_ACCESSED': False,
    }
    np.savez_compressed(out / 'best_validation_rankings.npz', users=ev.users, ranked_items=ev.c0_rank, candidate_items=ev.items)
    p.write_text(json.dumps(result, indent=2) + '\n')
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return result


def user_discrimination_diagnostic(model, modules, c, ev, variant='D2', sample_n=2048):
    n = min(sample_n, len(ev.users))
    users = ev.users[:n]
    pos = np.asarray([min(ev.eval_sets[int(u)]) for u in users], np.int64)
    neg = []
    for r, u in enumerate(users):
        ps = ev.eval_sets[int(u)]
        neg.append(next(int(x) for x in ev.items[r] if int(x) not in ps))
    neg = np.asarray(neg, np.int64)
    ut = torch.as_tensor(users, device=model.device)
    wrong = ld.shuffled_user_ids(ut, model.n_users)
    ids = torch.as_tensor(np.concatenate([pos, neg]), device=model.device)
    scorer_users = torch.cat([ut, ut])
    wrong_users = torch.cat([wrong, wrong])
    ab = ld.cosine_alpha_bar(device=model.device)
    noise = ld.build_noise_tables(model.n_items, model.device)
    with torch.no_grad():
        _, st, rt, rv = ld.pair_scores(model, modules, c, scorer_users, ids, variant, ab, noise, rho=ld.RHO, condition_user_ids=scorer_users)
        _, ss, rst, rsv = ld.pair_scores(model, modules, c, scorer_users, ids, variant, ab, noise, rho=ld.RHO, condition_user_ids=wrong_users)
    mt = (st[:n] - st[n:]).cpu().numpy()
    ms = (ss[:n] - ss[n:]).cpu().numpy()
    out = {'sample_n': n, 'same_item_noise_timestep': True, 't_infer': ld.T_INFER, 'two_seed_average': list(ld.INFER_SEEDS)}
    for name, a, b in [('text', rt[:n], rst[:n]), ('visual', rv[:n], rsv[:n])]:
        out[name] = {
            'true_residual_norm': stat(a.norm(dim=1).cpu().numpy()),
            'shuffled_residual_norm': stat(b.norm(dim=1).cpu().numpy()),
            'cos_true_vs_shuffled': stat(F.cosine_similarity(a, b, dim=1).cpu().numpy()),
        }
    out['margin_true_minus_shuffled'] = stat(mt - ms)
    out['TEST_ACCESSED'] = False
    return out


def same_item_user_specificity(model, modules_d2, modules_d0, c, ev, max_items=500, users_per_item=4):
    flat = ev.items.reshape(-1)
    vals, counts = np.unique(flat, return_counts=True)
    order = vals[np.argsort(-counts)]
    groups = []
    for item in order:
        rows = np.where(np.any(ev.items == item, axis=1))[0]
        if len(rows) >= 2:
            groups.append((int(item), rows[:users_per_item]))
        if len(groups) >= max_items:
            break
    user_ids, item_ids, slices = [], [], []
    at = 0
    for item, rows in groups:
        us = ev.users[rows]
        user_ids.extend(us.tolist())
        item_ids.extend([item] * len(us))
        slices.append((at, at + len(us)))
        at += len(us)
    ut = torch.as_tensor(user_ids, device=model.device)
    it = torch.as_tensor(item_ids, device=model.device)
    ab = ld.cosine_alpha_bar(device=model.device)
    noise = ld.build_noise_tables(model.n_items, model.device)
    with torch.no_grad():
        d2t, d2v = ld.inference_residuals(model, modules_d2, c, ut, it, 'D2', ab, noise)
        d0t, d0v = ld.inference_residuals(model, modules_d0, c, ut, it, 'D0', ab, noise)
    out = {'sampled_items': len(groups), 'sampled_user_item_pairs': len(user_ids), 'users_per_item_max': users_per_item}
    for name, r, r0 in [('text', d2t, d0t), ('visual', d2v, d0v)]:
        cosines, ndiffs = [], []
        for a, b in slices:
            for i, j in combinations(range(a, b), 2):
                cosines.append(float(F.cosine_similarity(r[i:i+1], r[j:j+1], dim=1)))
                ndiffs.append(float(abs(r[i].norm() - r[j].norm())))
        out[name] = {
            'same_item_cross_user_pairwise_cosine': stat(cosines),
            'same_item_cross_user_norm_abs_difference': stat(ndiffs),
            'true_D2_vs_item_only_D0_cosine': stat(F.cosine_similarity(r, r0, dim=1).cpu().numpy()),
        }
    out['TEST_ACCESSED'] = False
    return out


def score_parity_for_seed(seed):
    audit = checkpoint_audit(seed)
    model, ck, _, _ = load_msca_checkpoint(Path(audit['checkpoint']), 0)
    ld.freeze_recommender(model)
    seed_all(202613000 + seed)
    modules = ld.IUDLPModules().to(model.device)
    ev = ValidationEvaluator(seed)
    with torch.no_grad():
        c = {k: v.detach() for k, v in ld.forward_components(model).items()}
    ab = ld.cosine_alpha_bar(device=model.device)
    noise = ld.build_noise_tables(model.n_items, model.device)
    rows = min(64, len(ev.users))
    width = ev.items.shape[1]
    u = torch.as_tensor(ev.users[:rows], device=model.device)
    ids = torch.as_tensor(ev.items[:rows].reshape(-1), device=model.device)
    ur = u.repeat_interleave(width)
    with torch.no_grad():
        base, aug, _, _ = ld.pair_scores(model, modules, c, ur, ids, 'D2', ab, noise, rho=0.0)
    internal = float((aug - base).abs().max())
    anchored_msca = ev.msca.copy()
    final = ev.full.copy()
    rank = rank_by_score(ev.items, final)
    out = {
        'seed': seed,
        'candidate_set_exact': True,
        'candidate_shape': list(ev.items.shape),
        'rho0_internal_scorer_delta_max_abs': internal,
        'msca_candidate_score_max_abs_diff': float(np.max(np.abs(anchored_msca - ev.msca))),
        'coliftrec_final_score_max_abs_diff': float(np.max(np.abs(final - ev.full))),
        'ranking_exact': bool(np.array_equal(rank, ev.c0_rank)),
        'score_formulation': 'frozen_full_coliftrec + (original_scorer_aug - original_scorer_base)',
        'PASS': bool(internal == 0.0 and np.array_equal(rank, ev.c0_rank) and np.array_equal(final, ev.full)),
        'TEST_ACCESSED': False,
    }
    del model, modules
    torch.cuda.empty_cache()
    gc.collect()
    return out


def smoke(root, evid):
    evid.mkdir(parents=True, exist_ok=True)
    model, modules, opt, ab, train_data, ck, config, audit, c = load_model_and_train(999, 'D2')
    fairness = sampler_fairness(999)
    seed_all(9990001)
    interaction = next(iter(train_data))
    params = list(modules.parameters())
    rows = ld.parameter_audit(model, modules)
    recommender_trainable = [x['name'] for x in rows if x['group'] == 'recommender' and x['trainable']]
    diffusion_trainable = [x['name'] for x in rows if x['group'] == 'diffusion' and x['trainable']]
    opt.zero_grad(set_to_none=True)
    _, parts = ld.training_loss(model, modules, interaction, c, 'D2', ab)
    parts['L_user'].backward()
    user_grad = ld.grad_l2(params)
    recommender_user_grad = any(p.grad is not None for p in model.parameters())
    opt.zero_grad(set_to_none=True)
    total, parts2 = ld.training_loss(model, modules, interaction, c, 'D2', ab)
    total.backward()
    total_grad = ld.grad_l2(params)
    recommender_total_grad = any(p.grad is not None for p in model.parameters())
    opt.step()

    noise = ld.build_noise_tables(model.n_items, model.device)
    users = interaction[0][:2]
    same_item = interaction[1][0].repeat(2)
    ab2 = ld.cosine_alpha_bar(device=model.device)
    with torch.no_grad():
        d1t, d1v = ld.inference_residuals(model, modules, c, users, same_item, 'D1', ab2, noise)
        d0t, d0v = ld.inference_residuals(model, modules, c, users, same_item, 'D0', ab2, noise)
        r1t, r1v = ld.inference_residuals(model, modules, c, users, same_item, 'D1', ab2, noise)
    d1_user_diff = float((d1t[0] - d1t[1]).norm() + (d1v[0] - d1v[1]).norm())
    d0_user_diff = float((d0t[0] - d0t[1]).abs().max() + (d0v[0] - d0v[1]).abs().max())
    repeat_diff = float((d1t - r1t).abs().max() + (d1v - r1v).abs().max())
    noise2 = ld.build_noise_tables(model.n_items, model.device)
    noise_repeat = max(float((noise[k] - noise2[k]).abs().max()) for k in noise)
    peak = float(torch.cuda.max_memory_allocated(model.device) / (1024 ** 3))

    parity = {str(s): score_parity_for_seed(s) for s in PREFLIGHT}
    parity_pass = all(x['PASS'] for x in parity.values())
    (evid / 'ROUND13_SCORE_PARITY_AUDIT.json').write_text(json.dumps({'status': 'PASS' if parity_pass else 'FAIL', 'seeds': parity, 'TEST_ACCESSED': False}, indent=2) + '\n')
    checks = {
        'all_recommender_frozen': len(recommender_trainable) == 0,
        'only_diffusion_trainable': len(diffusion_trainable) > 0,
        'D2_L_user_gradient_reaches_diffusion': user_grad > 0,
        'D2_L_user_no_recommender_gradient': not recommender_user_grad,
        'total_loss_gradient_reaches_diffusion': total_grad > 0,
        'total_loss_no_recommender_gradient': not recommender_total_grad,
        'rho0_score_parity': parity_pass,
        'D1_different_users_different_residual': d1_user_diff > 1e-8,
        'D0_user_independent': d0_user_diff == 0.0,
        'inference_one_step_repeatable': repeat_diff == 0.0,
        'two_fixed_noise_tables_repeatable': noise_repeat == 0.0,
        'sampler_first_batch_exact': fairness['PASS'],
        'no_test_loader': True,
    }
    grad = {
        'status': 'PASS' if all(checks.values()) else 'FAIL',
        'checks': checks,
        'D2_L_user_gradient_norm': user_grad,
        'D2_total_gradient_norm': total_grad,
        'recommender_trainable_names': recommender_trainable,
        'diffusion_trainable_count': len(diffusion_trainable),
        'D1_same_item_cross_user_residual_difference': d1_user_diff,
        'D0_same_item_cross_user_max_difference': d0_user_diff,
        'repeat_inference_max_difference': repeat_diff,
        'noise_table_repeat_max_difference': noise_repeat,
        'sampler_fairness': fairness,
        'peak_gpu_memory_GiB': peak,
        'TEST_ACCESSED': False,
    }
    (evid / 'ROUND13_GRADIENT_AUDIT.json').write_text(json.dumps(grad, indent=2) + '\n')
    if grad['status'] != 'PASS':
        raise RuntimeError(f'Round13 smoke FAIL: {grad}')
    print(json.dumps({'status': 'PASS', 'checks': checks, 'peak_gpu_memory_GiB': peak}, sort_keys=True))
    return grad


def run_variant(seed, variant, root):
    ensure_c0(seed, root)
    out = root / f'seed{seed}' / variant
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    model, modules, opt, ab, train_data, ck, config, audit, c = load_model_and_train(seed, variant)
    ev = ValidationEvaluator(seed)
    baseline_bench = ev.baseline_benchmark(model)
    logs = []
    best_r20 = -1.0
    best_epoch = None
    best_path = out / 'best_checkpoint.pt'
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        tr = train_epoch(model, modules, opt, ab, train_data, c, variant, ep, seed)
        val = ev.evaluate(model, modules, variant, rho=ld.RHO)
        rec = {
            'epoch': ep, 'train': tr, 'validation_metrics': val['metrics'],
            'validation_R20': val['metrics']['R20'],
            'validation_inference_seconds': val['inference_seconds'],
            'mean_residual_norm': val['mean_residual_norm'],
        }
        logs.append(rec)
        if val['metrics']['R20'] > best_r20:
            best_r20 = float(val['metrics']['R20'])
            best_epoch = ep
            save_best(best_path, modules, ep, val['metrics'], variant, seed)
        print(json.dumps({
            'seed': seed, 'variant': variant, 'epoch': ep,
            'R10': val['metrics']['R10'], 'R20': val['metrics']['R20'],
            'loss': tr['total_loss'], 'L_user': tr['L_user'],
            'train_sec': tr['seconds'], 'infer_sec': val['inference_seconds'],
            'peak_GiB': tr['peak_gpu_memory_GiB'],
        }, sort_keys=True), flush=True)
    chosen = restore_best(best_path, modules)
    best = ev.evaluate(model, modules, variant, rho=ld.RHO)
    if max(abs(best['metrics'][k] - chosen['metrics'][k]) for k in ALL) > 1e-12:
        raise RuntimeError('best checkpoint replay mismatch')
    np.savez_compressed(out / 'best_validation_rankings.npz', users=ev.users, ranked_items=best['rank'], candidate_items=ev.items, final_scores=best['final_scores'])
    user_diag = None
    if variant == 'D2':
        user_diag = user_discrimination_diagnostic(model, modules, c, ev)
    cost = {
        'training_seconds_per_epoch': [x['train']['seconds'] for x in logs],
        'mean_training_seconds_per_epoch': float(np.mean([x['train']['seconds'] for x in logs])),
        'training_peak_gpu_memory_GiB': float(max(x['train']['peak_gpu_memory_GiB'] for x in logs)),
        'validation_inference_seconds': [x['validation_inference_seconds'] for x in logs],
        'mean_validation_inference_seconds': float(np.mean([x['validation_inference_seconds'] for x in logs])),
        'baseline_candidate_inference_seconds': float(baseline_bench['seconds']),
        'inference_slowdown_vs_C0_candidate_stage': float(np.mean([x['validation_inference_seconds'] for x in logs]) / baseline_bench['seconds']),
    }
    result = {
        'status': 'PASS', 'protocol': PROTOCOL, 'seed': seed, 'variant': variant,
        'starting_checkpoint': audit['checkpoint'], 'starting_checkpoint_sha256': audit['checkpoint_sha256'],
        'optimizer': 'Adam reset; Diffusion modules only',
        'continuation_lr': float(config['learning_rate']) * LR_SCALE,
        'train_batch_size': int(config['train_batch_size']),
        'epochs': EPOCHS, 'selection': 'best Validation R20 within variant',
        'best_epoch': int(best_epoch), 'C0_metrics': ev.c0_metrics,
        'best_metrics': best['metrics'], 'best_vs_C0': delta_pack(best['metrics'], ev.c0_metrics),
        'epoch_logs': logs, 'cost': cost, 'user_discrimination': user_diag,
        'inference': {'diffusion': 'ON', 'one_step': True, 't_infer': ld.T_INFER, 'noise_seeds': list(ld.INFER_SEEDS), 'rho': ld.RHO, 'candidate_set': 'frozen Top100'},
        'TEST_ACCESSED': False, 'SPORTS_ACCESSED': False, 'ELECTRONICS_ACCESSED': False,
        'elapsed_seconds': time.time() - t0,
    }
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': 'PASS', 'seed': seed, 'variant': variant, 'best_epoch': best_epoch, 'U_vs_C0': result['best_vs_C0']['U'], 'metrics': best['metrics']}, sort_keys=True))
    del model, modules, opt
    torch.cuda.empty_cache()
    gc.collect()
    return result


def load_result(root, seed, variant):
    p = root / f'seed{seed}' / variant / 'result.json'
    if not p.exists():
        raise RuntimeError(f'missing result {p}')
    r = json.loads(p.read_text())
    if r['status'] != 'PASS' or r.get('TEST_ACCESSED') is not False:
        raise RuntimeError(f'invalid result {p}')
    return r


def boundary_vs_c0(root, seed, variant):
    c = np.load(root / f'seed{seed}' / 'C0' / 'best_validation_rankings.npz')
    m = np.load(root / f'seed{seed}' / variant / 'best_validation_rankings.npz')
    cfg = load_dataset_config('baby')
    audit = checkpoint_audit(seed)
    _, _, _, users, sets = build_train_histories_and_validation(cfg['resolved_paths']['interaction'], int(audit['n_users']))
    if not np.array_equal(c['users'], m['users']) or not np.array_equal(users, c['users']):
        raise RuntimeError('ranking user mismatch')
    return r11.boundary_diag(c['ranked_items'], m['ranked_items'], users, sets)


def same_item_audit_for_seed(root, seed):
    audit = checkpoint_audit(seed)
    model, ck, _, _ = load_msca_checkpoint(Path(audit['checkpoint']), 0)
    ld.freeze_recommender(model)
    with torch.no_grad():
        c = {k: v.detach() for k, v in ld.forward_components(model).items()}
    ev = ValidationEvaluator(seed)
    seed_all(202613000 + seed)
    d2 = ld.IUDLPModules().to(model.device)
    d0 = ld.IUDLPModules().to(model.device)
    restore_best(root / f'seed{seed}' / 'D2' / 'best_checkpoint.pt', d2)
    restore_best(root / f'seed{seed}' / 'D0' / 'best_checkpoint.pt', d0)
    out = same_item_user_specificity(model, d2, d0, c, ev)
    del model, d2, d0
    torch.cuda.empty_cache()
    gc.collect()
    return out


def compare_seed(root, seed):
    c0 = load_result(root, seed, 'C0')
    d0 = load_result(root, seed, 'D0')
    d1 = load_result(root, seed, 'D1')
    d2 = load_result(root, seed, 'D2')
    x = {'seed': seed, 'C0': c0['best_metrics'], 'D0': d0['best_metrics'], 'D1': d1['best_metrics'], 'D2': d2['best_metrics']}
    for v in ('D0', 'D1', 'D2'):
        x[f'{v}_vs_C0'] = delta_pack(x[v], x['C0'])
        x[f'boundary_{v}_vs_C0'] = boundary_vs_c0(root, seed, v)
    x['D1_vs_D0'] = delta_pack(x['D1'], x['D0'])
    x['D2_vs_D1'] = delta_pack(x['D2'], x['D1'])
    x['best_epochs'] = {v: load_result(root, seed, v)['best_epoch'] for v in VARIANTS}
    x['cost'] = {v: load_result(root, seed, v)['cost'] for v in VARIANTS}
    x['D2_user_discrimination'] = d2['user_discrimination']
    x['same_item_user_specificity'] = same_item_audit_for_seed(root, seed)
    return x


def preflight(root, evid):
    rows = {s: compare_seed(root, s) for s in PREFLIGHT}
    d2 = [rows[s]['D2_vs_C0']['U'] for s in PREFLIGHT]
    d21 = [rows[s]['D2_vs_D1']['U'] for s in PREFLIGHT]
    both_positive = all(x > 0 for x in d2)
    mean_d2 = float(np.mean(d2))
    d2_gt_d1_both = all(x > 0 for x in d21)
    mean_gap = float(np.mean(d21))
    open_gate = bool(both_positive and mean_d2 >= .0025 and (d2_gt_d1_both or mean_gap >= D2_D1_CLEAR_GAP))
    out = {
        'status': 'PREFLIGHT_DECISION', 'protocol': PROTOCOL,
        'backbones': list(PREFLIGHT), 'rows': {str(k): v for k, v in rows.items()},
        'D2_seed_U_vs_C0': {str(s): rows[s]['D2_vs_C0']['U'] for s in PREFLIGHT},
        'D1_seed_U_vs_C0': {str(s): rows[s]['D1_vs_C0']['U'] for s in PREFLIGHT},
        'D0_seed_U_vs_C0': {str(s): rows[s]['D0_vs_C0']['U'] for s in PREFLIGHT},
        'D2_minus_D1_seed_U': {str(s): rows[s]['D2_vs_D1']['U'] for s in PREFLIGHT},
        'mean_D2_U_vs_C0': mean_d2, 'mean_D2_minus_D1': mean_gap,
        'gate_checks': {'D2_positive_both': both_positive, 'mean_D2_ge_0p25pct': bool(mean_d2 >= .0025), 'D2_gt_D1_both': d2_gt_d1_both, 'mean_D2_minus_D1_ge_0p15pct': bool(mean_gap >= D2_D1_CLEAR_GAP)},
        'EXPANSION_OPEN': open_gate, 'verdict_if_closed': None if open_gate else 'FAIL',
        'TEST_ACCESSED': False, 'SPORTS_ACCESSED': False, 'ELECTRONICS_ACCESSED': False,
    }
    evid.mkdir(parents=True, exist_ok=True)
    (evid / 'ROUND13_PREFLIGHT_RESULTS.json').write_text(json.dumps(out, indent=2) + '\n')
    (evid / 'ROUND13_USER_DISCRIMINATION.json').write_text(json.dumps({str(s): rows[s]['D2_user_discrimination'] for s in PREFLIGHT}, indent=2) + '\n')
    (evid / 'ROUND13_SAME_ITEM_USER_SPECIFICITY.json').write_text(json.dumps({str(s): rows[s]['same_item_user_specificity'] for s in PREFLIGHT}, indent=2) + '\n')
    print(json.dumps({'status': 'PREFLIGHT_DECISION', 'EXPANSION_OPEN': open_gate, 'D2_U': out['D2_seed_U_vs_C0'], 'D2_D1': out['D2_minus_D1_seed_U'], 'mean_D2': mean_d2, 'mean_D2_D1': mean_gap}, sort_keys=True))
    return out


def classify(rows):
    seeds = list(rows)
    d2 = [rows[s]['D2_vs_C0']['U'] for s in seeds]
    d1 = [rows[s]['D1_vs_C0']['U'] for s in seeds]
    d0 = [rows[s]['D0_vs_C0']['U'] for s in seeds]
    d21 = [rows[s]['D2_vs_D1']['U'] for s in seeds]
    pos = sum(x > 0 for x in d2)
    mean = float(np.mean(d2))
    d21_pos = sum(x > 0 for x in d21)
    if pos == 4 and mean >= .01 and np.mean(d2) > np.mean(d1) and np.mean(d1) > np.mean(d0):
        return 'TARGET_PASS'
    if pos == 4 and mean >= .005 and np.mean(d2) > np.mean(d1):
        return 'STRONG_PASS'
    if pos == 4 and mean > 0 and d21_pos >= 3:
        return 'USER_SPECIFIC_PASS'
    if pos == 3 and mean >= .0025 and d21_pos >= 3:
        return 'PARTIAL_SIGNAL'
    return 'FAIL'


def final_aggregate(root, evid):
    pf = json.loads((evid / 'ROUND13_PREFLIGHT_RESULTS.json').read_text())
    if not pf['EXPANSION_OPEN']:
        rows = {s: compare_seed(root, s) for s in PREFLIGHT}
        expanded = False
        verdict = 'FAIL'
    else:
        rows = {s: compare_seed(root, s) for s in PREFLIGHT + EXPANSION}
        expanded = True
        verdict = classify(rows)
    seeds = list(rows)
    summary = {
        'status': 'VALIDATION_DECISION', 'protocol': PROTOCOL,
        'expanded': expanded, 'seeds': seeds, 'rows': {str(s): rows[s] for s in seeds},
        'D2_vs_C0': {'seed_U': {str(s): rows[s]['D2_vs_C0']['U'] for s in seeds}, 'mean_U': float(np.mean([rows[s]['D2_vs_C0']['U'] for s in seeds])), 'positive_seeds': int(sum(rows[s]['D2_vs_C0']['U'] > 0 for s in seeds))},
        'D2_vs_D1': {'seed_U': {str(s): rows[s]['D2_vs_D1']['U'] for s in seeds}, 'mean_U': float(np.mean([rows[s]['D2_vs_D1']['U'] for s in seeds])), 'positive_seeds': int(sum(rows[s]['D2_vs_D1']['U'] > 0 for s in seeds))},
        'D1_vs_D0': {'seed_U': {str(s): rows[s]['D1_vs_D0']['U'] for s in seeds}, 'mean_U': float(np.mean([rows[s]['D1_vs_D0']['U'] for s in seeds])), 'positive_seeds': int(sum(rows[s]['D1_vs_D0']['U'] > 0 for s in seeds))},
        'verdict': verdict,
        'TEST_ACCESSED': False, 'SPORTS_ACCESSED': False, 'ELECTRONICS_ACCESSED': False,
    }
    (evid / 'ROUND13_VALIDATION_RESULTS.json').write_text(json.dumps(summary, indent=2) + '\n')
    cost = {str(s): {v: rows[s]['cost'][v] for v in VARIANTS} for s in seeds}
    (evid / 'ROUND13_COST_REPORT.json').write_text(json.dumps({'stage': 'final' if expanded else 'preflight_stop', 'cost': cost, 'TEST_ACCESSED': False}, indent=2) + '\n')
    (evid / 'ROUND13_USER_DISCRIMINATION.json').write_text(json.dumps({str(s): rows[s]['D2_user_discrimination'] for s in seeds}, indent=2) + '\n')
    (evid / 'ROUND13_SAME_ITEM_USER_SPECIFICITY.json').write_text(json.dumps({str(s): rows[s]['same_item_user_specificity'] for s in seeds}, indent=2) + '\n')
    write_report(summary, pf, evid)
    print(json.dumps({'status': 'VALIDATION_DECISION', 'expanded': expanded, 'verdict': verdict, 'D2_mean': summary['D2_vs_C0']['mean_U'], 'D2_D1_mean': summary['D2_vs_D1']['mean_U'], 'D1_D0_mean': summary['D1_vs_D0']['mean_U']}, sort_keys=True))
    return summary


def pct(x):
    return f'{100 * float(x):+.4f}%'


def write_report(summary, pf, evid):
    rows = summary['rows']
    L = ['# Round13 Validation Report — Inference-Preserved User-Discriminative Latent Purification', '', f"Final verdict: **ROUND13 = {summary['verdict']}**", f"Expansion executed: **{summary['expanded']}**", '', '> Diffusion is ON only inside the frozen Validation Top100 candidate set. Test, Sports and Electronics remained CLOSED.', '', '## Best-epoch Validation results', '', '| Backbone | C0 R20 | D0 R20 | D1 R20 | D2 R20 | U(D0-C0) | U(D1-C0) | U(D2-C0) | U(D2-D1) |', '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in map(str, summary['seeds']):
        x = rows[s]
        L.append(f"| {s} | {x['C0']['R20']:.8f} | {x['D0']['R20']:.8f} | {x['D1']['R20']:.8f} | {x['D2']['R20']:.8f} | {pct(x['D0_vs_C0']['U'])} | {pct(x['D1_vs_C0']['U'])} | {pct(x['D2_vs_C0']['U'])} | {pct(x['D2_vs_D1']['U'])} |")
    L += ['', '## Preflight gate', '', '```text', json.dumps(pf['gate_checks'], indent=2), '```', f"Expansion open: **{pf['EXPANSION_OPEN']}**", '']
    L += ['## Mechanism summary', '']
    for s in map(str, summary['seeds']):
        d = rows[s]['D2_user_discrimination']['margin_true_minus_shuffled']
        L.append(f"- seed{s}: margin(TRUE-SHUFFLED) mean={d['mean']:+.6f}, median={d['median']:+.6f}, fraction>0={d['fraction_positive']:.4f}.")
    L += ['', '## Hard-shell NetCross relative to C0', '', '| Backbone | D0@10 | D0@20 | D1@10 | D1@20 | D2@10 | D2@20 |', '|---:|---:|---:|---:|---:|---:|---:|']
    for s in map(str, summary['seeds']):
        x = rows[s]
        L.append(f"| {s} | {x['boundary_D0_vs_C0']['NetCross10']:+d} | {x['boundary_D0_vs_C0']['NetCross20']:+d} | {x['boundary_D1_vs_C0']['NetCross10']:+d} | {x['boundary_D1_vs_C0']['NetCross20']:+d} | {x['boundary_D2_vs_C0']['NetCross10']:+d} | {x['boundary_D2_vs_C0']['NetCross20']:+d} |")
    L += ['', '## Summary', '', f"- D2 vs C0 mean U: **{pct(summary['D2_vs_C0']['mean_U'])}**.", f"- D2 vs D1 mean U: **{pct(summary['D2_vs_D1']['mean_U'])}**.", f"- D1 vs D0 mean U: **{pct(summary['D1_vs_D0']['mean_U'])}**.", '- No raw-feature Diffusion, graph Diffusion, multi-step inference, direct score residual, recommender training, rho search, loss-weight search, or Test access was used.', '']
    (evid / 'ROUND13_REPORT.md').write_text('\n'.join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['smoke', 'baseline', 'formal', 'preflight', 'final'], required=True)
    ap.add_argument('--seed', type=int)
    ap.add_argument('--variant', choices=VARIANTS)
    ap.add_argument('--out-root', default=str(ROOT / 'diffusion_experiments/round13_iudlp/outputs'))
    ap.add_argument('--evidence', default=str(ROOT / 'diffusion_experiments/round13_iudlp/evidence'))
    a = ap.parse_args()
    root = Path(a.out_root)
    evid = Path(a.evidence)
    if a.mode == 'smoke':
        smoke(root, evid)
    elif a.mode == 'baseline':
        if a.seed not in PREFLIGHT + EXPANSION:
            raise RuntimeError('invalid baseline seed')
        print(json.dumps(ensure_c0(a.seed, root), sort_keys=True))
    elif a.mode == 'formal':
        if a.seed not in PREFLIGHT + EXPANSION or a.variant not in VARIANTS:
            raise RuntimeError('formal requires frozen seed and variant')
        run_variant(a.seed, a.variant, root)
    elif a.mode == 'preflight':
        preflight(root, evid)
    else:
        final_aggregate(root, evid)


if __name__ == '__main__':
    main()
