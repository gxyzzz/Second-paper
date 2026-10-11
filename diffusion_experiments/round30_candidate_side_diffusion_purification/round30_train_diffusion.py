from __future__ import annotations
import gc
import json
import numpy as np
import torch

from diffusion_experiments.round30_candidate_side_diffusion_purification import round30_core as c
from diffusion_experiments.round30_candidate_side_diffusion_purification.round30_colift_reference import FrozenCoLiftReference
from diffusion_experiments.round30_candidate_side_diffusion_purification.round30_candidate_field import CandidateAnchorBank, CandidateFieldBank, dynamic_field
from diffusion_experiments.round30_candidate_side_diffusion_purification import round30_diffusion as d


def _mean(rows, key):
    vals = [float(x[key]) for x in rows if key in x]
    return float(np.mean(vals)) if vals else 0.0


def train_diffusion(seed, epochs=None, formal=True, return_runtime=False):
    epochs = c.FORMAL_EPOCHS if epochs is None else int(epochs)
    model, config, td, vd, pack, b0, model_hash_before = c.load_frozen_b0_model(seed)
    events = c.TrainEvents(model)
    ref = FrozenCoLiftReference(seed, model, events)
    anchors = CandidateAnchorBank(seed, model, events, ref)
    fields = CandidateFieldBank(seed, model, events, ref, anchors)
    net, opt, sched = c.make_diff(seed, model.device)
    with torch.no_grad():
        b = c.forward_bundle(model)
        s = c.latent_stats(model)
    n = len(fields.users)
    trajectory = []
    total_steps = 0
    for ep in range(epochs):
        rng = np.random.default_rng(202630000 + int(seed) * 100 + ep)
        order = rng.permutation(n)
        losses = []
        for bi, st in enumerate(range(0, n, 256)):
            idx = order[st:st + 256]
            users = torch.as_tensor(fields.users[idx], device=model.device, dtype=torch.long)
            items = torch.as_tensor(fields.candidates[idx], device=model.device, dtype=torch.long)
            if ep < c.WARMUP_EPOCHS:
                q = d.base_step(net, opt, sched, b, users, items, s,
                                203000000 + int(seed) * 100000 + ep * 1000 + bi)
                losses.append(q); total_steps += 1
            else:
                static = fields.batch_static(idx, model.device)
                field = dynamic_field(b, items, static)
                for inner in range(c.DIFF_STEPS_ACTIVE):
                    q = d.field_step(net, opt, sched, b, users, items, s, field,
                                     203100000 + int(seed) * 100000 + ep * 10000 + bi * 10 + inner)
                    losses.append(q); total_steps += 1
        rec = {
            'epoch': ep,
            'phase': 'warmup' if ep < c.WARMUP_EPOCHS else 'active',
            'diffusion_optimizer_steps': int(len(losses)),
            'L_total': _mean(losses, 'total'), 'L_base': _mean(losses, 'base'),
            'L_dir': _mean(losses, 'dir'), 'L_sep': _mean(losses, 'sep'),
            'L_proj': _mean(losses, 'proj'), 'L_bound': _mean(losses, 'bound'),
        }
        if ep >= c.WARMUP_EPOCHS:
            rec.update({
                'A_true_mean': _mean(losses, 'A_true'), 'A_shuf_mean': _mean(losses, 'A_shuf'),
                'P_DeltaA_gt0_mean': _mean(losses, 'P_DeltaA_gt0'),
                'g_field_mean': _mean(losses, 'g_field_mean'), 'g_diff_mean': _mean(losses, 'g_diff_mean'),
                'projection_abs_mean': _mean(losses, 'proj_abs_mean'),
                'sign_agreement_mean': _mean(losses, 'sign_agreement'),
                'valid_fraction_mean': _mean(losses, 'valid_fraction'),
            })
        trajectory.append(rec)
        print(json.dumps({'phase':'round30_train','seed':int(seed), **rec}), flush=True)
    model_hash_after = c.state_hash(model.state_dict())
    if model_hash_after != model_hash_before:
        raise RuntimeError('Round30 frozen backbone changed')
    final_epoch = epochs - 1
    checkpoint = None
    if formal:
        if epochs != c.FORMAL_EPOCHS or final_epoch != 39:
            raise RuntimeError('formal Round30 checkpoint must be final epoch39')
        checkpoint = c.save_diffusion(net, seed, final_epoch, model_hash_before, model_hash_after, trajectory)
    out = {
        'protocol': c.PROTOCOL, 'protocol_hash': c.protocol_hash(), 'seed': int(seed), 'variant': c.VARIANT,
        'epochs_trained': int(epochs), 'warmup_epochs': c.WARMUP_EPOCHS,
        'active_epochs': max(0, int(epochs) - c.WARMUP_EPOCHS),
        'checkpoint_selection': 'FINAL_EPOCH_ONLY_NO_VALIDATION_SELECTION',
        'final_epoch': int(final_epoch), 'checkpoint': None if checkpoint is None else str(checkpoint),
        'checkpoint_sha256': None if checkpoint is None else c.sha256_file(checkpoint),
        'model_state_hash_before': model_hash_before, 'model_state_hash_after': model_hash_after,
        'BACKBONE_PARAMETER_CHANGE': 0, 'BACKBONE_TRAINABLE_PARAMETER_COUNT': int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        'DIFFUSION_TRAINABLE_PARAMETER_COUNT': int(sum(p.numel() for p in net.parameters() if p.requires_grad)),
        'RECOMMENDER_OPTIMIZER': False, 'RANKING_TRAINING_LOSS': False,
        'L_pur': False, 'AUXILIARY_BPR': False, 'HARD_NEGATIVE_BPR': False,
        'diffusion_optimizer_steps_total': int(total_steps), 'trajectory': trajectory,
        'candidate_anchor_bank_hash': anchors.bank_hash(), 'candidate_field_bank_hash': fields.bank_hash(),
        'candidate_anchor_source': 'FROZEN_B0_FULL_COLIFT_TRAIN_TOP100',
        'baseline': c.baseline_record(seed),
        'TEST_ACCESSED': False, 'SPORTS_ACCESSED': False, 'ELECTRONICS_ACCESSED': False,
    }
    if formal:
        c.json_write(c.EVID / f'ROUND30_DIFFUSION_SEED{seed}.json', out)
    if return_runtime:
        return out, (model, events, ref, anchors, fields, net, sched, b, s)
    del model, ref, anchors, fields, net, opt
    torch.cuda.empty_cache(); gc.collect()
    return out
