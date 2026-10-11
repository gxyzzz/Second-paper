from __future__ import annotations
import gc, hashlib, json
import numpy as np
import torch

from diffusion_experiments.round28_candidate_conditioned_interest_diffusion import round28_core as c
from diffusion_experiments.round28_candidate_conditioned_interest_diffusion.round28_anchor_bank import FrozenAnchorBank
from diffusion_experiments.round28_candidate_conditioned_interest_diffusion.round28_interest_field import CoherentInterestContext
from diffusion_experiments.round28_candidate_conditioned_interest_diffusion import round28_diffusion as d

def train_variant(seed, variant, epochs=None, smoke=False):
    if variant not in c.VARIANTS:
        raise ValueError(variant)
    formal_epochs = c.FORMAL_EPOCHS if epochs is None else int(epochs)
    model, config, td, vd, pack = c.instantiate_shared(seed)
    events = c.TrainEvents(model)
    evaluator = c.CachedEvaluator(seed, model)
    wd = float(config['weight_decay']) if config['weight_decay'] is not None else 0.0
    ropt = torch.optim.Adam(model.parameters(), lr=float(config['learning_rate']), weight_decay=wd)
    fac = config['learning_rate_scheduler']
    rsched = torch.optim.lr_scheduler.LambdaLR(ropt, lr_lambda=lambda e: fac[0] ** (e / fac[1]))

    bank = ctx = net = dopt = dsched = None
    if variant == 'D5':
        bank = FrozenAnchorBank(seed, model, events, persist=True)
        ctx = CoherentInterestContext(seed, model)
        net, dopt, dsched = c.make_diff(seed, model.device)

    normal_hashes = {}; pair_hashes = {}; anchor_hashes = {}; lr_traj = []; trajectory = []
    total_steps = 0; diff_steps = 0
    best = -1.0; best_epoch = -1; best_eval = None; best_path = None

    for ep in range(formal_epochs):
        if bank is not None:
            bank.reset_epoch()
        s = c.latent_stats(model) if variant == 'D5' else None
        hh = hashlib.sha256(); ph = hashlib.sha256(); ah = hashlib.sha256()
        nl = []; al = []; dl = []; field_diag_batches = []; dir_batches = []; cal_batches = []
        resolved_total = 0
        lr_traj.append(float(ropt.param_groups[0]['lr']))
        model.train()

        for bi, interaction in enumerate(td):
            u_np = interaction[0].detach().cpu().numpy(); p_np = interaction[1].detach().cpu().numpy(); n_np = interaction[2].detach().cpu().numpy()
            hh.update(u_np.tobytes()); hh.update(p_np.tobytes()); hh.update(n_np.tobytes()); ph.update(u_np.tobytes()); ph.update(p_np.tobytes())
            ropt.zero_grad(set_to_none=True)
            b = c.forward_bundle(model)
            lmsca = c.msca_loss_from_bundle(model, interaction, b)
            la = torch.zeros((), device=model.device)

            if variant == 'D5':
                eids = bank.resolve_event_ids(interaction[0], interaction[1]); resolved_total += len(eids)
                anchors = bank.anchors_for_event_ids(eids, ep, model.device)
                ah.update(np.asarray(eids, np.int64).tobytes()); ah.update(anchors.detach().cpu().numpy().astype(np.int64).tobytes())
                if ep < c.WARMUP_EPOCHS:
                    q = d.base_warmup_step(net, dopt, dsched, b, interaction, s, 202890000 + int(seed) * 100000 + ep * 1000 + bi)
                    dl.append(q); diff_steps += 1
                else:
                    ids = [anchors]
                    packs = ctx.targets(b, interaction[0], interaction[1], ids)
                    field_diag_batches.append([{'diag': p['diag']} for p in packs])
                    for inner in range(c.DIFF_STEPS_ACTIVE):
                        q = d.field_inner_step(net, dopt, dsched, b, interaction, s, ids, packs, 202891000 + int(seed) * 100000 + ep * 10000 + bi * 10 + inner)
                        dl.append(q); diff_steps += 1
                    dirs, ddiag = d.predict_ensemble_directions(net, dsched, b, interaction, s, ids, packs); dir_batches.append(ddiag)
                    user = b['fu'][interaction[0]].detach(); pos = b['fi'][interaction[1]].detach()
                    delta, cdiag = d.calibrate(user, pos, b['fi'][anchors].detach(), dirs[0], packs[0]['confidence']); cdiag['band'] = 0; cal_batches.append([cdiag])
                    if delta.requires_grad:
                        raise RuntimeError('Round28 ranking delta must be detached')
                    la = c.aux_bpr(b['fu'][interaction[0]], b['fi'][interaction[1]], [b['fi'][anchors] + delta])

            total = lmsca + (c.LAMBDA_HN * la if variant == 'D5' and ep >= c.WARMUP_EPOCHS else 0.0)
            if not torch.isfinite(total):
                raise RuntimeError(f'nonfinite recommender loss seed{seed} {variant} ep{ep} batch{bi}')
            total.backward()
            if config['clip_grad_norm']:
                torch.nn.utils.clip_grad_norm_(model.parameters(), **config['clip_grad_norm'])
            ropt.step(); nl.append(float(lmsca.detach())); al.append(float(la.detach()) if variant == 'D5' and ep >= c.WARMUP_EPOCHS else 0.0); total_steps += 1
            if smoke and bi >= 3:
                break

        rsched.step(); normal_hashes[str(ep)] = hh.hexdigest(); pair_hashes[str(ep)] = ph.hexdigest(); anchor_hashes[str(ep)] = ah.hexdigest()
        ev = evaluator.evaluate(model); score = float(ev['colift']['R20'])
        rec = {'epoch': ep, 'normal_loss_mean': float(np.mean(nl)), 'aux_loss_mean': float(np.mean(al)), 'diff_loss_mean': None if not dl else float(np.mean([x['total'] for x in dl])), 'lr': lr_traj[-1], 'evaluation': ev, 'full_colift_R20': score, 'resolved_event_count': int(resolved_total)}
        if variant == 'D5':
            rec['anchor_slot'] = int(ep % c.N_ANCHORS)
            if ep < c.WARMUP_EPOCHS:
                rec['warmup_diffusion'] = {key: c.stats([x[key] for x in dl]) for key in ('total', 'base')}
            else:
                mech = d.summarize_mechanism(field_diag_batches, dir_batches, cal_batches, dl); rec['mechanism'] = mech
                if not mech.get('angle_bound_pass', False):
                    raise RuntimeError(f'Round28 5-degree cap broken seed{seed} ep{ep}')
                if not mech.get('target_never_exceeds_positive', False):
                    raise RuntimeError('Round28 finite target overshoot')
                if mech['near_zero_residual_fraction'] >= 0.99:
                    raise RuntimeError('Round28 TRUE residual collapsed')
        trajectory.append(rec)
        if score > best:
            best = score; best_epoch = ep; best_eval = ev; best_path = c.save_best(model, net, seed, variant, ep, score)
        msg = {'phase': 'smoke' if smoke else 'formal', 'seed': seed, 'variant': variant, 'epoch': ep, 'R20': score, 'R10': ev['colift']['R10'], 'N10': ev['colift']['N10'], 'N20': ev['colift']['N20']}
        if variant == 'D5' and ep >= c.WARMUP_EPOCHS and 'mechanism' in rec:
            msg.update({'field_stability_proxy_rho': rec['mechanism']['field']['rho']['median'], 'confidence': rec['mechanism']['field']['confidence']['median'], 'A_true': rec['mechanism']['A_true']['median'], 'DeltaA': rec['mechanism']['DeltaA']['median'], 'P_DeltaA_gt0': rec['mechanism']['DeltaA']['p_gt_zero'], 'residual': rec['mechanism']['raw_residual_norm']['median'], 'applied': rec['mechanism']['applied_refinement_fraction']})
        print(json.dumps(msg), flush=True)

    out = {
        'protocol': c.PROTOCOL, 'protocol_hash': c.protocol_hash(), 'seed': int(seed), 'variant': variant,
        'initial_state_hash': pack['state_hash'], 'optimizer': 'Adam', 'learning_rate': float(config['learning_rate']), 'weight_decay': wd,
        'scheduler': list(config['learning_rate_scheduler']), 'epochs_trained': int(formal_epochs), 'warmup_epochs': c.WARMUP_EPOCHS,
        'lambda_HN': 0.0 if variant == 'B0' else c.LAMBDA_HN, 'normal_optimizer_steps_total': int(total_steps),
        'diffusion_optimizer_steps_total': int(diff_steps), 'normal_plan_hashes': normal_hashes, 'aux_pair_hashes': pair_hashes,
        'frozen_anchor_schedule_hashes': anchor_hashes, 'lr_trajectory': lr_traj, 'checkpoint_selection': 'Full CoLiftRec Validation R20',
        'best_epoch': int(best_epoch), 'best_full_colift_R20': float(best), 'best_evaluation': best_eval, 'best_checkpoint': str(best_path),
        'checkpoint_sha256': c.sha256_file(best_path), 'trajectory': trajectory, 'diffusion_inference_at_recommendation_eval': 0,
        'TEST_ACCESSED': False, 'SPORTS_ACCESSED': False, 'ELECTRONICS_ACCESSED': False,
    }
    if variant == 'D5':
        out['frozen_anchor_bank_hash'] = bank.bank_hash(); out['anchor_provenance'] = bank.provenance_audit(); out['ranking_gradient_to_diffusion'] = 0
    if not smoke:
        c.json_write(c.EVID / f'ROUND28_{variant}_SEED{seed}.json', out)
    del model, ropt, evaluator
    if net is not None:
        del net, dopt, dsched, ctx, bank
    torch.cuda.empty_cache(); gc.collect(); return out
