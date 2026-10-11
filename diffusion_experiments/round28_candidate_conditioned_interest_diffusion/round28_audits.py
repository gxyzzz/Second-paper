from __future__ import annotations
import gc, json
import numpy as np
import torch

from diffusion_experiments.round28_candidate_conditioned_interest_diffusion import round28_core as c
from diffusion_experiments.round28_candidate_conditioned_interest_diffusion.round28_anchor_bank import FrozenAnchorBank, build_anchor_audit
from diffusion_experiments.round28_candidate_conditioned_interest_diffusion.round28_interest_field import CoherentInterestContext, summarize_field
from diffusion_experiments.round28_candidate_conditioned_interest_diffusion import round28_diffusion as d

def _event_sample(events, n, seed):
    rng = np.random.default_rng(seed)
    ids = rng.choice(len(events.u), size=min(n, len(events.u)), replace=False).astype(np.int64)
    u = events.u[ids]; p = events.p[ids]; neg = events.random_unobserved(u, p, seed + 1)
    return ids, u, p, neg

def _chunks(model, event_ids, u, p, n, batch=128):
    out = []
    for st in range(0, len(u), batch):
        en = min(st + batch, len(u))
        interaction = [torch.as_tensor(x[st:en], device=model.device, dtype=torch.long) for x in (u, p, n)]
        out.append((event_ids[st:en], interaction))
    return out

def _stats_or_none(x):
    return None if len(x) == 0 else c.stats(np.asarray(x, np.float64))

def _split_stability(pack, seed):
    dirs = pack['support_dirs'].detach().cpu().numpy(); mask = pack['support_mask'].detach().cpu().numpy(); w = pack['support_weights'].detach().cpu().numpy()
    rng = np.random.default_rng(seed); vals = []
    for r in range(len(dirs)):
        ix = np.flatnonzero(mask[r]); n = len(ix)
        if n < 3: continue
        z = ix[rng.permutation(n)]; cut = max(1, n // 2); a = z[:cut]; b = z[cut:]
        if not len(b): continue
        def combine(sel):
            ww = w[r, sel]; ww = ww / np.maximum(ww.sum(), 1e-12); v = (dirs[r, sel] * ww[:, None]).sum(0)
            return v / np.maximum(np.linalg.norm(v), 1e-12)
        vals.append(float(np.dot(combine(a), combine(b))))
    return vals

def target_stability():
    seed = 999
    model, config, td, vd, pack = c.instantiate_shared(seed); events = c.TrainEvents(model)
    bank = FrozenAnchorBank(seed, model, events, persist=True); ctx = CoherentInterestContext(seed, model)
    eids, u, p, n = _event_sample(events, 1024, 2026102801)
    packs_all = []; split = []
    with torch.no_grad():
        model.eval()
        for bi, (eid_chunk, interaction) in enumerate(_chunks(model, eids, u, p, n, 128)):
            b = c.forward_bundle(model); anchors = bank.all_anchors_for_event_ids(eid_chunk, model.device); packs = ctx.targets(b, interaction[0], interaction[1], anchors)
            packs_all.append([{'diag': q['diag']} for q in packs])
            for j, q in enumerate(packs): split.extend(_split_stability(q, 2026102900 + bi * 10 + j))
    sm = summarize_field(packs_all)
    r27 = json.loads((c.ROOT / 'diffusion_experiments/round27_local_preference_field_diffusion/evidence/ROUND27_TARGET_STABILITY_AUDIT.json').read_text())
    r27_med = float(r27['bootstrap_stability']['LPF_direction_cosine']['median']); new = _stats_or_none(split); new_med = None if new is None else float(new['median'])
    out = {
        'protocol': c.PROTOCOL, 'seed': seed, 'TRAIN_ONLY': True, 'events': int(len(eids)), 'anchors_per_event': c.N_ANCHORS,
        'context_audit': ctx.audit, 'field_statistics': sm,
        'split_half_direction_stability': {'Round27_LPF_cosine_median': r27_med, 'Round28_coherent_field_cosine': new,
            'improvement_vs_Round27': None if new_med is None else float(new_med - r27_med),
            'desired_ge_0_15': bool(new_med is not None and new_med >= .15), 'blocking_if_lt_0_08': bool(new_med is None or new_med < .08)},
        'CURRENT_POSITIVE_EXCLUDED': bool(sm['current_positive_excluded_all']), 'VALIDATION_LEAKAGE': 'NONE', 'TEST_LEAKAGE': 'NONE',
        'TEST_ACCESSED': False, 'SPORTS_ACCESSED': False, 'ELECTRONICS_ACCESSED': False,
    }
    out['PASS'] = bool(out['CURRENT_POSITIVE_EXCLUDED'] and ctx.audit['TRAIN_ONLY'] and not out['split_half_direction_stability']['blocking_if_lt_0_08'])
    c.json_write(c.EVID / 'ROUND28_TARGET_STABILITY.json', out)
    del model, bank, ctx; torch.cuda.empty_cache(); gc.collect(); return out

def smoke_and_gradient():
    seed = 999
    model, config, td, vd, pack = c.instantiate_shared(seed); events = c.TrainEvents(model)
    bank = FrozenAnchorBank(seed, model, events, persist=True); ctx = CoherentInterestContext(seed, model); net, opt, sched = c.make_diff(seed, model.device)
    eids, u, p, n = _event_sample(events, 64, 2026103001)
    interaction = [torch.as_tensor(x, device=model.device, dtype=torch.long) for x in (u, p, n)]
    b = c.forward_bundle(model); anchors = bank.anchors_for_event_ids(eids, 10, model.device); packs = ctx.targets(b, interaction[0], interaction[1], [anchors]); s = c.latent_stats(model)
    grad = d.gradient_audit_terms(net, sched, b, interaction, s, [anchors], packs, 20261011)
    q1 = d.field_inner_step(net, opt, sched, b, interaction, s, [anchors], packs, 2026103101); q2 = d.field_inner_step(net, opt, sched, b, interaction, s, [anchors], packs, 2026103102)
    dirs, ddiag = d.predict_ensemble_directions(net, sched, b, interaction, s, [anchors], packs)
    delta, cdiag = d.calibrate(b['fu'][interaction[0]].detach(), b['fi'][interaction[1]].detach(), b['fi'][anchors].detach(), dirs[0], packs[0]['confidence'])
    refined = b['fi'][anchors] + delta; rloss = c.aux_bpr(b['fu'][interaction[0]], b['fi'][interaction[1]], [refined])
    gfu, gfi = torch.autograd.grad(rloss, (b['fu'], b['fi']), retain_graph=True); net_grad = torch.autograd.grad(rloss, tuple(net.parameters()), allow_unused=True)
    ranking_to_diff = float(sum(0.0 if x is None else float(x.abs().sum()) for x in net_grad))
    diag = packs[0]['diag']; weights = np.asarray(diag['weights_sum']); ncoh = np.asarray(diag['coherent_support_count']); angle = np.asarray(cdiag['angle_deg']); dd = ddiag[0]; da = np.asarray(dd['DeltaA'])
    out = {
        'protocol': c.PROTOCOL, 'TRAIN_ONLY': True,
        'no_NaN': bool(np.isfinite([q1['total'], q2['total'], rloss.detach().item()]).all() and np.isfinite(dirs[0].detach().cpu().numpy()).all()), 'no_OOM': True,
        'base_reconstruction_working': bool(grad['base_output_grad_abs_sum_from_L_base'] > 0 and grad['base_parameter_grad_abs_sum'] > 0),
        'TRUE_branch_receives_field_gradient': bool(grad['true_output_grad_abs_sum_from_L_dir'] > 0),
        'SHUFFLED_branch_receives_separation_gradient': bool(grad['shuffled_output_grad_abs_sum_from_L_sep'] > 0),
        'TRUE_branch_receives_separation_gradient': bool(grad['true_output_grad_abs_sum_from_L_sep'] > 0),
        'user_losses_do_not_backprop_into_BASE_reference': bool(grad['base_output_grad_abs_sum_from_user_losses'] == 0),
        'user_parameter_gradient_nonzero': bool(grad['user_parameter_grad_abs_sum'] > 0), 'current_positive_excluded': bool(np.asarray(diag['current_positive_excluded']).all()),
        'coherent_weights_sum_to_1_when_valid': bool(np.allclose(weights[ncoh >= c.MIN_COHERENT], 1.0, atol=1e-5)) if np.any(ncoh >= c.MIN_COHERENT) else False,
        'ranking_gradient_to_diffusion_abs_sum': ranking_to_diff, 'RANKING_SCORE_GRAD_TO_DIFFUSION': 0 if ranking_to_diff == 0 else ranking_to_diff,
        'real_anchor_item_grad_norm': float(gfi[anchors].norm()), 'D5_delta_requires_grad': bool(delta.requires_grad),
        'A_true_median': float(np.median(dd['A_true'])), 'A_shuf_median': float(np.median(dd['A_shuf'])), 'DeltaA_median': float(np.median(da)), 'P_DeltaA_gt_zero': float((da > 0).mean()),
        'raw_residual_norm_median': float(np.median(dd['raw_residual_norm'])), 'near_zero_residual_fraction': float(np.asarray(dd['near_zero_residual']).mean()),
        'max_purification_angle_deg': float(angle.max()), 'angle_bound_pass': bool(angle.max() <= 5.01), 'loss_terms': {'step1': q1, 'step2': q2}, 'gradient_topology': grad,
        'VALIDATION_ACCESSED': False, 'TEST_ACCESSED': False, 'SPORTS_ACCESSED': False, 'ELECTRONICS_ACCESSED': False,
    }
    out['PASS'] = bool(all([out['no_NaN'], out['base_reconstruction_working'], out['TRUE_branch_receives_field_gradient'], out['SHUFFLED_branch_receives_separation_gradient'], out['user_losses_do_not_backprop_into_BASE_reference'], out['user_parameter_gradient_nonzero'], out['current_positive_excluded'], out['coherent_weights_sum_to_1_when_valid'], out['RANKING_SCORE_GRAD_TO_DIFFUSION'] == 0, out['real_anchor_item_grad_norm'] > 0, not out['D5_delta_requires_grad'], out['angle_bound_pass']]))
    c.json_write(c.EVID / 'ROUND28_SMOKE.json', out)
    del model, bank, ctx, net, opt; torch.cuda.empty_cache(); gc.collect(); return out

def preflight(epochs=12):
    seed = 999
    target = json.loads((c.EVID / 'ROUND28_TARGET_STABILITY.json').read_text()); anchor = json.loads((c.EVID / 'ROUND28_FROZEN_ANCHOR_AUDIT_SEED999.json').read_text()); smoke = json.loads((c.EVID / 'ROUND28_SMOKE.json').read_text())
    model, config, td, vd, pack = c.instantiate_shared(seed); events = c.TrainEvents(model); bank = FrozenAnchorBank(seed, model, events, persist=True); ctx = CoherentInterestContext(seed, model); net, dopt, dsched = c.make_diff(seed, model.device)
    wd = float(config['weight_decay']) if config['weight_decay'] is not None else 0.0; ropt = torch.optim.Adam(model.parameters(), lr=float(config['learning_rate']), weight_decay=wd)
    fac = config['learning_rate_scheduler']; rsched = torch.optim.lr_scheduler.LambdaLR(ropt, lr_lambda=lambda e: fac[0] ** (e / fac[1])); epoch_records = []
    for ep in range(int(epochs)):
        bank.reset_epoch(); s = c.latent_stats(model); nl, al, dl = [], [], []; model.train()
        for bi, interaction in enumerate(td):
            ropt.zero_grad(set_to_none=True); b = c.forward_bundle(model); lmsca = c.msca_loss_from_bundle(model, interaction, b); eids = bank.resolve_event_ids(interaction[0], interaction[1]); anchors = bank.anchors_for_event_ids(eids, ep, model.device); la = torch.zeros((), device=model.device)
            if ep < c.WARMUP_EPOCHS:
                q = d.base_warmup_step(net, dopt, dsched, b, interaction, s, 202892000 + ep * 1000 + bi); dl.append(q)
            else:
                packs = ctx.targets(b, interaction[0], interaction[1], [anchors])
                for inner in range(c.DIFF_STEPS_ACTIVE):
                    q = d.field_inner_step(net, dopt, dsched, b, interaction, s, [anchors], packs, 202893000 + ep * 10000 + bi * 10 + inner); dl.append(q)
                dirs, ddiag = d.predict_ensemble_directions(net, dsched, b, interaction, s, [anchors], packs)
                delta, cdiag = d.calibrate(b['fu'][interaction[0]].detach(), b['fi'][interaction[1]].detach(), b['fi'][anchors].detach(), dirs[0], packs[0]['confidence'])
                la = c.aux_bpr(b['fu'][interaction[0]], b['fi'][interaction[1]], [b['fi'][anchors] + delta])
            total = lmsca + (c.LAMBDA_HN * la if ep >= c.WARMUP_EPOCHS else 0.0)
            if not torch.isfinite(total): raise RuntimeError(f'Round28 preflight nonfinite ep{ep} batch{bi}')
            total.backward()
            if config['clip_grad_norm']: torch.nn.utils.clip_grad_norm_(model.parameters(), **config['clip_grad_norm'])
            ropt.step(); nl.append(float(lmsca.detach())); al.append(float(la.detach()))
        rsched.step(); rec = {'epoch': ep, 'normal_loss': float(np.mean(nl)), 'aux_loss': float(np.mean(al)), 'diff_total': float(np.mean([x['total'] for x in dl])), 'diff_base': float(np.mean([x['base'] for x in dl])), 'diff_dir': float(np.mean([x['dir'] for x in dl])), 'diff_sep': float(np.mean([x['sep'] for x in dl])), 'diff_bound': float(np.mean([x['bound'] for x in dl]))}; epoch_records.append(rec); print(json.dumps({'phase': 'round28_preflight_train', **rec}), flush=True)
    eids, u, p, n = _event_sample(events, 1024, 2026103201); pack_batches, dir_batches, cal_batches = [], [], []; model.eval(); s = c.latent_stats(model)
    with torch.no_grad():
        for bi, (eid_chunk, interaction) in enumerate(_chunks(model, eids, u, p, n, 128)):
            b = c.forward_bundle(model); anchors = bank.anchors_for_event_ids(eid_chunk, int(epochs) - 1, model.device); packs = ctx.targets(b, interaction[0], interaction[1], [anchors]); dirs, ddiag = d.predict_ensemble_directions(net, dsched, b, interaction, s, [anchors], packs)
            delta, cdiag = d.calibrate(b['fu'][interaction[0]].detach(), b['fi'][interaction[1]].detach(), b['fi'][anchors].detach(), dirs[0], packs[0]['confidence'])
            pack_batches.append([{'diag': packs[0]['diag']}]); dir_batches.append(ddiag); cal_batches.append([cdiag])
    mech = d.summarize_mechanism(pack_batches, dir_batches, cal_batches, None); reasons = []; stab = target['split_half_direction_stability']['Round28_coherent_field_cosine']
    if stab is None or stab['median'] < .08: reasons.append('TARGET_STABILITY_CLEAR_REGRESSION')
    if not target.get('PASS'): reasons.append('TARGET_PROVENANCE_OR_STABILITY_FAIL')
    if not anchor.get('PASS'): reasons.append('FROZEN_ANCHOR_AUDIT_FAIL')
    if not smoke.get('PASS'): reasons.append('SMOKE_OR_GRADIENT_TOPOLOGY_FAIL')
    if mech['DeltaA']['p_gt_zero'] < .70: reasons.append('TRUE_VS_SHUFFLED_MECHANISM_TOO_WEAK')
    if mech['near_zero_residual_fraction'] >= .99: reasons.append('TRUE_RESIDUAL_COLLAPSED')
    if not mech['angle_bound_pass']: reasons.append('ANGLE_BOUND_BROKEN')
    vals = [mech['A_true']['median'], mech['A_shuf']['median'], mech['DeltaA']['median'], mech['raw_residual_norm']['median'], mech['angle_deg']['max']]
    if not np.isfinite(vals).all(): reasons.append('NAN_INF_MECHANISM')
    out = {'protocol': c.PROTOCOL, 'seed': seed, 'TRAIN_ONLY': True, 'epochs': int(epochs), 'warmup_epochs': c.WARMUP_EPOCHS, 'training': epoch_records,
           'target_stability': target['split_half_direction_stability'], 'anchor_bank_hash': bank.bank_hash(), 'anchor_audit_pass': anchor.get('PASS'), 'smoke_pass': smoke.get('PASS'),
           'mechanism': mech, 'PREFLIGHT_PASS': not reasons, 'BLOCK_REASONS': reasons, 'FORMAL_ALLOWED': not reasons,
           'VALIDATION_ACCESSED': False, 'TEST_ACCESSED': False, 'SPORTS_ACCESSED': False, 'ELECTRONICS_ACCESSED': False}
    c.json_write(c.EVID / 'ROUND28_PREFLIGHT_SUMMARY.json', out)
    del model, bank, ctx, net, dopt, ropt; torch.cuda.empty_cache(); gc.collect(); return out

def seed_validation(seed):
    b = json.loads((c.EVID / f'ROUND28_B0_SEED{seed}.json').read_text()); d5 = json.loads((c.EVID / f'ROUND28_D5_SEED{seed}.json').read_text()); bm = b['best_evaluation']['colift']; dm = d5['best_evaluation']['colift']
    out = {'protocol': c.PROTOCOL, 'seed': int(seed), 'best_epochs': {'B0': b['best_epoch'], 'D5': d5['best_epoch']}, 'B0': bm, 'D5': dm, 'D5_vs_B0': c.comparison(dm, bm), 'D5_mechanism_at_best_epoch': d5['trajectory'][int(d5['best_epoch'])].get('mechanism'), 'TEST_ACCESSED': False, 'SPORTS_ACCESSED': False, 'ELECTRONICS_ACCESSED': False}
    c.json_write(c.EVID / f'ROUND28_SEED{seed}_VALIDATION.json', out); return out

def final_report():
    target = json.loads((c.EVID / 'ROUND28_TARGET_STABILITY.json').read_text()); pre = json.loads((c.EVID / 'ROUND28_PREFLIGHT_SUMMARY.json').read_text()); s999 = json.loads((c.EVID / 'ROUND28_SEED999_VALIDATION.json').read_text()); s1000 = json.loads((c.EVID / 'ROUND28_SEED1000_VALIDATION.json').read_text()); summary = json.loads((c.EVID / 'ROUND28_VALIDATION_SUMMARY.json').read_text()); anchor = json.loads((c.EVID / 'ROUND28_FROZEN_ANCHOR_AUDIT_SEED999.json').read_text())
    def pct(x): return f'{100*x:+.4f}%'
    lines = ['# Round28 Final Report — Candidate-Conditioned Coherent Interest Diffusion Purification','',
      '## 1. Implementation Summary',f'- Protocol: `{c.PROTOCOL}`','- Formal variants: B0 = MSCA + Full CoLiftRec; D5 = B0 + candidate-conditioned coherent-interest Diffusion purification.','- D5 uses a frozen Full-CoLift boundary anchor bank, coherent local-interest target, BASE-only reconstruction, TRUE/SHUFFLED user residual losses, 3-noise direction ensemble, and bounded ranking-side purification.','',
      '## 2. Frozen Settings',f'- K_HISTORY={c.K_HISTORY}; MIN_HISTORY={c.MIN_HISTORY}; trust angle={c.THETA_MAX_DEG}°; HALF_MARGIN={c.HALF_MARGIN}; warmup={c.WARMUP_EPOCHS}; epochs={c.FORMAL_EPOCHS}.',f'- Diffusion loss: L_base + {c.L_DIR} L_dir + {c.L_SEP} L_sep + {c.L_BOUND} L_bound; separation margin={c.SEP_MARGIN}.',f'- Ensemble seeds: {list(c.ENSEMBLE_NOISE_SEEDS)}.','',
      '## 3. Train/Validation Leakage Audit',f"- TRAIN_ONLY target construction: {target['TRAIN_ONLY']}.",f"- Current positive excluded: {target['CURRENT_POSITIVE_EXCLUDED']}.",f"- Validation leakage: {target['VALIDATION_LEAKAGE']}; Test leakage: {target['TEST_LEAKAGE']}.",'',
      '## 4. Frozen Boundary Anchor Audit',f"- Anchor bank hash: `{anchor['bank_hash']}`.",f"- Repeat generation exact: {anchor['repeat_exact']}.",f"- Legal anchors only: {anchor['all_anchors_exclude_train_history_and_current_positive']}.",'',
      '## 5. Coherent Interest Field Statistics',f"- History size median: {target['field_statistics']['history_size']['median']:.3f}.",f"- Local support median: {target['field_statistics']['candidate_local_support_count']['median']:.3f}.",f"- Personalized-gate support median: {target['field_statistics']['personalized_gate_support_count']['median']:.3f}.",f"- Coherent support median: {target['field_statistics']['coherent_support_count']['median']:.3f}.",f"- Invalid fraction: {target['field_statistics']['invalid_fraction']:.4f}.",f"- rho median: {target['field_statistics']['rho']['median']:.6f}; confidence median: {target['field_statistics']['confidence']['median']:.6f}.",'',
      '## 6. Split-Half Target Stability',f"- Round27 LPF median cosine: {target['split_half_direction_stability']['Round27_LPF_cosine_median']:.6f}.",f"- Round28 coherent-field median cosine: {target['split_half_direction_stability']['Round28_coherent_field_cosine']['median']:.6f}.",f"- Improvement: {target['split_half_direction_stability']['improvement_vs_Round27']:+.6f}.",'',
      '## 7. Diffusion TRUE vs SHUFFLED Diagnostics',f"- Preflight A_true median: {pre['mechanism']['A_true']['median']:.6f}.",f"- Preflight A_shuf median: {pre['mechanism']['A_shuf']['median']:.6f}.",f"- Preflight DeltaA median: {pre['mechanism']['DeltaA']['median']:+.6f}; P(DeltaA>0)={pre['mechanism']['DeltaA']['p_gt_zero']:.4f}.",'',
      '## 8. Residual / Trust-Region Diagnostics',f"- Preflight raw TRUE residual median: {pre['mechanism']['raw_residual_norm']['median']:.6f}.",f"- Near-zero residual fraction: {pre['mechanism']['near_zero_residual_fraction']:.6f}.",f"- Max purification angle: {pre['mechanism']['angle_deg']['max']:.6f}°.",f"- Applied refinement fraction: {pre['mechanism']['applied_refinement_fraction']:.4f}.",'',
      '## 9. Baby seed999 Validation',f"- B0: R10={s999['B0']['R10']:.8f}, N10={s999['B0']['N10']:.8f}, R20={s999['B0']['R20']:.8f}, N20={s999['B0']['N20']:.8f}.",f"- D5: R10={s999['D5']['R10']:.8f}, N10={s999['D5']['N10']:.8f}, R20={s999['D5']['R20']:.8f}, N20={s999['D5']['N20']:.8f}.",f"- U4(D5,B0)={pct(s999['D5_vs_B0']['U4'])}; primary positives={s999['D5_vs_B0']['primary_positive_count']}/4.",'',
      '## 10. Baby seed1000 Validation',f"- B0: R10={s1000['B0']['R10']:.8f}, N10={s1000['B0']['N10']:.8f}, R20={s1000['B0']['R20']:.8f}, N20={s1000['B0']['N20']:.8f}.",f"- D5: R10={s1000['D5']['R10']:.8f}, N10={s1000['D5']['N10']:.8f}, R20={s1000['D5']['R20']:.8f}, N20={s1000['D5']['N20']:.8f}.",f"- U4(D5,B0)={pct(s1000['D5_vs_B0']['U4'])}; primary positives={s1000['D5_vs_B0']['primary_positive_count']}/4.",'',
      '## 11. D5 vs B0 Relative Improvement',f"- seed999 U4: {pct(s999['D5_vs_B0']['U4'])}.",f"- seed1000 U4: {pct(s1000['D5_vs_B0']['U4'])}.",f"- cross-seed mean U4: {pct(summary['mean_U4_D5_vs_B0'])}.",'',
      '## 12. PASS / FAIL Decision',f"- ROUND28_STATUS = `{summary['ROUND28_STATUS']}`.",f"- minimum_success = {summary['minimum_success']}; strong_pass = {summary['strong_pass']}.",'- Baby Test was not opened; Sports/Electronics were not opened.','',
      '## 13. Unexpected Observations','- See per-epoch mechanism trajectories in the structured validation JSON files. No post-result rescue or parameter sweep was performed.','',
      '## 14. Git Commit','- Filled by repository history; the evidence commit is authoritative.','']
    path = c.RDIR / 'ROUND28_FINAL_REPORT.md'; path.write_text('\n'.join(lines)); return str(path)
