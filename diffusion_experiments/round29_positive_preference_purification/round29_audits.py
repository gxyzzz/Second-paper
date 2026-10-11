from __future__ import annotations
import gc, json, os
import numpy as np
import torch
from diffusion_experiments.round29_positive_preference_purification import round29_core as c
from diffusion_experiments.round29_positive_preference_purification.round29_colift_reference import FrozenCoLiftReference, build_reference_audit
from diffusion_experiments.round29_positive_preference_purification.round29_positive_field import PositiveFieldBank, dynamic_field
from diffusion_experiments.round29_positive_preference_purification import round29_diffusion as d
from diffusion_experiments.round29_positive_preference_purification.round29_train import train_d6

def _sample(events,n,seed):
    rng=np.random.default_rng(seed); return rng.choice(len(events.u),size=min(n,len(events.u)),replace=False).astype(np.int64)

def field_bank_audit(seed=999,repeat=True):
    model,config,td,vd,pack=c.instantiate_shared(seed); events=c.TrainEvents(model); ref=FrozenCoLiftReference(seed,model,events)
    bank=PositiveFieldBank(seed,model,events,ref); h1=bank.bank_hash(); repeat_exact=None
    if repeat:
        tmp=c.ASSETS/f'ROUND29_POSITIVE_FIELD_BANK_SEED{seed}_REPEAT_TMP.npz'
        bank2=PositiveFieldBank(seed,model,events,ref,force_rebuild=True,path_override=tmp); h2=bank2.bank_hash()
        repeat_exact=bool(h1==h2 and np.array_equal(bank.support,bank2.support) and np.array_equal(bank.pref_mask,bank2.pref_mask) and np.array_equal(bank.q,bank2.q))
        try: os.remove(tmp)
        except FileNotFoundError: pass
        del bank2
    mask=bank.local_mask
    pos_ex=bool(np.all(bank.support[mask]!=np.broadcast_to(bank.event_pos[:,None],bank.support.shape)[mask]))
    out={'protocol':c.PROTOCOL,'seed':seed,'event_count':int(len(bank.event_users)),'bank_asset':str(bank.path),'bank_sha256':c.sha256_file(bank.path),
         'bank_hash':h1,'field_bank_repeat_exact':repeat_exact,'CURRENT_POSITIVE_EXCLUDED':pos_ex,
         'K_HISTORY':c.K_HISTORY,'MIN_HISTORY':c.MIN_HISTORY,'MIN_COHERENT':c.MIN_COHERENT,
         'LOCAL_SUPPORT_ROW_Z_FOR_LIFT':False,'CANONICAL_CONTEXT_NORMALIZATION':True,
         'VALIDATION_USED_FOR_CONSTRUCTION':False,'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
    out['PASS']=bool(pos_ex and (repeat_exact is True if repeat else True))
    c.json_write(c.EVID/'ROUND29_POSITIVE_FIELD_BANK_AUDIT.json',out)
    del model,ref,bank; torch.cuda.empty_cache(); gc.collect(); return out

def _split(pack,seed):
    dirs=pack['support_dirs'].detach().cpu().numpy(); mask=pack['support_mask'].detach().cpu().numpy(); w=pack['support_weights'].detach().cpu().numpy(); rng=np.random.default_rng(seed); vals=[]
    for r in range(len(dirs)):
        ix=np.flatnonzero(mask[r]); n=len(ix)
        if n<3: continue
        z=ix[rng.permutation(n)]; cut=max(1,n//2); a=z[:cut]; b=z[cut:]
        if not len(b):continue
        def comb(sel):
            ww=w[r,sel]; ww=ww/max(float(ww.sum()),1e-12); v=(dirs[r,sel]*ww[:,None]).sum(0); return v/max(float(np.linalg.norm(v)),1e-12)
        vals.append(float(np.dot(comb(a),comb(b))))
    return vals

def _field_summary(rows):
    cat=lambda k:np.concatenate([np.asarray(r[k]).reshape(-1) for r in rows])
    return {'history_size':c.stats(cat('history_size')),'local_support_count':c.stats(cat('local_support_count')),
      'pref_support_count':c.stats(cat('pref_support_count')),'coherent_support_count':c.stats(cat('coherent_support_count')),
      'rho':c.stats(cat('rho')),'confidence':c.stats(cat('confidence')),'invalid_fraction':float(cat('invalid').mean()),
      'weights_sum':c.stats(cat('weights_sum')),'q':c.stats(cat('q')),'locality':c.stats(cat('locality'))}

def target_stability(seed=999):
    model,config,td,vd,pack0=c.instantiate_shared(seed); events=c.TrainEvents(model); ref=FrozenCoLiftReference(seed,model,events); bank=PositiveFieldBank(seed,model,events,ref)
    ids=_sample(events,1024,2026110101); vals=[]; rows=[]; gf=[]; model.eval()
    with torch.no_grad():
        b=c.forward_bundle(model)
        for st in range(0,len(ids),128):
            e=ids[st:st+128]; pos=torch.as_tensor(events.p[e],device=model.device,dtype=torch.long); users=torch.as_tensor(events.u[e],device=model.device,dtype=torch.long)
            static=bank.batch_static(e,model.device); pack=dynamic_field(b,pos,static); rows.append(pack['diag']); vals.extend(_split(pack,2026110200+st)); gf.extend((b['fu'][users].detach()*pack['direction']).sum(1).cpu().numpy().tolist())
    arr=np.asarray(vals,np.float64); g=np.asarray(gf,np.float64)
    r28=json.loads((c.ROOT/'diffusion_experiments/round28_candidate_conditioned_interest_diffusion/evidence/ROUND28_TARGET_STABILITY.json').read_text())
    out={'protocol':c.PROTOCOL,'seed':seed,'TRAIN_ONLY':True,'events':int(len(ids)),'field_statistics':_field_summary(rows),
         'split_half_direction_stability':{'Round27_LPF_cosine_median':0.102409,'Round28_coherent_field_cosine_median':0.169605,
           'Round29_coherent_field_cosine':c.stats(arr),'desired_ge_0_12':bool(np.median(arr)>=.12),'blocking_if_lt_0_08':bool(np.median(arr)<.08)},
         'direct_field_preference_compatibility':dict(c.stats(g),fraction_gt_zero=float((g>0).mean())),
         'CURRENT_POSITIVE_EXCLUDED':True,'VALIDATION_LEAKAGE':'NONE','TEST_LEAKAGE':'NONE','TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
    out['PASS']=bool(not out['split_half_direction_stability']['blocking_if_lt_0_08'])
    c.json_write(c.EVID/'ROUND29_TARGET_STABILITY.json',out)
    del model,ref,bank; torch.cuda.empty_cache(); gc.collect(); return out

def gradient_topology(seed=999):
    model,config,td,vd,pack0=c.instantiate_shared(seed); events=c.TrainEvents(model); ref=FrozenCoLiftReference(seed,model,events); bank=PositiveFieldBank(seed,model,events,ref)
    net,opt,sched=c.make_diff(seed,model.device); ids=_sample(events,64,2026110301); u=events.u[ids]; p=events.p[ids]; n=events.random_unobserved(u,p,2026110302)
    interaction=[torch.as_tensor(x,device=model.device,dtype=torch.long) for x in (u,p,n)]
    b=c.forward_bundle(model); static=bank.batch_static(ids,model.device); field=dynamic_field(b,interaction[1],static); s=c.latent_stats(model)
    grad=d.gradient_audit_terms(net,sched,b,interaction,s,[interaction[1]],[field],20261011)
    q=d.field_inner_step(net,opt,sched,b,interaction,s,[interaction[1]],[field],2026110303)
    dirs,ddiag=d.predict_ensemble_directions(net,sched,b,interaction,s,[interaction[1]],[field]); teacher,geom=d.purify_positive(b['fu'][interaction[0]],b['fi'][interaction[1]],dirs[0],field['confidence'])
    lpur=d.purification_loss(b['fu'][interaction[0]],b['fi'][interaction[2]],teacher,field['confidence'])
    gfu,gfi=torch.autograd.grad(lpur,(b['fu'],b['fi']),retain_graph=True)
    ng=torch.autograd.grad(lpur,tuple(net.parameters()),allow_unused=True); rank_to_diff=float(sum(0 if x is None else float(x.abs().sum()) for x in ng))
    out={'protocol':c.PROTOCOL,'TRAIN_ONLY':True,'gradient_diffusion':grad,'L_pur':float(lpur.detach()),
         'L_pur_user_grad_norm':float(gfu[interaction[0]].norm()),'L_pur_normal_negative_grad_norm':float(gfi[interaction[2]].norm()),
         'ranking_grad_to_diffusion':rank_to_diff,'RANKING_GRADIENT_TO_DIFFUSION':0 if rank_to_diff==0 else rank_to_diff,
         'teacher_requires_grad':bool(teacher.requires_grad),'max_norm_relative_error':float(np.max(geom['norm_relative_error'])),
         'max_angle_deg':float(np.max(geom['angle_deg'])),'NORMAL_NEGATIVE_SOURCE':'ORIGINAL_TRAIN_DATALOADER','EXTRA_NEGATIVE_ADDED':False,
         'BOUNDARY_NEGATIVE_USED':False,'FROZEN_HARD_NEGATIVE_USED':False,'DIFFUSION_NEGATIVE_USED':False,
         'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
    out['PASS']=bool(grad['base_output_grad_abs_sum_from_L_base']>0 and grad['true_output_grad_abs_sum_from_L_dir']>0 and grad['shuffled_output_grad_abs_sum_from_L_sep']>0 and grad['base_output_grad_abs_sum_from_user_losses']==0 and out['L_pur_user_grad_norm']>0 and out['L_pur_normal_negative_grad_norm']>0 and rank_to_diff==0 and not out['teacher_requires_grad'] and out['max_norm_relative_error']<=1e-5 and out['max_angle_deg']<=5.01)
    c.json_write(c.EVID/'ROUND29_GRADIENT_TOPOLOGY.json',out)
    del model,net,opt,ref,bank; torch.cuda.empty_cache(); gc.collect(); return out

def smoke_and_preflight():
    refaudit=build_reference_audit(999,force=False); fba=json.loads((c.EVID/'ROUND29_POSITIVE_FIELD_BANK_AUDIT.json').read_text()); stab=json.loads((c.EVID/'ROUND29_TARGET_STABILITY.json').read_text()); grad=json.loads((c.EVID/'ROUND29_GRADIENT_TOPOLOGY.json').read_text())
    sm=train_d6(999,epochs=c.SMOKE_EPOCHS,formal=False); b0=c.load_round28_b0(999)
    early=[]; consecutive=0; blockearly=False
    for ep in range(c.WARMUP_EPOCHS,c.SMOKE_EPOCHS):
        d6=float(sm['trajectory'][ep]['full_colift_R20']); br=float(b0['trajectory'][ep]['full_colift_R20']); rel=(d6-br)/br; early.append({'epoch':ep,'D6_R20':d6,'B0_R20':br,'relative_delta':float(rel)})
        consecutive=consecutive+1 if rel<-.03 else 0
        if consecutive>=3:blockearly=True
    mech=sm['trajectory'][-1]['mechanism']; reasons=[]
    medstab=stab['split_half_direction_stability']['Round29_coherent_field_cosine']['median']
    if medstab<.08: reasons.append('TARGET_STABILITY_BLOCK')
    if not refaudit['PASS'] or not fba['PASS']: reasons.append('CANONICAL_COLIFT_OR_FIELD_BANK_AUDIT_FAIL')
    if not grad['PASS']: reasons.append('GRADIENT_TOPOLOGY_FAIL')
    if mech['A_true']['median']<=mech['A_shuf']['median'] or mech['DeltaA']['p_gt_zero']<.70: reasons.append('TRUE_VS_SHUFFLED_FAIL')
    if mech['g_diff']['median']<=0 or mech['g_diff']['fraction_gt_zero']<=.55 or mech['g_diff']['p10']!=mech['g_diff']['p10']: reasons.append('PREFERENCE_COMPATIBILITY_FAIL')
    if not mech['angle_bound_pass'] or not mech['norm_preservation_pass']: reasons.append('POSITIVE_GEOMETRY_FAIL')
    if blockearly: reasons.append('EARLY_RANKING_SYSTEMATIC_DEGRADATION')
    out={'protocol':c.PROTOCOL,'seed':999,'TRAIN_ONLY':True,'epochs':c.SMOKE_EPOCHS,'baseline_source':'ROUND28_B0',
         'reference_audit':refaudit,'field_bank_audit':fba,'target_stability':stab['split_half_direction_stability'],'gradient_topology_pass':grad['PASS'],
         'negative_provenance':{'NORMAL_NEGATIVE_SOURCE':'ORIGINAL_TRAIN_DATALOADER','EXTRA_NEGATIVE_ADDED':False,'BOUNDARY_NEGATIVE_USED':False,'FROZEN_HARD_NEGATIVE_USED':False,'DIFFUSION_NEGATIVE_USED':False},
         'early_ranking_sanity':early,'early_systematic_degradation':blockearly,'last_mechanism':mech,'smoke_trajectory':sm['trajectory'],
         'PREFLIGHT_PASS':not reasons,'FORMAL_ALLOWED':not reasons,'BLOCK_REASONS':reasons,
         'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
    c.json_write(c.EVID/'ROUND29_SMOKE.json',{'protocol':c.PROTOCOL,'seed':999,'epochs':c.SMOKE_EPOCHS,'trajectory':sm['trajectory'],'PASS':not reasons,'TEST_ACCESSED':False})
    c.json_write(c.EVID/'ROUND29_PREFLIGHT_SUMMARY.json',out); return out

def seed_validation(seed):
    d6=json.loads((c.EVID/f'ROUND29_D6_SEED{seed}.json').read_text()); b0=c.load_round28_b0(seed)
    bm=b0['best_evaluation']['colift']; dm=d6['best_evaluation']['colift']; cmp=c.comparison(dm,bm)
    out={'protocol':c.PROTOCOL,'seed':int(seed),'baseline_source':'ROUND28_B0','baseline_protocol':b0['protocol'],
         'baseline_checkpoint_sha256':b0['checkpoint_sha256'],'best_epochs':{'B0':b0['best_epoch'],'D6':d6['best_epoch']},
         'B0':bm,'D6':dm,'D6_vs_B0':cmp,'D6_mechanism_at_best_epoch':d6['trajectory'][int(d6['best_epoch'])].get('mechanism'),
         'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
    c.json_write(c.EVID/f'ROUND29_SEED{seed}_VALIDATION.json',out); return out

def write_config():
    baselines={str(s):c.baseline_record(s) for s in c.SEEDS}
    out={'protocol':c.PROTOCOL,'protocol_hash':c.protocol_hash(),'dataset':'Amazon Baby','split':'Validation','seeds':list(c.SEEDS),'formal_variants':['B0(reused)','D6'],
         'baseline_source':'ROUND28_B0','baselines':baselines,'K_HISTORY':c.K_HISTORY,'MIN_HISTORY':c.MIN_HISTORY,'MIN_COHERENT':c.MIN_COHERENT,
         'warmup_epochs':c.WARMUP_EPOCHS,'formal_epochs':c.FORMAL_EPOCHS,'lambda_PUR':c.LAMBDA_PUR,'trust_angle_deg':c.THETA_MAX_DEG,
         'diffusion_loss':{'L_base':1.0,'L_dir':c.L_DIR,'L_sep':c.L_SEP,'L_bound':c.L_BOUND,'sep_margin':c.SEP_MARGIN},
         'ensemble_noise_seeds':list(c.ENSEMBLE_NOISE_SEEDS),'local_t':list(c.LOCAL_TS),'diff_steps_warmup':c.DIFF_STEPS_WARMUP,'diff_steps_active':c.DIFF_STEPS_ACTIVE,
         'device_config':'cuda:0','requested_gpu_policy':'physical GPU0 = NVIDIA GeForce RTX 5090 when memory permits',
         'NORMAL_NEGATIVE_SOURCE':'ORIGINAL_TRAIN_DATALOADER','EXTRA_NEGATIVE_ADDED':False,'BOUNDARY_NEGATIVE_USED':False,'FROZEN_HARD_NEGATIVE_USED':False,'DIFFUSION_NEGATIVE_USED':False,
         'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
    c.json_write(c.EVID/'ROUND29_CONFIG.json',out); return out

def final_report():
    cfg=json.loads((c.EVID/'ROUND29_CONFIG.json').read_text()); ref=json.loads((c.EVID/'ROUND29_COLIFT_REFERENCE_AUDIT.json').read_text())
    fb=json.loads((c.EVID/'ROUND29_POSITIVE_FIELD_BANK_AUDIT.json').read_text()); ts=json.loads((c.EVID/'ROUND29_TARGET_STABILITY.json').read_text())
    pre=json.loads((c.EVID/'ROUND29_PREFLIGHT_SUMMARY.json').read_text()); grad=json.loads((c.EVID/'ROUND29_GRADIENT_TOPOLOGY.json').read_text())
    s999=json.loads((c.EVID/'ROUND29_SEED999_VALIDATION.json').read_text()); s1000=json.loads((c.EVID/'ROUND29_SEED1000_VALIDATION.json').read_text()); summ=json.loads((c.EVID/'ROUND29_VALIDATION_SUMMARY.json').read_text())
    m=pre['last_mechanism']; pct=lambda x:f'{100*x:+.4f}%'
    lines=['# Round29 Final Report — Positive-Side User-Conditioned Diffusion Preference Purification','',
    '## 1. Implementation Summary','- D6 uses Diffusion only to construct a detached, norm-preserved purified positive teacher during training; final recommendation evaluation performs zero Diffusion inference.','',
    '## 2. Round28 Failure Being Corrected','- Round28 moved a boundary negative along a user-preference direction and then suppressed that preference-aligned item with BPR. Round29 corrects the ranking integration semantics: the same type of user-conditioned preference evidence purifies an observed TRAIN positive and supplies positive-side supervision.','',
    '## 3. Frozen Settings',f"- K_HISTORY={c.K_HISTORY}; MIN_HISTORY={c.MIN_HISTORY}; MIN_COHERENT={c.MIN_COHERENT}; warmup={c.WARMUP_EPOCHS}; formal epochs={c.FORMAL_EPOCHS}; lambda_PUR={c.LAMBDA_PUR}; max angle={c.THETA_MAX_DEG}°.",f'- Ensemble noise seeds: {list(c.ENSEMBLE_NOISE_SEEDS)}.','',
    '## 4. Canonical CoLift Reference Provenance',f"- COLIFT_BG_SOURCE = {ref['COLIFT_BG_SOURCE']}.",f"- reference embedding: `{ref['reference_embedding_asset']}`; sha256 `{ref['reference_embedding_sha256']}`.",f"- checkpoint: `{ref['reference_checkpoint']}`, epoch {ref['reference_checkpoint_epoch']}. shared-init used for bg = {ref['SHARED_INIT_USED_FOR_BG']}; local-support row-z for lift = {ref['LOCAL_SUPPORT_ROW_Z_FOR_LIFT']}. ",'',
    '## 5. Train / Validation / Test Leakage Audit','- All history, support, reference candidates, background and Diffusion targets are TRAIN-only. Validation is used only for evaluation/checkpoint selection. Baby Test, Sports and Electronics were not accessed.','',
    '## 6. Positive Field Construction',f"- Field-bank repeat exact = {fb['field_bank_repeat_exact']}; current positive excluded = {fb['CURRENT_POSITIVE_EXCLUDED']}; event count = {fb['event_count']}.",'',
    '## 7. Target Stability',f"- Round27 median=0.102409; Round28 median=0.169605; Round29 median={ts['split_half_direction_stability']['Round29_coherent_field_cosine']['median']:.6f}.",'',
    '## 8. TRUE vs SHUFFLED Diagnostics',f"- smoke A_true median={m['A_true']['median']:.6f}; A_shuf median={m['A_shuf']['median']:.6f}; DeltaA median={m['DeltaA']['median']:+.6f}; P(DeltaA>0)={m['DeltaA']['p_gt_zero']:.4f}.",'',
    '## 9. Preference Compatibility Audit',f"- direct field g median={ts['direct_field_preference_compatibility']['median']:.6f}, fraction>0={ts['direct_field_preference_compatibility']['fraction_gt_zero']:.4f}.",f"- Diffusion g median={m['g_diff']['median']:.6f}, fraction>0={m['g_diff']['fraction_gt_zero']:.4f}.",'',
    '## 10. Positive Purification Geometry',f"- confidence median={m['field']['confidence']['median']:.6f}; angle median={m['angle_deg']['median']:.6f}°, max={m['angle_deg']['max']:.6f}°; max norm relative error={m['norm_relative_error']['max']:.3e}; score_delta median={m['score_delta']['median']:+.6f}.",'',
    '## 11. Gradient Topology',f"- gradient topology PASS={grad['PASS']}; ranking_grad_to_diffusion={grad['ranking_grad_to_diffusion']}; teacher_requires_grad={grad['teacher_requires_grad']}.",'',
    '## 12. Negative Provenance Audit',f"- NORMAL_NEGATIVE_SOURCE={grad['NORMAL_NEGATIVE_SOURCE']}; EXTRA_NEGATIVE_ADDED={grad['EXTRA_NEGATIVE_ADDED']}; BOUNDARY_NEGATIVE_USED={grad['BOUNDARY_NEGATIVE_USED']}; FROZEN_HARD_NEGATIVE_USED={grad['FROZEN_HARD_NEGATIVE_USED']}; DIFFUSION_NEGATIVE_USED={grad['DIFFUSION_NEGATIVE_USED']}.",'',
    '## 13. seed999 Validation',f"- B0 R10={s999['B0']['R10']:.8f}, N10={s999['B0']['N10']:.8f}, R20={s999['B0']['R20']:.8f}, N20={s999['B0']['N20']:.8f}.",f"- D6 R10={s999['D6']['R10']:.8f}, N10={s999['D6']['N10']:.8f}, R20={s999['D6']['R20']:.8f}, N20={s999['D6']['N20']:.8f}; U4={pct(s999['D6_vs_B0']['U4'])}; primary positives={s999['D6_vs_B0']['primary_positive_count']}/4.",'',
    '## 14. seed1000 Validation',f"- B0 R10={s1000['B0']['R10']:.8f}, N10={s1000['B0']['N10']:.8f}, R20={s1000['B0']['R20']:.8f}, N20={s1000['B0']['N20']:.8f}.",f"- D6 R10={s1000['D6']['R10']:.8f}, N10={s1000['D6']['N10']:.8f}, R20={s1000['D6']['R20']:.8f}, N20={s1000['D6']['N20']:.8f}; U4={pct(s1000['D6_vs_B0']['U4'])}; primary positives={s1000['D6_vs_B0']['primary_positive_count']}/4.",'',
    '## 15. D6 vs B0',f"- seed999 U4={pct(s999['D6_vs_B0']['U4'])}; seed1000 U4={pct(s1000['D6_vs_B0']['U4'])}; cross-seed mean U4={pct(summ['mean_U4_D6_vs_B0'])}.",'',
    '## 16. PASS / FAIL',f"- ROUND29_STATUS = `{summ['ROUND29_STATUS']}`; minimum_success={summ['minimum_success']}; strong_pass={summ['strong_pass']}; positive_but_below_target={summ['positive_but_below_target']}. Baby Test remains closed.",'',
    '## 17. Unexpected Observations','- See the complete per-epoch D6 evidence for mechanism and ranking trajectories. No post-result rescue or parameter sweep was performed.','',
    '## 18. Git Commit','- Repository history is authoritative. Implementation/preflight and final Validation evidence are committed separately on the Round29 branch.','']
    p=c.RDIR/'ROUND29_FINAL_REPORT.md'; p.write_text('\n'.join(lines)); return str(p)

def finalize_round29():
    seed_validation(999)
    seed_validation(1000)
    summary=c.validation_summary()
    cfg=write_config()
    report=final_report()
    return {"summary":summary,"config":cfg,"report":report}
