from __future__ import annotations
import gc, json, os
import numpy as np
import torch
from diffusion_experiments.round30_candidate_side_diffusion_purification import round30_core as c
from diffusion_experiments.round30_candidate_side_diffusion_purification.round30_colift_reference import FrozenCoLiftReference, build_reference_audit
from diffusion_experiments.round30_candidate_side_diffusion_purification.round30_candidate_field import CandidateAnchorBank, CandidateFieldBank, dynamic_field
from diffusion_experiments.round30_candidate_side_diffusion_purification import round30_diffusion as d
from diffusion_experiments.round30_candidate_side_diffusion_purification.round30_train_diffusion import train_diffusion
from diffusion_experiments.round30_candidate_side_diffusion_purification.round30_inference import evaluate_d7


def _runtime(seed=999):
    model,config,td,vd,pack,b0,h=c.load_frozen_b0_model(seed); events=c.TrainEvents(model); ref=FrozenCoLiftReference(seed,model,events); anchors=CandidateAnchorBank(seed,model,events,ref); fields=CandidateFieldBank(seed,model,events,ref,anchors); return model,events,ref,anchors,fields,h

def anchor_audit(seed=999,repeat=True):
    model,events,ref,bank,fields,h=_runtime(seed)
    h1=bank.bank_hash()
    expected_ranks=np.empty_like(bank.ranks)
    expected_anchors=np.empty_like(bank.anchors)
    for r,u in enumerate(bank.users):
        for j,(lo,hi) in enumerate(c.ANCHOR_BANDS):
            from diffusion_experiments.round30_candidate_side_diffusion_purification.round30_candidate_field import _det_rank
            rk=_det_rank(u,seed,j,lo,hi)
            expected_ranks[r,j]=rk
            expected_anchors[r,j]=bank.full_top100[r,rk-1]
    repeat_exact=bool(np.array_equal(expected_ranks,bank.ranks) and np.array_equal(expected_anchors,bank.anchors))
    violations=0
    for r,u in enumerate(bank.users):
        hist=set(events.histories[int(u)])
        violations+=sum(int(x) in hist for x in bank.anchors[r])
    bands_ok=all(np.all((bank.ranks[:,j]>=lo)&(bank.ranks[:,j]<=hi)) for j,(lo,hi) in enumerate(c.ANCHOR_BANDS))
    out={'protocol':c.PROTOCOL,'seed':seed,'users':int(len(bank.users)),'anchors_per_user':4,'event_count':int(len(bank.event_users)),'anchor_bands':[list(x) for x in c.ANCHOR_BANDS],'rank_stats':c.stats(bank.ranks.reshape(-1)),'bands_exact':bool(bands_ok),'TRAIN_HISTORY_COLLISION_COUNT':int(violations),'deterministic_repeat_exact':repeat_exact,'determinism_definition':'recompute hash(user,seed,band) ranks against immutable frozen Full-CoLift Top100 source bank','frozen_bank_hash':h1,'frozen_bank_sha256':c.sha256_file(bank.path),'candidate_source':'FROZEN_ROUND28_B0_TOP100_RERANKED_BY_FROZEN_FULL_COLIFT','anchor_semantics':'UNLABELED_CANDIDATE_ANCHOR_NOT_NEGATIVE','VALIDATION_USED_FOR_CONSTRUCTION':False,'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False,'PASS':bool(bands_ok and violations==0 and repeat_exact)}
    c.json_write(c.EVID/'ROUND30_CANDIDATE_ANCHOR_BANK_AUDIT.json',out)
    del model,ref,bank,fields; torch.cuda.empty_cache();gc.collect();return out

def q_audit(seed=999):
    model,events,ref,anchors,fields,h=_runtime(seed); rng=np.random.default_rng(20263001); idx=rng.choice(len(fields.users),size=min(1024,len(fields.users)),replace=False); mask=fields.local_mask[idx]; q=fields.q[idx][mask]; aq=np.abs(q); out={'protocol':c.PROTOCOL,'seed':seed,'TRAIN_ONLY':True,'sampled_events':int(len(idx)),'q_value_count':int(len(q)),'q_stats':dict(c.stats(q),p90=float(np.quantile(q,.9)),p99=float(np.quantile(q,.99))),'abs_q_max':float(aq.max()) if len(aq) else 0.,'abs_q_gt20_fraction':float((aq>20).mean()) if len(aq) else 0.,'abs_q_gt100_fraction':float((aq>100).mean()) if len(aq) else 0.,'finite':bool(np.isfinite(q).all()),'STD_FLOOR':c.STD_FLOOR,'semantic_reliable_fraction':fields.semantic_reliable[idx].mean(0).tolist(),'lift_reliable_fraction':fields.lift_reliable[idx].mean(0).tolist(),'semantic_std':{str(i):c.stats(fields.semantic_std[idx,i]) for i in range(3)},'lift_std':{str(i):c.stats(fields.lift_std[idx,i]) for i in range(3)},'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
    out['PASS']=bool(out['finite'] and out['abs_q_max']<100); c.json_write(c.EVID/'ROUND30_Q_STABILITY_AUDIT.json',out); del model,ref,anchors,fields;torch.cuda.empty_cache();gc.collect();return out

def _split(pack,seed):
    dirs=pack['support_dirs'].detach().cpu().numpy(); mask=pack['support_mask'].detach().cpu().numpy(); w=pack['support_weights'].detach().cpu().numpy(); rng=np.random.default_rng(seed); vals=[]
    for r in range(len(dirs)):
        ix=np.flatnonzero(mask[r]); n=len(ix)
        if n<3: continue
        z=ix[rng.permutation(n)]; cut=max(1,n//2); a=z[:cut]; bb=z[cut:]
        if not len(bb): continue
        def comb(sel):
            ww=w[r,sel]; ww=ww/max(float(ww.sum()),1e-12); v=(dirs[r,sel]*ww[:,None]).sum(0); return v/max(float(np.linalg.norm(v)),1e-12)
        vals.append(float(np.dot(comb(a),comb(bb))))
    return vals

def target_stability(seed=999):
    model,events,ref,anchors,fields,h=_runtime(seed); rng=np.random.default_rng(20263002); idx=rng.choice(len(fields.users),size=min(1024,len(fields.users)),replace=False); vals=[]; rows=[]
    with torch.no_grad():
        b=c.forward_bundle(model)
        for st in range(0,len(idx),128):
            e=idx[st:st+128]; it=torch.as_tensor(fields.candidates[e],device=model.device,dtype=torch.long); pack=dynamic_field(b,it,fields.batch_static(e,model.device)); vals.extend(_split(pack,20263020+st)); rows.append(pack['diag'])
    arr=np.asarray(vals,np.float64); med=float(np.median(arr)) if len(arr) else float('nan'); out={'protocol':c.PROTOCOL,'seed':seed,'TRAIN_ONLY':True,'sampled_events':int(len(idx)),'valid_split_events':int(len(arr)),'history_rounds':{'Round27':0.102409,'Round28':0.169605,'Round29':0.169504},'Round30_split_half_cosine':c.stats(arr),'median_ge_0_12':bool(med>=.12),'blocking_lt_0_08':bool(med<.08),'preflight_Q2_pass':bool(med>=.12),'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False,'PASS':bool(med>=.12)}; c.json_write(c.EVID/'ROUND30_TARGET_STABILITY.json',out); del model,ref,anchors,fields;torch.cuda.empty_cache();gc.collect();return out

def shortcut_audit_from_runtime(runtime,seed=999):
    model,events,ref,anchors,fields,net,sched,b,s=runtime; rng=np.random.default_rng(20263003); idx=rng.choice(len(fields.users),size=min(1024,len(fields.users)),replace=False); rows=[]
    with torch.no_grad():
        for st in range(0,len(idx),128):
            e=idx[st:st+128]; ut=torch.as_tensor(fields.users[e],device=model.device,dtype=torch.long); it=torch.as_tensor(fields.candidates[e],device=model.device,dtype=torch.long); field=dynamic_field(b,it,fields.batch_static(e,model.device)); _,diag=d.predict_ensemble(net,sched,b,ut,it,s,field); rows.append(diag)
    keys=('g_field','g_diff','A_true','A_shuf','DeltaA','raw_residual_norm','sign_agreement'); merged={k:np.concatenate([np.asarray(r[k]).reshape(-1) for r in rows]) for k in keys}; out=d.shortcut_stats(merged); gf=merged['g_field']; gd=merged['g_diff']; shortcut_fail=bool(abs(float(np.median(gf)))<.02 and float(np.median(gd))>=.10 and float((gd>0).mean())>=.95); out.update({'protocol':c.PROTOCOL,'seed':seed,'TRAIN_ONLY':True,'sampled_events':int(len(idx)),'g_field_fraction_gt_zero':float((gf>0).mean()),'g_diff_fraction_gt_zero':float((gd>0).mean()),'residual_nonzero_fraction':float((merged['raw_residual_norm']>c.EPS).mean()),'projection_match_pass':bool(out['median_abs_gdiff_minus_gfield']<=.05),'sign_agreement_pass':bool(out['sign_agreement']>=.70),'true_shuffled_pass':bool(out['DeltaA']['p_gt_zero']>=.70 and out['A_true']['median']>out['A_shuf']['median']),'USER_SHORTCUT_FAIL':shortcut_fail,'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}); out['PASS']=bool(out['projection_match_pass'] and out['sign_agreement_pass'] and out['true_shuffled_pass'] and out['residual_nonzero_fraction']>.99 and not shortcut_fail); c.json_write(c.EVID/'ROUND30_SHORTCUT_AUDIT.json',out); return out

def smoke_and_preflight():
    ref=build_reference_audit(999,force=False); aa=anchor_audit(999,repeat=True); qa=q_audit(999); ts=target_stability(999); reasons=[]
    if not ref.get('PASS',False): reasons.append('CANONICAL_COLIFT_REFERENCE_FAIL')
    if not aa['PASS']: reasons.append('CANDIDATE_ANCHOR_BANK_FAIL')
    if not qa['PASS']: reasons.append('Q_NUMERICAL_STABILITY_BLOCK')
    if not ts['PASS']: reasons.append('TARGET_STABILITY_Q2_FAIL')
    if reasons:
        out={'protocol':c.PROTOCOL,'seed':999,'PREFLIGHT_PASS':False,'FORMAL_ALLOWED':False,'BLOCK_REASONS':reasons,'reference':ref,'anchor':aa,'q':qa,'target':ts,'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}; c.json_write(c.EVID/'ROUND30_PREFLIGHT_SUMMARY.json',out); return out
    sm,runtime=train_diffusion(999,epochs=c.SMOKE_EPOCHS,formal=False,return_runtime=True); short=shortcut_audit_from_runtime(runtime,999); model,events,refobj,anchors,fields,net,sched,b,s=runtime; rank=evaluate_d7(999,net,sched,model,events,refobj,max_users=512,write_evidence=False); geom=rank['geometry']; interface=rank['ranking_change'];
    if not short['true_shuffled_pass']: reasons.append('TRUE_SHUFFLED_Q3_FAIL')
    if not short['projection_match_pass'] or not short['sign_agreement_pass'] or short['USER_SHORTCUT_FAIL']: reasons.append('USER_SHORTCUT_Q4_FAIL')
    if interface['Top20_membership_changed_fraction']<.01: reasons.append('INTERFACE_TOO_WEAK')
    if geom['angle_deg']['max']>5.01 or geom['norm_relative_error']['max']>1e-5: reasons.append('PURIFICATION_GEOMETRY_FAIL')
    out={'protocol':c.PROTOCOL,'protocol_hash':c.protocol_hash(),'seed':999,'Q1_q_pass':qa['PASS'],'Q2_target_stability_pass':ts['PASS'],'Q3_true_shuffled_pass':short['true_shuffled_pass'],'Q4_shortcut_pass':bool(short['projection_match_pass'] and short['sign_agreement_pass'] and not short['USER_SHORTCUT_FAIL']),'Q5_interface_pass':bool(interface['Top20_membership_changed_fraction']>=.01),'smoke_epochs':c.SMOKE_EPOCHS,'smoke_diffusion':sm,'shortcut':short,'ranking_smoke':rank,'PREFLIGHT_PASS':not reasons,'FORMAL_ALLOWED':not reasons,'BLOCK_REASONS':reasons,'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}; c.json_write(c.EVID/'ROUND30_PREFLIGHT_SUMMARY.json',out); del runtime; torch.cuda.empty_cache();gc.collect(); return out

def load_final_diffusion(seed,model):
    p=c.OUT/f'seed{seed}'/c.VARIANT/'epoch39.pt'; obj=torch.load(p,map_location='cpu',weights_only=False); net,opt,sched=c.make_diff(seed,model.device); net.load_state_dict(obj['diffusion_state_dict'],strict=True); net.eval(); return net,sched,obj

def formal_validation(seed):
    model,config,td,vd,pack,b0,h=c.load_frozen_b0_model(seed); events=c.TrainEvents(model); ref=FrozenCoLiftReference(seed,model,events); net,sched,obj=load_final_diffusion(seed,model); out=evaluate_d7(seed,net,sched,model,events,ref,max_users=None,write_evidence=True); out['baseline']=c.baseline_record(seed); out['diffusion_checkpoint']=str(c.OUT/f'seed{seed}'/c.VARIANT/'epoch39.pt'); out['diffusion_checkpoint_sha256']=c.sha256_file(out['diffusion_checkpoint']); out['BACKBONE_PARAMETER_CHANGE']=obj['BACKBONE_PARAMETER_CHANGE']; c.json_write(c.EVID/f'ROUND30_SEED{seed}_VALIDATION.json',out); del model,net,ref;torch.cuda.empty_cache();gc.collect(); return out

def write_config():
    out={'protocol':c.PROTOCOL,'protocol_hash':c.protocol_hash(),'dataset':'Amazon Baby','split':'Validation','seeds':list(c.SEEDS),'variant':'D7','baseline_source':'ROUND28_B0','baselines':{str(s):c.baseline_record(s) for s in c.SEEDS},'STD_FLOOR':c.STD_FLOOR,'K_HISTORY':c.K_HISTORY,'MIN_HISTORY':c.MIN_HISTORY,'MIN_COHERENT':c.MIN_COHERENT,'anchor_bands':[list(x) for x in c.ANCHOR_BANDS],'warmup_epochs':c.WARMUP_EPOCHS,'active_epochs':c.ACTIVE_EPOCHS,'formal_epochs':c.FORMAL_EPOCHS,'checkpoint_selection':'FINAL_EPOCH_39_ONLY','trust_angle_deg':c.THETA_MAX_DEG,'purification_topk':c.TOP_PURIFY,'diffusion_loss':{'L_base':1.0,'L_dir':c.L_DIR,'L_sep':c.L_SEP,'L_proj':c.L_PROJ,'L_bound':c.L_BOUND,'sep_margin':c.SEP_MARGIN},'ensemble_noise_seeds':list(c.ENSEMBLE_NOISE_SEEDS),'local_t':list(c.LOCAL_TS),'diff_steps_active':c.DIFF_STEPS_ACTIVE,'RECOMMENDER_OPTIMIZER':False,'RANKING_TRAINING_LOSS':False,'ADDITIONAL_RANKING_NETWORK':False,'CANDIDATE_SET_CHANGED':False,'COLIFT_PARAMETERS_CHANGED':False,'BACKBONE_CHANGED':False,'device_config':'cuda:0','requested_gpu_policy':'physical GPU0 = NVIDIA GeForce RTX 5090','TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}; c.json_write(c.EVID/'ROUND30_CONFIG.json',out); return out

def final_report():
    cfg=json.loads((c.EVID/'ROUND30_CONFIG.json').read_text()); ref=json.loads((c.EVID/'ROUND30_COLIFT_REFERENCE_AUDIT.json').read_text()); qa=json.loads((c.EVID/'ROUND30_Q_STABILITY_AUDIT.json').read_text()); aa=json.loads((c.EVID/'ROUND30_CANDIDATE_ANCHOR_BANK_AUDIT.json').read_text()); ts=json.loads((c.EVID/'ROUND30_TARGET_STABILITY.json').read_text()); pre=json.loads((c.EVID/'ROUND30_PREFLIGHT_SUMMARY.json').read_text()); s999=json.loads((c.EVID/'ROUND30_SEED999_VALIDATION.json').read_text()); s1000=json.loads((c.EVID/'ROUND30_SEED1000_VALIDATION.json').read_text()); summ=json.loads((c.EVID/'ROUND30_VALIDATION_SUMMARY.json').read_text()); pct=lambda x:f'{100*x:+.4f}%'
    short=pre['shortcut']; smoke=pre['ranking_smoke']; lines=['# Round30 Final Report — Frozen-Backbone Candidate-Side Diffusion Preference Purification','',
    '## 1. Implementation Summary','- D7 freezes the recommender and Full CoLiftRec, trains only Diffusion on unlabeled TRAIN candidate anchors, then directly purifies final B0 Top30 candidate representations at Validation ranking time.','',
    '## 2. Why Round29 Failed','- Round29 used a detached positive teacher inside an extra BPR term, so it mostly repeated existing user/negative optimization and did not directly change candidate representations at inference.','',
    '## 3. Frozen Backbone / CoLift Setup',f"- Baseline source=ROUND28_B0. Recommender optimizer=False; ranking training loss=False; BACKBONE_CHANGED=False; CoLift parameters unchanged.",'',
    '## 4. Canonical CoLift Provenance',f"- COLIFT_BG_SOURCE={ref['COLIFT_BG_SOURCE']}; canonical embedding `{ref['reference_embedding_asset']}`; shared-init used for background={ref['SHARED_INIT_USED_FOR_BG']}.",'',
    '## 5. q Numerical Stability Fix',f"- STD_FLOOR={c.STD_FLOOR}; q median={qa['q_stats']['median']:.6f}, p99={qa['q_stats']['p99']:.6f}, min={qa['q_stats']['min']:.6f}, max={qa['q_stats']['max']:.6f}; |q|>20={qa['abs_q_gt20_fraction']:.6f}; |q|>100={qa['abs_q_gt100_fraction']:.6f}; PASS={qa['PASS']}.",'',
    '## 6. Candidate Anchor Construction',f"- {aa['anchors_per_user']} deterministic unlabeled anchors/user from frozen B0 Full-CoLift TRAIN Top100; bands={aa['anchor_bands']}; TRAIN-history collisions={aa['TRAIN_HISTORY_COLLISION_COUNT']}; repeat exact={aa['deterministic_repeat_exact']}.",'',
    '## 7. Candidate-Conditioned Field Statistics',f"- K_HISTORY={c.K_HISTORY}, MIN_HISTORY={c.MIN_HISTORY}, MIN_COHERENT={c.MIN_COHERENT}; q uses reliable canonical-context CoLift normalization.",'',
    '## 8. Target Stability',f"- Round27=0.102409; Round28=0.169605; Round29=0.169504; Round30={ts['Round30_split_half_cosine']['median']:.6f}.",'',
    '## 9. TRUE vs SHUFFLED',f"- A_true median={short['A_true']['median']:.6f}; A_shuf median={short['A_shuf']['median']:.6f}; P(DeltaA>0)={short['DeltaA']['p_gt_zero']:.4f}.",'',
    '## 10. User-Projection Shortcut Audit',f"- g_field median={short['g_field']['median']:.6f}; g_diff median={short['g_diff']['median']:.6f}; median |g_diff-g_field|={short['median_abs_gdiff_minus_gfield']:.6f}; sign agreement={short['sign_agreement']:.4f}; USER_SHORTCUT_FAIL={short['USER_SHORTCUT_FAIL']}.",'',
    '## 11. Diffusion Training','- Diffusion only: 10 warmup + 30 active epochs, final epoch39 checkpoint, no Validation-based checkpoint selection, no recommender optimizer/ranking loss.','',
    '## 12. Candidate Purification Geometry',f"- smoke angle max={smoke['geometry']['angle_deg']['max']:.6f}°, norm-relative-error max={smoke['geometry']['norm_relative_error']['max']:.3e}; no score-based gate is used.",'',
    '## 13. Ranking Change Statistics',f"- smoke Top30 order changed={smoke['ranking_change']['Top30_order_changed_fraction']:.4f}; Top10 membership changed={smoke['ranking_change']['Top10_membership_changed_fraction']:.4f}; Top20 membership changed={smoke['ranking_change']['Top20_membership_changed_fraction']:.4f}; mean |score delta|={smoke['ranking_change']['mean_absolute_final_score_delta']:.6f}.",'',
    '## 14. seed999 Validation',f"- B0 R10={s999['B0']['R10']:.8f}, N10={s999['B0']['N10']:.8f}, R20={s999['B0']['R20']:.8f}, N20={s999['B0']['N20']:.8f}.",f"- D7 R10={s999['D7']['R10']:.8f}, N10={s999['D7']['N10']:.8f}, R20={s999['D7']['R20']:.8f}, N20={s999['D7']['N20']:.8f}; U4={pct(s999['D7_vs_B0']['U4'])}; primary positives={s999['D7_vs_B0']['primary_positive_count']}/4.",'',
    '## 15. seed1000 Validation',f"- B0 R10={s1000['B0']['R10']:.8f}, N10={s1000['B0']['N10']:.8f}, R20={s1000['B0']['R20']:.8f}, N20={s1000['B0']['N20']:.8f}.",f"- D7 R10={s1000['D7']['R10']:.8f}, N10={s1000['D7']['N10']:.8f}, R20={s1000['D7']['R20']:.8f}, N20={s1000['D7']['N20']:.8f}; U4={pct(s1000['D7_vs_B0']['U4'])}; primary positives={s1000['D7_vs_B0']['primary_positive_count']}/4.",'',
    '## 16. D7 vs B0',f"- seed999 U4={pct(s999['D7_vs_B0']['U4'])}; seed1000 U4={pct(s1000['D7_vs_B0']['U4'])}; cross-seed mean U4={pct(summ['mean_U4_D7_vs_B0'])}.",'',
    '## 17. PASS / FAIL',f"- ROUND30_STATUS=`{summ['ROUND30_STATUS']}`; minimum_success={summ['minimum_success']}; strong_pass={summ['strong_pass']}; positive_but_below_target={summ['positive_but_below_target']}. Baby Test remains closed.",'',
    '## 18. Unexpected Observations','- See per-seed TRAIN mechanism trajectories and ranking-change evidence. No post-result rescue or parameter sweep was performed.','',
    '## 19. Git Commit','- Repository history is authoritative. Implementation/preflight and final Validation evidence are committed separately on the Round30 branch.','']
    p=c.RDIR/'ROUND30_FINAL_REPORT.md'; p.write_text('\n'.join(lines)); return str(p)

def finalize_round30():
    summary=c.validation_summary(); cfg=write_config(); report=final_report(); return {'summary':summary,'config':cfg,'report':report}
