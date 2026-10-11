# Round30 Final Report — Frozen-Backbone Candidate-Side Diffusion Preference Purification

## 1. Implementation Summary
- Implemented D7 as frozen recommender + frozen Full CoLiftRec + candidate-side user-conditioned Diffusion purification.
- Only Diffusion is trainable. The recommender has no optimizer and receives no ranking-training loss.
- Candidate purification is defined for the frozen B0 final-ranking Top30; no score-based acceptance gate is used.

## 2. Why Round29 Failed
- Round29 used a detached purified-positive teacher inside an extra BPR term. That teacher did not receive representation-learning gradients, so the auxiliary objective largely repeated existing user/negative optimization.
- Round30 therefore moved Diffusion to direct candidate-representation purification at ranking inference.

## 3. Frozen Backbone / CoLift Setup
- Baseline source: Round28 B0.
- `BACKBONE_PARAMETER_CHANGE = 0`; pre/post recommender state hashes are identical in smoke.
- `RECOMMENDER_OPTIMIZER = FALSE`; `RANKING_TRAINING_LOSS = FALSE`.
- `ADDITIONAL_RANKING_NETWORK = FALSE`; `COLIFT_PARAMETERS_CHANGED = FALSE`.

## 4. Canonical CoLift Provenance
- `COLIFT_BG_SOURCE = CANONICAL_FROZEN_TRAINED_BACKBONE`.
- Generic backgrounds use canonical trained-backbone TRAIN pseudo-history Top100, official T/A/V semantic z, and `fit_backgrounds(...)`.
- Validation/Test information is not used to construct generic backgrounds.

## 5. q Numerical Stability Fix
- `STD_FLOOR = 1e-4`; unreliable per-user modality/reference rows contribute zero standardized signal rather than division by an epsilon-scale std.
- TRAIN-only 1024-event audit: q mean `1.901639`, median `1.749707`, p90 `2.969026`, p99 `5.433446`, min `-0.468462`, max `21.526705`.
- `|q| > 20` fraction = `0.0001883`; `|q| > 100` fraction = `0`; all q values finite.
- Q1 PASS. The Round29 million-scale q pathology is removed.

## 6. Candidate Anchor Construction
- 19,445 TRAIN-history users; four deterministic unlabeled anchors/user; total 77,780 candidate-anchor events.
- Bands are fixed at ranks `1–10`, `11–30`, `31–60`, `61–100` in frozen B0 Full-CoLift TRAIN Top100.
- TRAIN-history collisions = `0`; deterministic hash-rank reconstruction exact = `TRUE`.
- These are candidate anchors, not negatives and never enter BPR.

## 7. Candidate-Conditioned Field Statistics
- `K_HISTORY=8`, `MIN_HISTORY=3`, `MIN_COHERENT=2` are frozen.
- Candidate-local supports use text/visual/attribute locality and canonical-context CoLift q for the fixed personalized gate.

## 8. Target Stability
- Historical medians: Round27 `0.102409`, Round28 `0.169605`, Round29 `0.169504`.
- Round30 TRAIN-only split-half median = `0.330739` (980 valid split events from 1024 sampled anchors).
- Q2 PASS (`>=0.12`).

## 9. TRUE vs SHUFFLED
- After the fixed 15-epoch smoke, ensemble audit: A_true median `0.704731`, A_shuf median `0.088628`.
- DeltaA median `0.596446`; `P(DeltaA>0)=0.986328`; residual non-zero fraction = `1.0`.
- Q3 PASS.

## 10. User-Projection Shortcut Audit
- Normalized-user projection: g_field median `0.649044`; g_diff median `0.743619`.
- Sign agreement = `0.999023`, which passes the `>=0.70` criterion.
- However median `|g_diff-g_field| = 0.107167`, exceeding the fixed `<=0.05` gate.
- The special “g_field≈0 but g_diff≈0.1+” failure pattern is not present (`USER_SHORTCUT_FAIL=FALSE`), but projection matching itself is still insufficient.
- **Q4 FAIL → blocking gate triggered.**

## 11. Diffusion Training
- Only the preregistered seed999 smoke was run: 10 BASE warmup epochs + 5 active epochs (epochs 10–14).
- No Formal 40-epoch Diffusion training was started because Q4 blocked it.
- Smoke active epoch14: P(DeltaA>0) mean `0.985253`; sign-agreement mean `0.999826`; projection absolute-error mean remained about `0.127713`.

## 12. Candidate Purification Geometry
- Validation smoke candidate geometry remains within the frozen trust region: max angle `4.958475°` and max norm-relative error `1.683e-7`.
- No `u·d_diff>0` gate and no score-increase acceptance rule is used.

## 13. Ranking Change Statistics
- 512-user Validation smoke was used only for interface sanity, not parameter selection.
- Top30 order changed fraction = `1.0000`.
- Top10 membership changed fraction = `0.5000`.
- Top20 membership changed fraction = `0.78125`.
- Mean absolute final-score delta = `0.085413`; median = `0.049159`.
- Q5 PASS: the interface is not too weak.

## 14. seed999 Validation
- Formal Validation was **not run** because preflight Q4 failed.
- No seed999 D7 formal metric is reported or inferred.

## 15. seed1000 Validation
- Formal training/evaluation was **not run** because preflight Q4 failed before Formal was allowed.
- No seed1000 D7 formal metric is reported or inferred.

## 16. D7 vs B0
- Formal D7-vs-B0 U4 is unavailable by protocol because Formal was blocked.
- B0 was not retrained.

## 17. PASS / FAIL
- `ROUND30_STATUS = PREFLIGHT_BLOCKED`.
- Q1 PASS; Q2 PASS; Q3 PASS; **Q4 FAIL**; Q5 PASS.
- `FORMAL_ALLOWED = FALSE`; `FORMAL_RUN = FALSE`.
- Baby Test remains closed; Sports and Electronics remain closed.

## 18. Unexpected Observations
- Reliable normalization fixes the Round29 q numerical pathology and substantially improves candidate-field split-half stability.
- Direct candidate purification is strong enough to materially move Top10/Top20 ordering, so the ranking interface is not negligible.
- The remaining blocking issue is narrower: Diffusion still does not quantitatively match the field’s normalized-user projection closely enough under the fixed `L_proj=0.25` protocol, despite strong TRUE-vs-SHUFFLED discrimination and sign agreement.
- No post-result change to `L_proj`, angle, Top30 scope, local t, ensemble seeds, K_HISTORY, or CoLift alpha/lambda was made.

## 19. Git Commit
- Repository history is authoritative. Round30 implementation and preflight-block evidence are committed on the dedicated Round30 branch; no Formal evidence commit is fabricated.
