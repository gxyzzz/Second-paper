# Round28 Final Report — Candidate-Conditioned Coherent Interest Diffusion Purification

## 1. Implementation Summary
- Protocol: `ROUND28_CANDIDATE_CONDITIONED_COHERENT_INTEREST_DIFFUSION_V1`
- Formal variants: B0 = MSCA + Full CoLiftRec; D5 = B0 + candidate-conditioned coherent-interest Diffusion purification.
- D5 uses a frozen Full-CoLift boundary anchor bank, coherent local-interest target, BASE-only reconstruction, TRUE/SHUFFLED user residual losses, 3-noise direction ensemble, and bounded ranking-side purification.

## 2. Frozen Settings
- K_HISTORY=8; MIN_HISTORY=3; trust angle=5.0°; HALF_MARGIN=0.5; warmup=10; epochs=60.
- Diffusion loss: L_base + 0.5 L_dir + 0.25 L_sep + 0.05 L_bound; separation margin=0.2.
- Ensemble seeds: [20261011, 20261012, 20261013].

## 3. Train/Validation Leakage Audit
- TRAIN_ONLY target construction: True.
- Current positive excluded: True.
- Validation leakage: NONE; Test leakage: NONE.

## 4. Frozen Boundary Anchor Audit
- Anchor bank hash: `2707be5477976a68fe7e45f84833555d22b4cc3810c93b20e4f393c7549eadaf`.
- Repeat generation exact: True.
- Legal anchors only: True.

## 5. Coherent Interest Field Statistics
- History size median: 6.000.
- Local support median: 6.000.
- Personalized-gate support median: 3.000.
- Coherent support median: 3.000.
- Invalid fraction: 0.1497.
- rho median: 0.791843; confidence median: 0.606008.

## 6. Split-Half Target Stability
- Round27 LPF median cosine: 0.102409.
- Round28 coherent-field median cosine: 0.169605.
- Improvement: +0.067196.

## 7. Diffusion TRUE vs SHUFFLED Diagnostics
- Preflight A_true median: 0.482418.
- Preflight A_shuf median: 0.000000.
- Preflight DeltaA median: +0.467102; P(DeltaA>0)=0.8311.

## 8. Residual / Trust-Region Diagnostics
- Preflight raw TRUE residual median: 0.012160.
- Near-zero residual fraction: 0.000000.
- Max purification angle: 5.000143°.
- Applied refinement fraction: 0.6680.

## 9. Baby seed999 Validation
- B0: R10=0.07166146, N10=0.03881882, R20=0.10677472, N20=0.04777998.
- D5: R10=0.06603558, N10=0.03750857, R20=0.09380499, N20=0.04461067.
- U4(D5,B0)=-7.5015%; primary positives=0/4.

## 10. Baby seed1000 Validation
- B0: R10=0.07127588, N10=0.03852826, R20=0.10817544, N20=0.04792441.
- D5: R10=0.06790667, N10=0.03829307, R20=0.09859555, N20=0.04608585.
- U4(D5,B0)=-4.5074%; primary positives=0/4.

## 11. D5 vs B0 Relative Improvement
- seed999 U4: -7.5015%.
- seed1000 U4: -4.5074%.
- cross-seed mean U4: -6.0044%.

## 12. PASS / FAIL Decision
- ROUND28_STATUS = `FAIL`.
- minimum_success = False; strong_pass = False.
- Baby Test was not opened; Sports/Electronics were not opened.

## 13. Unexpected Observations
- The target-side redesign worked mechanistically: split-half stability improved from 0.102409 to 0.169605, while preflight TRUE-vs-SHUFFLED remained strong (median DeltaA +0.467102; P(DeltaA>0)=0.8311).
- D5 and B0 are identical through epoch9. The ranking gap begins only when the frozen-boundary purification auxiliary objective activates at epoch10.
- Diffusion alignment remains healthy while ranking degrades: around the selected/best region A_true is about 0.60 and P(DeltaA>0) about 0.83, but D5 R20 falls below B0 on both seeds. Thus the failure is not residual collapse or inability to learn the coherent field.
- TRUE residual magnitude grows during training and applied-refinement fraction rises later, while the 5-degree bound remains valid. This is consistent with an increasingly influential auxiliary boundary signal that does not translate into recommendation utility.
- No post-result rescue, parameter sweep, Test access, or cross-domain access was performed. A separate seed1000 4060 Ti attempt was aborted and excluded; the final evidence uses the completed formal logs listed in ROUND28_CONFIG.json.

## 14. Git Commit
- Implementation/preflight HEAD before formal evidence: `0d4cbe1`.
- Final Validation evidence is committed and pushed immediately after this report is generated; repository history is authoritative for the evidence commit hash.
