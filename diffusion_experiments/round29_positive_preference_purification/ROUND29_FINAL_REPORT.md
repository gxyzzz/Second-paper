# Round29 Final Report — Positive-Side User-Conditioned Diffusion Preference Purification

## 1. Implementation Summary
- D6 uses Diffusion only to construct a detached, norm-preserved purified positive teacher during training; final recommendation evaluation performs zero Diffusion inference.

## 2. Round28 Failure Being Corrected
- Round28 moved a boundary negative along a user-preference direction and then suppressed that preference-aligned item with BPR. Round29 corrects the ranking integration semantics: the same type of user-conditioned preference evidence purifies an observed TRAIN positive and supplies positive-side supervision.

## 3. Frozen Settings
- K_HISTORY=8; MIN_HISTORY=3; MIN_COHERENT=2; warmup=10; formal epochs=60; lambda_PUR=0.1; max angle=5.0°.
- Ensemble noise seeds: [20261011, 20261012, 20261013].

## 4. Canonical CoLift Reference Provenance
- COLIFT_BG_SOURCE = CANONICAL_FROZEN_TRAINED_BACKBONE.
- reference embedding: `/home/gxy/code/Second-paper/runs/assets/msca_baby_seed999/embeddings.npz`; sha256 `48522f88766ee88f3f4867896f5b26faac7e560ad5d09e7ce8608262355d41cc`.
- seed999 checkpoint: `/home/gxy/code/Second-paper/runs/checkpoints/MSCA-Sep-26-2026-12-48-04.pth`, epoch 37. shared-init used for bg = False; local-support row-z for lift = False.
- seed1000 reference embedding: `/home/gxy/code/Second-paper/runs/backbone_robustness/baby_seed1000/msca/assets/embeddings.npz`; sha256 `408d0ba948a9befed677a09bab90e224b55e654fa54603ff9354575413e3535d`; checkpoint epoch 45. Its per-seed provenance audit also PASSed.

## 5. Train / Validation / Test Leakage Audit
- All history, support, reference candidates, background and Diffusion targets are TRAIN-only. Validation is used only for evaluation/checkpoint selection. Baby Test, Sports and Electronics were not accessed.

## 6. Positive Field Construction
- Field-bank repeat exact = True; current positive excluded = True; event count = 118551.

## 7. Target Stability
- Round27 median=0.102409; Round28 median=0.169605; Round29 median=0.169504.

## 8. TRUE vs SHUFFLED Diagnostics
- smoke A_true median=0.477159; A_shuf median=0.000000; DeltaA median=+0.463014; P(DeltaA>0)=0.8173.

## 9. Preference Compatibility Audit
- direct field g median=0.005532, fraction>0=0.8135.
- Diffusion g median=0.111252, fraction>0=0.9985.

## 10. Positive Purification Geometry
- confidence median=0.626785; angle median=3.134003°, max=4.980748°; max norm relative error=2.523e-07; score_delta median=+0.001585.

## 11. Gradient Topology
- gradient topology PASS=True; ranking_grad_to_diffusion=0.0; teacher_requires_grad=False.

## 12. Negative Provenance Audit
- NORMAL_NEGATIVE_SOURCE=ORIGINAL_TRAIN_DATALOADER; EXTRA_NEGATIVE_ADDED=False; BOUNDARY_NEGATIVE_USED=False; FROZEN_HARD_NEGATIVE_USED=False; DIFFUSION_NEGATIVE_USED=False.

## 13. seed999 Validation
- B0 R10=0.07166146, N10=0.03881882, R20=0.10677472, N20=0.04777998.
- D6 R10=0.07194737, N10=0.03876227, R20=0.10696115, N20=0.04767566; U4=+0.0524%; primary positives=2/4.

## 14. seed1000 Validation
- B0 R10=0.07127588, N10=0.03852826, R20=0.10817544, N20=0.04792441.
- D6 R10=0.07126731, N10=0.03853651, R20=0.10772288, N20=0.04781403; U4=-0.1598%; primary positives=1/4.

## 15. D6 vs B0
- seed999 U4=+0.0524%; seed1000 U4=-0.1598%; cross-seed mean U4=-0.0537%.

## 16. PASS / FAIL
- ROUND29_STATUS = `FAIL`; minimum_success=False; strong_pass=False; positive_but_below_target=False. Baby Test remains closed.

## 17. Unexpected Observations
- The semantic integration correction materially changed the failure mode relative to Round28: Round28 D5 had cross-seed mean U4 -6.0044%, whereas Round29 D6 is near-neutral at -0.0537%. Thus using preference-aligned Diffusion evidence as a detached positive teacher removes the large systematic ranking damage caused by treating the same evidence as a negative.
- The mechanism itself remains healthy: target stability is essentially preserved (0.169504), TRUE-user alignment beats SHUFFLED, preference compatibility is positive, the positive teacher is norm-preserved, and ranking gradients do not enter Diffusion.
- Nevertheless the ranking gain is not robust: seed999 is only +0.0524% U4 (2/4 primary positive) and seed1000 is -0.1598% U4 (1/4 primary positive), so the preregistered cross-seed success criterion fails.
- No post-result rescue, parameter sweep, Test access, or Sports/Electronics access was performed.

## 18. Git Commit
- Repository history is authoritative. Implementation/preflight and final Validation evidence are committed separately on the Round29 branch.
