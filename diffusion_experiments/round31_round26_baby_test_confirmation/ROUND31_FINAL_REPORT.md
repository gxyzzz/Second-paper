# Round31 Final Report — Frozen Round26 Baby Test Confirmation

## 1. Objective
Confirm whether the frozen Round26 D3 Validation gain over B0 survives on Baby Test, without retraining, reselection, or Test-driven tuning.

## 2. Frozen Round26 Protocol
Source protocol: `ROUND26_PREFERENCE_DIRECTED_DIFFUSION_V1`. Round26 settings remain frozen: 10 warmup epochs, 60 formal epochs, lambda_HN=0.20, theta_max=5°, half_margin=0.50, local_t=[1,2,3,4,5], Diffusion active steps=2.

## 3. Checkpoint Integrity
Checkpoint audit PASS = True.
- seed999 B0: epoch 39; SHA256 `fc00e034f280c49da1511eb14786e79c4e25e9be6b2660e2b642df5c009bde20`; PASS=True.
- seed999 D3: epoch 24; SHA256 `d369e6bd6274c4e9886a6d91db0c46a3957dc13188321ee3195c194a5498ac3d`; PASS=True.
- seed1000 B0: epoch 31; SHA256 `3d099347a4bbb0ce9df143f68bdb96d2c513fb99852b2e1817f84bf08577c944`; PASS=True.
- seed1000 D3: epoch 22; SHA256 `d13817bf6fb7c47e765a5425e48d2e1a0e44216bef3505d05c0ef1df6358db78`; PASS=True.

## 4. Test Protocol Audit
Protocol audit PASS = True; same Test users=True; same ground truth=True; same Full CoLift parameters=True; all checkpoint repeats exact=True; Diffusion inference calls all zero=True.

## 5. No-Test-Tuning Statement
Baby Test was opened only for frozen Round31 confirmation. Test was not used for parameter, epoch, threshold, architecture, angle, lambda, margin, or curriculum selection. No post-Test rescue was performed.

## 6. seed999 Validation Recap
B0: R10=0.07194566, N10=0.03885132, R20=0.10713900, N20=0.04780885, R50=0.17358994, N50=0.06106452.
D3: R10=0.07200260, N10=0.03925706, R20=0.10785433, N20=0.04839351, R50=0.17398566, N50=0.06162610.
Validation U4=+0.7535%; primary positives=4/4; overall positives=6/6.

## 7. seed999 Test
B0: R10=0.07265945, N10=0.03988176, R20=0.10824611, N20=0.04902828, R50=0.17607191, N50=0.06280770.
D3: R10=0.07270488, N10=0.04000267, R20=0.10801361, N20=0.04909718, R50=0.17854021, N50=0.06340447.
Test U4=+0.0729%; primary positives=3/4; overall positives=5/6.
Relative deltas: R10=+0.0625%, N10=+0.3032%, R20=-0.2148%, N20=+0.1405%, R50=+1.4019%, N50=+0.9502%.

## 8. seed1000 Validation Recap
B0: R10=0.07145588, N10=0.03869697, R20=0.10777688, N20=0.04794087, R50=0.17519322, N50=0.06144520.
D3: R10=0.07279163, N10=0.03926039, R20=0.10914607, N20=0.04846495, R50=0.17488209, N50=0.06162380.
Validation U4=+1.4222%; primary positives=4/4; overall positives=5/6.

## 9. seed1000 Test
B0: R10=0.07122589, N10=0.03894483, R20=0.10729273, N20=0.04826928, R50=0.17953900, N50=0.06295467.
D3: R10=0.07114066, N10=0.03926950, R20=0.10871461, N20=0.04897275, R50=0.17910860, N50=0.06318172.
Test U4=+0.8742%; primary positives=3/4; overall positives=4/6.
Relative deltas: R10=-0.1197%, N10=+0.8337%, R20=+1.3252%, N20=+1.4574%, R50=-0.2397%, N50=+0.3607%.

## 10. Validation vs Test Comparison
Validation cross-seed mean U4=+1.0879%; Test cross-seed mean U4=+0.4735%; change=-0.6144%.

## 11. Cross-Seed Mean Utility
seed999 Test U4=+0.0729%; seed1000 Test U4=+0.8742%; mean=+0.4735%.

## 12. Metric-Level Analysis
- seed999: non-positive metrics = ['R20']; primary positives 3/4; overall positives 5/6.
- seed1000: non-positive metrics = ['R10', 'R50']; primary positives 3/4; overall positives 4/6.

## 13. STRONG_PASS / PASS / MIXED / FAIL
`ROUND31_STATUS = PASS_BELOW_TARGET`.

## 14. Data Access Audit
- Baby Validation: historically used for Round26 checkpoint selection.
- Baby Test: opened only in Round31 for frozen confirmation.
- Sports: NOT ACCESSED.
- Electronics: NOT ACCESSED.

## 15. Git Commit
Round31 evaluator/audits and Test evidence are committed separately on the dedicated Round31 branch.

## Scientific Interpretation
If D3 improves Test, the correct interpretation is training-time Diffusion-guided boundary refinement: Diffusion is not called during Test inference; the trained recommender has absorbed the auxiliary boundary supervision.
