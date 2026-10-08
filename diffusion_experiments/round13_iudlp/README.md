# ROUND13_IUDLP

Inference-Preserved User-Discriminative Latent Purification.

Frozen protocol:
- source: exp/round12-ucldt-20261008 @ 5e217a1
- recommender: fully frozen
- trainable: ConditionEncoder + Text/Visual 64D latent x0 denoisers only
- latent: post modality-graph, pre adaptive-fusion 64D Text/Visual
- variants: C0 frozen baseline; D0 item-only differential; D1 TRUE-user differential; D2 TRUE-vs-SHUFFLED supervised differential
- rho=0.20, T_latent=20, t_infer=3, inference noise seeds 20261301/20261302
- loss: L_rec + 0.25 L_diff; D2 adds 0.5 L_user with margin 0.05
- five training epochs; select best Validation R20 within each variant
- candidate recall is frozen historical Top100; Diffusion only perturbs pair-specific Text/Visual latent inside Top100
- final score anchor is frozen Full-CoLiftRec score plus the score change produced by purified latent through the original frozen fusion/scorer. This is algebraically the frozen CoLiftRec correction applied to the augmented MSCA scorer, while preserving bitwise rho=0 baseline parity and avoiding float32 subtract/add loss.
- no direct Diffusion ranking score/residual; no raw-feature/graph/multistep Diffusion; no rho/loss search
- Test/Sports/Electronics/main remain closed

The first smoke parity attempt exposed float32 non-invertibility from computing `(full-msca)+msca`, with <=9.54e-7 score differences that changed near-tie rankings. The implementation was corrected before any formal run to anchor at frozen `full_coliftrec` and add only the original scorer response `(s_aug-s_base)`. After this correction rho=0 is exact.
