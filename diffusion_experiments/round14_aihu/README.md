# ROUND14_AIHU

Anchored Inference-Aligned Hard-Shell User Purification.

Frozen protocol before formal execution:
- source: exp/round13-iudlp-20261008 @ 56cbd7597161176d67cce7ac7044dd56f8362f1a
- recommender fully frozen; trainable only ConditionEncoder + Text/Visual denoisers
- latent interface unchanged: post modality-graph 64D Text/Visual, pre adaptive fusion
- A1: dual TRUE/ZERO reconstruction anchor; preference t remains random 1..20; original TRAIN negative
- A2: A1 plus preference/user supervision fixed at t_pref=3; reconstruction remains random 1..20
- A3: A2 plus frozen TRAIN-only Full-CoLiftRec hard negative uniformly sampled from ranks 6-30
- rho=0.20, lambda_rec=1.0, lambda_anchor=0.25, lambda_user=0.5, margin=0.05
- five epochs; best epoch first requires mean(TRUE-SHUFFLED margin)>=0, then maximum Validation R20; if no sane epoch exists, max-R20 is report-only and PASS-ineligible
- inference unchanged from Round13: frozen Top100, one-step t=3, noise seeds 20261301/20261302
- score integration unchanged: frozen Full-CoLiftRec + original frozen scorer response to purified latent
- Test/Sports/Electronics closed; main untouched

Pre-registered interpretation of the Round14 gate's alternative "hard-shell clearly better" clause: if A3 mean U is not greater than A2 mean U, A3 must improve the aggregate `NetCross@10 + NetCross@20` across seed999/1000 by at least +2 over A2. This operationalization is frozen before any formal Round14 result is observed.
