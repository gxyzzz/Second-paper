# Round21 GDNSM-style Curriculum Hard-Negative Diffusion

- Baby only, seeds 999/1000.
- Test/Sports/Electronics closed.
- 64D current MSCA user/item latent; current projected text/visual 64D conditions.
- epsilon-prediction DDPM, T=25 linear beta 1e-4..0.02, MLP 320-256-256-64, dropout .1.
- modality CFG dropout .05/.05; user never dropped; guidance 1.1.
- generation starts from Gaussian and returns synthetic 64D negative latent directly used in auxiliary BPR.
- normal MSCA objective/normal negatives remain unchanged; lambda_HN=.20.
- refresh every 5 recommender epochs; diffusion pretrain 5 epochs then 1 epoch each refresh.
- M=2 each for V/T/TV; curriculum g(e): 0 before epoch10, 2 at 10-19, 4 at 20-29, 6 at >=30.
- C1 matched control uses online current-backbone Top100 unobserved ranks 61-100 / 31-60 / 11-30, M=2 each.
- B0/C1/D1 share exact initialization and normal random plan. Diffusion RNG is isolated from recommender RNG.
- checkpoint selection is Full-CoLift Validation R20 for all variants.
- no Test access in this round.
