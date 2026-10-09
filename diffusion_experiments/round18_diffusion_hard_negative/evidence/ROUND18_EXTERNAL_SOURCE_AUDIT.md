# Round18 External Source Audit

## CCDRec

- Repo: `Yimeng-yang/CCDRec`
- Branch: `main`
- Commit: `4be16f3761f2d1a0b35ba61395b601eb3e55c949`
- Inspected: `ccdrec.py`, `diffusion_ver15.py`, `Unet.py`.
- Calibration used: reverse-diffusion trajectory and quarter/half/three-quarter/full stages as a curriculum difficulty axis.
- Not borrowed: the complete recommender, complete sampler, multimodal-alignment objective, or reported hyperparameter search.
- External baseline was **not run**.

## SDPDRec

- Repo: `YongfuZha/SDPDRec`
- Branch: `main`
- Commit: `e949c1ae2bfccffe4d55819a369cd1ecf1fbdb1a`
- Inspected: `diffusier.py`, `smore_model.py`, `training.py`, `main.py`.
- Calibration used: 32-step cosine schedule, sinusoidal timestep representation, history ID/visual/text conditioning, self-attention, and training/inference separation.
- Not borrowed: personalized mean/variance noise prior, SDPDRec recommender path, contrastive loss, or stochastic catalog hard-negative sampler.
- External baseline was **not run**.

## History length resolution

SDPDRec does not expose a fixed Baby history length of 20. Its current code sets `max_len` to the integer mean interaction length and randomly resamples each user's history to that size. The Round18 advisor protocol says to use the most recent 20 prefix items when the official code has no fixed length, so Round18 fixes `history_len=20` without search.
