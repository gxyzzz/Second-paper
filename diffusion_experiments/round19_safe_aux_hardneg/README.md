# Round19 SA-DHN

Safe Auxiliary Diffusion Hard-Negative Training.

Round19 retains the original full MSCA TRAIN distribution, original normal-negative sampling, original contrastive alignment and regularization. A fixed-weight (`lambda_hard=0.20`) auxiliary BPR term is the only additional training signal for A2/A3/A4.

- A0: frozen starting checkpoint.
- A1: original full-TRAIN continuation, 3 epochs.
- A2: A1 + safe rank21-40 auxiliary negative.
- A3: A1 + positive-similarity Top5 auxiliary negative from frozen Full-CoLift rank21-100.
- A4: A1 + frozen Round18 history-conditioned Diffusion Q75 Top5 auxiliary negative from the identical safe pool.

All TRAIN-observed items and the auxiliary target are excluded from the safe pool. Rank1-20 is protected. Q75, candidate pool, top5 policy, `lambda_hard`, optimizer family, learning rate, epoch count and frozen CoLift parameters are preregistered and are not searched. Diffusion is not loaded during recommender fine-tuning or recommendation evaluation.
