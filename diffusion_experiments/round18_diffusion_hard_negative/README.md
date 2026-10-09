# Round18 HCD-HNC

History-Conditioned Diffusion Hard-Negative Curriculum.

Diffusion is training-only. It predicts frozen positive final-item representations from noisy x_t under frozen starting-backbone history ID/visual/text conditions. Reverse-stage queries only select real catalog negative item IDs from a frozen Full-CoLift rank 11-30 boundary. Recommender fine-tuning uses the original MSCA `calculate_loss`; Diffusion is unloaded for recommendation evaluation.
