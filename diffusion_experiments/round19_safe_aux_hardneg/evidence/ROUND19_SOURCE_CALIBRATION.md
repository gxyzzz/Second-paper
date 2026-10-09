# Round19 Source Calibration

## Original MSCA pipeline

The repository's original `TrainDataLoader` is retained as the source of full TRAIN user-positive interactions and normal negatives. `MSCA.calculate_loss` consists of normal BPR plus the existing modality/collaborative contrastive alignment and regularization. Round19's loss wrapper mirrors that expression and adds only `0.20 * L_aux` for A2/A3/A4. With `lambda_hard=0`, numerical parity against the original `calculate_loss` is required to be below `1e-6`.

## CCDRec official source

Calibrated against local clone `/tmp/round18_sources/CCDRec`, repository `Yimeng-yang/CCDRec`, branch `main`, commit `4be16f3761f2d1a0b35ba61395b601eb3e55c949`.

In official `src/models/ccdrec.py::calculate_loss`, the model keeps BPR supervision using the ordinary sampled negative and separately computes a BPR term using the diffusion-mined negative. The two are mixed by a weight; the ordinary negative is not globally replaced by the hard negative. Round19 does not copy CCDRec's full model or its exact objective. It follows the advisor-frozen safer integration: preserve the complete original MSCA objective and add a small auxiliary hard-negative BPR term.

## Frozen Round18 generator

Round19 does not retrain or modify Diffusion. It reuses the Round18 HCDDiffuser epoch10 checkpoints exactly:

- seed999 SHA256 `a4b6aba18442fe60e9c6648faa8d28df491ca919b4b0d7fb98b86aa7a6f805cf`
- seed1000 SHA256 `a6fadc12432ffe4f6d5ba33b28be75250070e2d4c5a4a9c484ce3614f3f0d922`

The Diffusion stage is fixed to Q75 (24 of 32 reverse steps), with the Round18 history-conditioning architecture and history length 20 unchanged. Generator parameters are frozen, and queries/maps are precomputed before recommender training.
