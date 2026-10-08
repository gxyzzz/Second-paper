# Round12 — User-Conditioned Latent Diffusion Training

Protocol: `ROUND12_UCLDT_V1`.

Source branch/evidence: `exp/round11-cdtc-20261008` at `e9c41390a4bbd16ca3beeb1934607c57fbbe6491`.

## Frozen implementation choices before formal results

- Latent interface: item-side Text/Visual 64D outputs of `semantic_encode`, after the existing modality projection/gate and modality graph propagation, immediately before `query_common` adaptive fusion. This is the first clean, explicitly materialized modality-specific tensor at the final fusion interface. The earlier pre-propagation projected/gated tensor was audited but is not selected because a user-specific perturbation there would alter graph propagation globally; the selected interface allows an exact user-specific replacement at the original fusion consumer without inventing a scoring head.
- Diffusion operates on L2-normalized latent direction. The conservative augmented view is normalized in direction and then rescaled by the source latent norm before insertion into MSCA fusion. This makes rho=0 an exact identity despite native Text/Visual latent norms not being 1.
- C0/D0/D1 use the same checkpoint, Adam optimizer reset, LR = 0.1 × original LR, 3 epochs, original batch size and original TrainDataLoader negative sampler. Epoch RNG is reset from the same `(backbone, epoch)` seed so training order/negative sampling are matched across variants.
- Selective freezing is used. Frozen: user/item ID embeddings and collaborative graph core. Trainable recommender subset for all variants: `image_trs`, `text_trs`, `gate_v`, `gate_t`, `query_common`. Raw pretrained feature tables remain frozen. D0/D1 additionally train the condition encoder and two latent denoisers.
- `L_rec_raw` is the exact original MSCA objective (BPR + original contrastive terms + original regularization). `L_rec_aug` reuses MSCA's original adaptive multi-view fusion and `cal_bpr_loss`; only the Text/Visual item latent at the selected interface is replaced by the user-conditioned augmented latent. No independent ranking head or score residual exists.
- Best epoch is selected independently for C0/D0/D1 by Validation Full-CoLiftRec R20. Diffusion is OFF at inference; standard updated MSCA embeddings are retrieved and frozen Full CoLiftRec is applied.
- Operationalization frozen for expansion-gate wording “D1 mean clearly exceeds D0 mean”: `mean_U(D1 vs C0) - mean_U(D0 vs C0) >= 0.0010` (0.10 percentage points). The alternative branch `D1 >= D0 on both preflight backbones` is checked first.
- Test, Sports and Electronics remain CLOSED. No hyperparameter search is allowed.

### Numerical/fairness exactness amendment before any continuation epoch

The first formal launcher was stopped at the initial M0 replay gate before any continuation epoch because GPU sparse float32 replay produced tiny near-tie differences versus the historical frozen asset: seed999 raw Top100 score max difference was `2.86e-6`, 12/1,944,500 candidate cells differed, and only N20/N50 changed by `2.26e-7`. This is recorded as numerical replay parity rather than falsely labeled rank-exact. The frozen M0 metrics remain the historical reference; all C0/D0/D1 continuation outputs use the same current Round12 evaluator. The replay gate is fixed to metric max absolute difference <=1e-6, candidate-cell mismatch fraction <=1e-4, and raw Top100 score max absolute difference <=1e-5.

`TrainDataLoader` samples negatives online with Python `random.sample()` and shuffles `all_items` in `pretrain_setup()`. Therefore Round12 now freezes `pretrain_setup` RNG to `202612700 + backbone_seed`, independent of C0/D0/D1, and resets the same `(backbone, epoch)` RNG before each epoch. Smoke must show C0 and D1 first `(user, positive, negative)` batch is byte-identical under the same epoch seed.
