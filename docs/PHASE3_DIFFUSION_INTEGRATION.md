# Phase 3 — Condition-Adaptive Diffusion Semantic Purification Integration

## Scope

This phase integrates only the final M31/M31C/M32-line Condition-Adaptive Diffusion Semantic Purification infrastructure. No historical diffusion checkpoint or purified feature is consumed, and no formal Diffusion Validation grid is run.

## Frozen method

- native state: Text384 + Visual4096 = 4480D
- denoiser: 4480 -> 1024 -> 512 -> 1024 -> 4480
- prediction target: x0
- schedule: cosine
- diffusion steps: 50
- time embedding: 64
- condition projection path retained from final M31 core
- block-balanced reconstruction: 0.5 * MSE(Text) + 0.5 * MSE(Visual)
- contrastive weight: lambda_ctr = 0.1 for smoke
- classifier-free condition dropout: p_uncond = 0.15
- optimizer: AdamW, lr=1e-3, weight_decay=1e-4
- condition: Norm[c_collab + beta * (c_final - c_collab)], beta constrained to [0,1]
- Attribute is not diffusion-purified.

No beta/t_edit/guidance/rho selection is performed in this phase.

## Current-run provenance

Condition endpoints come only from the current Second-paper MSCA run:
- checkpoint epoch: 37
- checkpoint SHA256: aa389ef8576e491ffb56eaec9d165444487bef9c1e542f280e991e2e063cc1cb
- source: runs/assets/msca_baby_seed999/embeddings.npz
- c_collab: current-run collab_item
- c_final: current-run final_item

Diffusion training-item eligibility is unique itemID appearing in x_label==0 TRAIN only:
- TRAIN items: 7047
- TEST_ACCESSED = false
- TRAIN_ONLY_ITEM_AUDIT = PASS

## Integration smoke

The gume/RTX 5090 smoke verifies:
- module import: PASS
- current-run condition extraction: PASS
- beta 0 / 0.5 / 1.0 condition shape: (7050,64)
- native joint state shape: (8,4480)
- small-batch forward: PASS
- small-batch backward: PASS
- AdamW step: PASS
- checkpoint save/load with strict=True: PASS
- purified Text shape: (8,384)
- purified Visual shape: (8,4096)
- finite check: PASS

The smoke does not represent formal diffusion training.

## rho=0 full Validation identity

The Diffusion framework generated rho_T=rho_V=0 Text/Visual assets and the complete reusable CoLiftRec Validation pipeline consumed those assets.

Compared with the frozen current-run MSCA + Full CoLiftRec Validation output:
- users: exact
- candidate item IDs: exact
- MSCA scores: exact
- Text/Visual CoLiftRec scores: exact
- raw-Attribute scores: exact
- Full CoLiftRec scores: exact
- max_abs_diff: 0.0
- ranking IDs: exact
- metrics: exact

DIFFUSION_INTEGRATION_IDENTITY = PASS

## State

PHASE1_MSCA_FROM_SCRATCH = COMPLETE
PHASE2_COLIFTREC_BABY = COMPLETE
PHASE3_DIFFUSION_INTEGRATION = READY
DIFFUSION_FORMAL_TRAINING = NOT_STARTED
