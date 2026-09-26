# Phase 1 — Baby MSCA From-Scratch Reproduction

## Protocol

- environment: gume
- GPU: NVIDIA GeForce RTX 5090
- dataset: Baby
- seed: 999
- initialization: random
- selection metric: Validation Recall@20 only
- training-time Test evaluation: disabled
- historical checkpoint dependency: none
- current-run graph caches regenerated inside Second-paper

TRAINING_TEST_ISOLATION = PASS
TEST_USED_FOR_SELECTION = false

## Validation freeze

- best epoch: 37 (0-based)
- R10: 0.0671
- N10: 0.0367
- R20: 0.1035
- N20: 0.0459
- R50: 0.1697
- N50: 0.0591
- checkpoint: runs/checkpoints/MSCA-Sep-26-2026-12-48-04.pth
- checkpoint SHA256: aa389ef8576e491ffb56eaec9d165444487bef9c1e542f280e991e2e063cc1cb

BABY_MSCA_FROM_SCRATCH_FROZEN = TRUE

## Historical parity audit

Historical Frozen MSCA Validation reference from DiCalRec M10A:

- R10: 0.06727559
- N10: 0.03675184
- R20: 0.10258556
- N20: 0.04572824
- R50: 0.16836029
- N50: 0.05889247

The from-scratch current-run checkpoint is in the same performance regime and does not show a systematic degradation. Dataset cardinalities, x_label split, feature hashes, model configuration, embedding dimensions, and evaluator definitions match the audited frozen baseline.

MSCA_FROM_SCRATCH_PARITY = PASS

## Explicit frozen Test

The frozen Validation-selected checkpoint was evaluated exactly once after training completed.

- R10: 0.0698
- N10: 0.0381
- R20: 0.1038
- N20: 0.0469
- R50: 0.1714
- N50: 0.0606

BABY_MSCA_TEST_RUN_COUNT = 1
TEST_USED_FOR_SELECTION = false
NO_POST_TEST_TUNING = true
