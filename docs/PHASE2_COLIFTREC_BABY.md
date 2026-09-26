# Phase 2  Full CoLiftRec T/A/V Migration on Baby

## Scope

Full CoLiftRec was migrated as reusable source modules into Second-paper. No historical CoLiftRec ranking, generic-background cache, Top100 cache, or historical MSCA checkpoint is used at runtime.

The only MSCA checkpoint used here is the current-run Baby seed-999 checkpoint trained from random initialization in Phase 1:

- epoch: 37
- SHA256: aa389ef8576e491ffb56eaec9d165444487bef9c1e542f280e991e2e063cc1cb

Current-run assets are regenerated under runs/ and remain untracked.

## Frozen definition

For each modality m:

L_ui^m = row-z(z_ui^m - lambda_m * mu_i^m)

Final score coordinate:

row-z(S_MSCA) + alpha_T * L_T + alpha_A * L_A + alpha_V * L_V

Baby frozen parameters:

- lambda_T = 1.0
- lambda_A = 0.75
- lambda_V = 0.25
- alpha_T = 0.25
- alpha_A = 0.15
- alpha_V = 0.025

Attribute uses TRAIN-history profiles with:

- title TF-IDF: 0.45
- brand one-hot: 0.20
- description TF-IDF: 0.35

All parameters and branch enables are controlled by src/configs/second_paper.yaml.

## Gate audit

- COLIFTREC_ALPHA0_EXACT = PASS
- COLIFTREC_SHAPE_ID_EXACT = PASS
- COLIFTREC_TRAIN_ONLY_BACKGROUND = PASS
- COLIFTREC_BRANCH_TOGGLES = PASS
- COLIFTREC_CONFIG_CONTROL = PASS
- TEST_ACCESSED = false
- TEST_USED_FOR_SELECTION = false

Config refactor parity:

- all metrics exact
- TV ranking IDs exact
- raw-Attribute ranking IDs exact
- Full T/A/V ranking IDs exact
- Full-score maximum floating-point difference: 9.5367431640625e-07
- difference source: float32 addition order only

## Formal Baby Validation

| Method | R10 | N10 | R20 | N20 | R50 | N50 |
|---|---:|---:|---:|---:|---:|---:|
| MSCA | 0.06713416 | 0.03666963 | 0.10347268 | 0.04590319 | 0.16968148 | 0.05914653 |
| MSCA + CoLiftRec T/V | 0.07013885 | 0.03787216 | 0.10523363 | 0.04680999 | 0.17128819 | 0.06001176 |
| MSCA + T/V + raw Attribute | 0.07215565 | 0.03871028 | 0.10638902 | 0.04743498 | 0.17340098 | 0.06082712 |
| MSCA + Full CoLiftRec T/A/V | 0.07253278 | 0.03890351 | 0.10686472 | 0.04763388 | 0.17409954 | 0.06105030 |

Full CoLiftRec relative to current-run MSCA:

- R10: +0.00539862 / +8.0415%
- N10: +0.00223388 / +6.0919%
- R20: +0.00339205 / +3.2782%
- N20: +0.00173070 / +3.7703%
- R50: +0.00441806 / +2.6037%
- N50: +0.00190377 / +3.2187%

Gates:

- Full vs MSCA primary positive: 4/4
- Full vs T/V primary positive: 4/4
- Full vs raw Attribute primary positive: 4/4

PHASE2_COLIFTREC_BABY = PASS

## Frozen Baby Test

After all Validation-side gates passed, the frozen Baby CoLiftRec parameters were evaluated once on x_label == 2 Test targets. User semantic histories and generic backgrounds remain TRAIN-only.

| Method | R10 | N10 | R20 | N20 | R50 | N50 |
|---|---:|---:|---:|---:|---:|---:|
| MSCA | 0.06975933 | 0.03805872 | 0.10383620 | 0.04685222 | 0.17140490 | 0.06057006 |
| MSCA + Full CoLiftRec T/A/V | 0.07239375 | 0.03979682 | 0.10820199 | 0.04902839 | 0.17597763 | 0.06281241 |

Full CoLiftRec delta vs current-run MSCA:

- R10: +0.00263441
- N10: +0.00173810
- R20: +0.00436579
- N20: +0.00217617
- R50: +0.00457273
- N50: +0.00224235

Frozen Test gates:

- primary positive: 4/4
- overall positive: 6/6
- BABY_COLIFTREC_TEST_RUN_COUNT = 1
- TEST_USED_FOR_SELECTION = false
- NO_POST_TEST_TUNING = true

PHASE2_COLIFTREC_BABY = COMPLETE
