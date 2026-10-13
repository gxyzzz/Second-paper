# Second-paper

Official reproducibility repository for the current multimodal recommendation method built on a generic multimodal recommender backbone, **CoLiftRec**, and **Preference-Directed Diffusion Boundary Refinement (PDBR)**.

All current formal results use **seed1000**.

## 1. Project introduction

The method addresses two different sources of ranking error. First, multimodal content relevance can contain item-generic semantic similarity that is not equivalent to a user's preference. Second, after this generic component is reduced, difficult candidates remain around the Top-100 ranking boundary. The current pipeline therefore separates semantic debiasing from preference-directed boundary supervision.

## 2. Current method

```text
Multimodal Recommender Backbone
  -> Top-100 candidates / representations
  -> Full CoLiftRec
  -> Preference-Directed Diffusion Boundary Refinement (training only)
  -> Validation-selected recommender checkpoint
  -> frozen Test evaluation
```

Diffusion provides preference-directed boundary supervision during training; the final recommender absorbs this supervision, while Test inference uses no additional Diffusion sampling.

```text
diffusion_inference_calls = 0
```

## 3. CoLiftRec

For modality `m`, original relevance contains both generic semantic relevance and user-specific preference. CoLiftRec estimates an item-generic semantic background and constructs the personalized lift

```text
L_ui^m = z_ui^m - lambda_m * mu_i^m
```

before combining the enabled text, attribute, and visual lifts with the backbone score. Its purpose is to reduce the interference of generic content similarity in personalized recommendation.

Frozen dataset-specific parameters are stored in `src/configs/model/CoLiftRecDiffusion.yaml`.

## 4. Preference-Directed Diffusion Boundary Refinement

After CoLiftRec, behaviorally plausible and semantically similar candidates can still be hard to distinguish near the ranking boundary. PDBR uses the current user, the boundary candidate representation, and the observed positive to learn a **user-conditioned local preference direction**.

The Diffusion network uses three conditions during training:

```text
BASE     = zero-user condition
TRUE     = current-user condition
SHUFFLED = shuffled-user condition
```

The preference direction is the tangent component of

```text
positive_embedding - boundary_candidate_embedding
```

around the boundary candidate. The resulting candidate refinement is deliberately small and bounded.

## 5. Bounded boundary supervision

The frozen M25 configuration is:

```text
seed              = 1000
lambda_HN         = 0.20
half_margin       = 0.25
theta_max_deg     = 5.0
local_t           = [1, 2, 3, 4, 5]
warmup_epochs     = 10
formal_epochs     = 60
diffusion_steps   = 2 per active training batch
inference_enabled = false
```

The requested score movement is at most 25% of the current positive-candidate margin, subject to an angular trust region of 5 degrees. The recommender objective is

```text
L_total = L_normal + 0.20 * L_boundary
```

Boundary bands and curriculum are frozen in the formal runtime.

## 6. Overall framework

For a full formal run, two fair branches start from the same seed1000 initialization and the same normal training plan:

```text
Shared seed1000 initialization
  |-- Baseline branch -> Validation-selected checkpoint
  |      |-- MSCA Test
  |      `-- MSCA + Full CoLiftRec Test
  |
  `-- Full branch -> PDBR boundary supervision -> Validation-selected checkpoint
         `-- MSCA + Full CoLiftRec + PDBR Test
```

This design preserves the successful experimental protocol. The full branch is not continued from the baseline branch's best checkpoint.

## 7. seed1000 policy

Baby, Sports, and Electronics all use:

```text
seed = 1000
```

The same policy is used for the three-stage ablation:

```text
MSCA
MSCA + CoLiftRec
MSCA + CoLiftRec + PDBR
```

Historical experiment branches and evidence are preserved separately and are not rewritten by the current runtime.

## 8. Datasets

Supported publication datasets:

- Baby (`baby`)
- Sports (`sports`)
- Electronics (`elec`)

Local data assets live under `data/<dataset>/` and are not committed.

## 9. Terminal commands

Run from the repository root after activating the `gume` environment.

```bash
cd /home/gxy/code/Second-paper
conda activate gume

python src/main.py -m MSCA -d baby --stage full
python src/main.py -m MSCA -d sports --stage full
python src/main.py -m MSCA -d elec --stage full
```

Individual stages are also supported:

```bash
python src/main.py -m MSCA -d baby --stage msca
python src/main.py -m MSCA -d baby --stage coliftrec
python src/main.py -m MSCA -d baby --stage full
```

Engineering smoke:

```bash
python src/main.py -m MSCA -d baby --stage full --smoke
```

Smoke uses small batches, exercises warmup and active boundary refinement, and never accesses formal Test.

Configuration-only dry-run:

```bash
python src/main.py -m MSCA -d baby --stage full --dry-run
```

See `TERMINAL_COMMANDS.md` for the complete command set.

## 10. Training / Validation / Test workflow

A formal full run executes:

```text
1. Create shared seed1000 initialization
2. Train baseline recommender for 60 epochs
3. Select baseline checkpoint using Validation Full CoLiftRec R@20
4. Train the PDBR branch from the same initialization for 60 epochs
5. Apply bounded preference-directed boundary auxiliary supervision during training
6. Select full checkpoint using Validation Full CoLiftRec R@20
7. Freeze both selected checkpoints
8. Evaluate Test exactly once
```

Test never selects an epoch, checkpoint, seed, CoLiftRec parameter, or Diffusion parameter. PDBR is not called during Test ranking.

## 11. Three-domain Test results

### Baby — seed1000

| Method | R@10 | N@10 | R@20 | N@20 | R@50 | N@50 |
|---|---:|---:|---:|---:|---:|---:|
| MSCA | 0.067720 | 0.037031 | 0.103784 | 0.046318 | 0.174483 | 0.060638 |
| MSCA + CoLiftRec | 0.071226 | 0.038945 | 0.107293 | 0.048269 | 0.179539 | 0.062955 |
| MSCA + CoLiftRec + PDBR | **0.072726** | **0.039750** | **0.110479** | **0.049464** | 0.179375 | **0.063351** |

### Sports — seed1000

| Method | R@10 | N@10 | R@20 | N@20 | R@50 | N@50 |
|---|---:|---:|---:|---:|---:|---:|
| MSCA | 0.078633 | 0.042836 | 0.117032 | 0.052734 | 0.189671 | 0.067531 |
| MSCA + CoLiftRec | 0.083048 | 0.044891 | 0.124388 | 0.055527 | 0.195963 | 0.070075 |
| MSCA + CoLiftRec + PDBR | **0.086167** | **0.046832** | **0.126438** | **0.057239** | **0.197375** | **0.071669** |

### Electronics — seed1000

| Method | R@10 | N@10 | R@20 | N@20 | R@50 | N@50 |
|---|---:|---:|---:|---:|---:|---:|
| MSCA | 0.049395 | 0.027998 | 0.072552 | 0.033973 | 0.116521 | 0.042912 |
| MSCA + CoLiftRec | 0.050878 | 0.028767 | 0.074946 | 0.034986 | 0.119535 | 0.044051 |
| MSCA + CoLiftRec + PDBR | **0.053055** | **0.029549** | **0.078976** | **0.036249** | **0.123687** | **0.045357** |

## 12. Improvement summary

`U4` is the mean relative change over R@10, N@10, R@20, and N@20.

| Dataset | CoLiftRec vs MSCA | PDBR vs CoLiftRec | Full vs MSCA |
|---|---:|---:|---:|
| Baby | +4.48% | **+2.40%** | +6.99% |
| Sports | +5.50% | **+3.20%** | +8.87% |
| Electronics | +3.01% | **+4.00%** | +7.13% |

PDBR is **4/4 positive on the four primary Test metrics for all three datasets**. Sports and Electronics are also 6/6 positive over the six reported metrics; Baby is 5/6 positive.

Machine-readable current references are stored under `docs/evidence/current/`.

## 13. Repository structure

```text
Second-paper/
├── README.md
├── PROVENANCE.md
├── TERMINAL_COMMANDS.md
├── src/
│   ├── main.py
│   ├── models/
│   ├── modules/
│   │   ├── coliftrec.py
│   │   └── preference_diffusion.py
│   ├── pipelines/
│   │   ├── dataset_config.py
│   │   ├── preference_pipeline.py
│   │   └── runner.py
│   └── configs/
├── scripts/
│   └── audit_main_runtime_parity.py
├── tests/
│   └── test_preference_diffusion.py
└── docs/
    ├── REPRODUCTION.md
    ├── MAIN_SEED1000_MIGRATION_REPORT.md
    └── evidence/current/
```

Development experiment history is intentionally separate from the formal `src/` runtime. The current runtime does not import experiment code.

## 14. Reproducibility

The formal configuration is frozen in `src/configs/model/CoLiftRecDiffusion.yaml`. Dataset-specific MSCA overrides remain under `src/configs/model/MSCA/`.

The following gates are provided:

```bash
python scripts/audit_main_runtime_parity.py
python -m unittest -v tests.test_preference_diffusion
python src/main.py -m MSCA -d baby --stage full --dry-run
python src/main.py -m MSCA -d baby --stage full --smoke
```

The deterministic parity audit checks the frozen seed, CoLiftRec parameters, PDBR parameters, boundary curriculum, MSCA overrides, zero Test-time Diffusion calls, and absence of runtime dependency on experiment implementations.

## 15. Acknowledgement

This repository is built upon **MSCA** and the **MMRec** framework. We thank the original authors and maintainers for their work.

- MSCA upstream: `recomall/MSCA`
- Frozen upstream commit: `48455de8efa943e16d49db665e7f2fcb0c6c5e17`
- MMRec: `enoche/MMRec`

The upstream GNU GPL v3.0 license is retained in `LICENSE`.
