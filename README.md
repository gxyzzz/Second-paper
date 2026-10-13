# Second-paper

Official reproducibility repository for our multimodal recommendation study built on MSCA / MMRec.
The current method combines **CoLiftRec** with **Preference-Directed Diffusion Boundary Refinement**.

## Introduction

The current pipeline addresses two complementary problems in multimodal recommendation:

1. **CoLiftRec** removes item-generic multimodal relevance from user-item semantic scores and retains the user-specific lift used for candidate reranking.
2. **Preference-Directed Diffusion** focuses on difficult ranking-boundary candidates during training. Conditioned on the current user and candidate multimodal information, the diffusion model learns a local preference direction and applies a bounded refinement to construct more informative boundary supervision.

The current unified experimental seed is **1000** for Baby, Sports, and Electronics.

The frozen preference-directed configuration used for the current three-domain results is:

```text
lambda_HN   = 0.20
half_margin = 0.25
theta_max   = 5 degrees
local_t     = [1, 2, 3, 4, 5]
warmup      = 10 epochs
formal      = 60 epochs
diffusion active steps = 2
```

This configuration corresponds to **M25**. It was selected on Baby in Round32 and then transferred unchanged to Sports and Electronics in Round33.

## Overall Framework

```text
Multimodal recommender backbone
        |
        v
Collaborative / multimodal candidate representation
        |
        +------------------------------+
        |                              |
        v                              v
   Full CoLiftRec              Boundary candidate mining
 generic-semantic              + user-conditioned diffusion
 debiasing                     + bounded preference refinement
        |                              |
        +--------------+---------------+
                       v
            improved ranking model
                       |
                       v
              Validation / Test
```

Diffusion is used as a **training-time auxiliary module**. At Validation/Test inference time, the trained recommender and Full CoLiftRec are evaluated normally:

```text
diffusion_inference_calls = 0
```

Therefore the current Diffusion module introduces no iterative diffusion sampling cost during final recommendation inference.

## Environment

The upstream MSCA repository was released for Python 3.8.10 and PyTorch 1.11.0+cu113. The current RTX 5090 research server has also been validated with Python 3.9 and PyTorch 2.7.0+cu128.

```bash
pip install -r requirements.txt
```

See `docs/REPRODUCTION.md` and `PROVENANCE.md` for additional repository provenance.

## Dataset

Current datasets:

- Amazon Baby
- Amazon Sports
- Amazon Electronics (`elec` internally)

Place local interaction, text, visual, and metadata assets under `data/<dataset>/`. Large datasets, checkpoints, caches, and runtime outputs are intentionally excluded from Git.

## Current Experimental Protocol

### Baby

Baby uses the Round32 parameter-sweep implementation. The current seed1000 configuration is **M25**:

```bash
python diffusion_experiments/round32_round26_parameter_sweep/run_round32.py \
  formal --seed 1000 --variant M25
```

The corresponding evidence is stored in:

```text
diffusion_experiments/round32_round26_parameter_sweep/evidence/
```

In particular:

```text
ROUND32_B0_REFERENCE_SEED1000.json
ROUND32_M25_SEED1000.json
```

Baby Test was used as a development benchmark during the Round32 parameter comparison, so it should not be described as a completely untouched final Test split after Round31.

### Sports and Electronics

Round33 transfers the Baby seed1000 M25 configuration to Sports and Electronics **without domain-specific parameter search**:

```bash
python diffusion_experiments/round33_seed1000_cross_domain/run_round33.py --dataset sports
python diffusion_experiments/round33_seed1000_cross_domain/run_round33.py --dataset elec
```

For Sports and Electronics:

```text
seed = 1000
lambda_HN = 0.20
half_margin = 0.25
theta_max = 5 degrees
```

The configuration is frozen before cross-domain evaluation. Checkpoints are selected using Validation Full-CoLiftRec R@20, followed by one Test evaluation. Test is not used for Sports/Electronics parameter tuning.

Round33 evidence is stored in:

```text
diffusion_experiments/round33_seed1000_cross_domain/evidence/
```

## Method Components

### CoLiftRec

For modality `m`, CoLiftRec estimates an item-generic background and constructs a personalized semantic lift:

```text
L_ui^m = z_ui^m - lambda_m * mu_i^m
```

After normalization, the final multimodal score combines the backbone score with text, attribute, and visual personalized lifts.

The purpose is to reduce the influence of generic content similarity that is not specific to the current user's preference.

### Preference-Directed Diffusion

For a user, observed positive item, and a difficult candidate near the current ranking boundary, the diffusion module learns a user-conditioned local preference direction.

The candidate is refined only when the direction is compatible with the user's current preference. Refinement is bounded by both:

- a fraction of the current positive-candidate score margin (`half_margin`), and
- a maximum angular trust region (`theta_max`).

The refined candidate is then used in a lightweight auxiliary ranking loss:

```text
L = L_normal + lambda_HN * L_boundary
```

The current three-domain configuration uses:

```text
lambda_HN = 0.20
half_margin = 0.25
theta_max = 5 degrees
```

## Results

All results below use **seed1000**.

The main comparison is:

```text
MSCA
-> MSCA + Full CoLiftRec
-> MSCA + Full CoLiftRec + Preference-Directed Diffusion
```

The reported `U4` improvement is the mean relative improvement over the four primary metrics:

```text
R@10, N@10, R@20, N@20
```

### Three-Dataset Improvement Summary

| Dataset | CoLiftRec vs MSCA (U4) | Diffusion vs CoLiftRec (U4) | Full Method vs MSCA (U4) | Diffusion Primary | Diffusion Overall |
|---|---:|---:|---:|---:|---:|
| Baby | **+4.48%** | **+2.40%** | **+6.99%** | 4/4 | 5/6 |
| Sports | **+5.50%** | **+3.20%** | **+8.87%** | 4/4 | 6/6 |
| Electronics | **+3.01%** | **+4.00%** | **+7.13%** | 4/4 | 6/6 |

Across all three datasets, Preference-Directed Diffusion improves **all four primary Test metrics** over MSCA + CoLiftRec.

### Baby — Test

| Method | R@10 | N@10 | R@20 | N@20 | R@50 | N@50 |
|---|---:|---:|---:|---:|---:|---:|
| MSCA | 0.067720 | 0.037031 | 0.103784 | 0.046318 | 0.174483 | 0.060638 |
| MSCA + CoLiftRec | 0.071226 | 0.038945 | 0.107293 | 0.048269 | 0.179539 | 0.062955 |
| MSCA + CoLiftRec + Diffusion | **0.072726** | **0.039750** | **0.110479** | **0.049464** | 0.179375 | **0.063351** |

Diffusion relative to CoLiftRec on the four primary metrics:

```text
R@10  +2.11%
N@10  +2.07%
R@20  +2.97%
N@20  +2.47%
U4    +2.40%
```

Paired user bootstrap for Diffusion vs CoLiftRec:

```text
95% CI = [+0.99%, +3.95%]
positive replicate fraction = 1.000
```

Evidence:

```text
diffusion_experiments/round32_round26_parameter_sweep/evidence/ROUND32_M25_SEED1000.json
```

### Sports — Test

| Method | R@10 | N@10 | R@20 | N@20 | R@50 | N@50 |
|---|---:|---:|---:|---:|---:|---:|
| MSCA | 0.078633 | 0.042836 | 0.117032 | 0.052734 | 0.189671 | 0.067531 |
| MSCA + CoLiftRec | 0.083048 | 0.044891 | 0.124388 | 0.055527 | 0.195963 | 0.070075 |
| MSCA + CoLiftRec + Diffusion | **0.086167** | **0.046832** | **0.126438** | **0.057239** | **0.197375** | **0.071669** |

Diffusion relative to CoLiftRec:

```text
R@10  +3.76%
N@10  +4.32%
R@20  +1.65%
N@20  +3.08%
U4    +3.20%
```

Paired user bootstrap:

```text
95% CI = [+1.87%, +4.49%]
positive replicate fraction = 1.000
```

Evidence:

```text
diffusion_experiments/round33_seed1000_cross_domain/evidence/ROUND33_SPORTS_TEST.json
```

### Electronics — Test

| Method | R@10 | N@10 | R@20 | N@20 | R@50 | N@50 |
|---|---:|---:|---:|---:|---:|---:|
| MSCA | 0.049395 | 0.027998 | 0.072552 | 0.033973 | 0.116521 | 0.042912 |
| MSCA + CoLiftRec | 0.050878 | 0.028767 | 0.074946 | 0.034986 | 0.119535 | 0.044051 |
| MSCA + CoLiftRec + Diffusion | **0.053055** | **0.029549** | **0.078976** | **0.036249** | **0.123687** | **0.045357** |

Diffusion relative to CoLiftRec:

```text
R@10  +4.28%
N@10  +2.72%
R@20  +5.38%
N@20  +3.61%
U4    +4.00%
```

Paired user bootstrap:

```text
95% CI = [+3.38%, +4.71%]
positive replicate fraction = 1.000
```

Evidence:

```text
diffusion_experiments/round33_seed1000_cross_domain/evidence/ROUND33_ELEC_TEST.json
```

## Result Interpretation

The current seed1000 results show a consistent two-stage gain:

1. **CoLiftRec** improves the backbone on all three datasets by removing generic semantic background and emphasizing personalized multimodal lift.
2. **Preference-Directed Diffusion** provides a further positive improvement over CoLiftRec on all four primary Test metrics for Baby, Sports, and Electronics.

The Diffusion increment is not produced by Test-time reranking or iterative sampling. The preference-directed diffusion module operates during training, while final Test inference uses the trained recommender with Full CoLiftRec and records:

```text
diffusion_inference_calls = 0
```

## Legacy Unified CLI

The repository also retains the earlier unified MMRec-style CLI:

```bash
python src/main.py -m MSCA -d baby
python src/main.py -m MSCA -d baby --stage coliftrec
python src/main.py -m MSCA -d baby --stage full
```

The latest seed1000 Preference-Directed Diffusion results reported above are produced by the Round32/33 experiment modules and their committed evidence, rather than the earlier condition-adaptive diffusion configuration in the legacy `--stage full` pipeline.

## Repository Structure

```text
Second-paper/
├── README.md
├── LICENSE
├── PROVENANCE.md
├── requirements.txt
├── data/
├── images/
├── src/
├── scripts/
├── diffusion_experiments/
│   ├── round32_round26_parameter_sweep/
│   └── round33_seed1000_cross_domain/
├── experiments/
├── tests/
└── docs/
```

Development and diagnostic experiments are retained for traceability. Current seed1000 cross-domain evidence is under `diffusion_experiments/round32_round26_parameter_sweep/evidence/` and `diffusion_experiments/round33_seed1000_cross_domain/evidence/`.

## Acknowledgement

This repository is built upon **MSCA** and the **MMRec** framework. We thank the original authors and maintainers for their work.

- MSCA upstream: `recomall/MSCA`
- Frozen upstream commit: `48455de8efa943e16d49db665e7f2fcb0c6c5e17`
- MMRec: `enoche/MMRec`

The upstream GNU GPL v3.0 license is retained in `LICENSE`.

## Citation

The citation entry for this work will be added after the paper metadata is finalized. For the MSCA backbone, please cite the original MSCA paper described by the upstream project.
