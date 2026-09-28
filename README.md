# Second-paper

Official reproducibility repository for our multimodal recommendation study built on MSCA / MMRec.
The publication pipeline combines a frozen MSCA backbone with **CoLiftRec** and **Condition-Adaptive Diffusion Semantic Purification**.

## Introduction

The repository keeps MSCA as an independently runnable backbone and adds two post-backbone stages:

1. **CoLiftRec** removes item-generic multimodal relevance from user-item semantic scores and retains the user-specific lift used for candidate reranking.
2. **Diffusion Semantic Purification** edits native text and visual item features under collaborative conditions, then blends the purified features back with fixed, dataset-specific coefficients.

The final publication seed is **999** for Baby, Sports, and Electronics. Publication-facing method parameters are frozen in:

```text
src/configs/model/CoLiftRecDiffusion.yaml
```

## Overall Framework

```text
MSCA
  -> current-run embeddings / Top-100 candidates
  -> Full CoLiftRec
  -> Condition-Adaptive Diffusion Semantic Purification
  -> Validation / Test evaluation
```

Runtime code does not depend on archived development scripts, historical checkpoints, old ranking caches, or legacy purified features.

## Environment

The upstream MSCA repository was released for Python 3.8.10 and PyTorch 1.11.0+cu113. The current RTX 5090 research server has also been validated with Python 3.9 and PyTorch 2.7.0+cu128.

```bash
pip install -r requirements.txt
```

See `docs/REPRODUCTION.md` and `PROVENANCE.md` for details.

## Dataset

Supported publication datasets:

- Baby
- Sports
- Electronics (`elec` internally)

Place local interaction, text, visual, and metadata assets under `data/<dataset>/`. Large datasets, checkpoints, caches, purified arrays, and runtime outputs are intentionally excluded from Git.

Dataset metadata is defined under `src/configs/dataset/`; method parameters are kept separate.

## Quick Start

MSCA remains runnable through the upstream-style entry point:

```bash
cd src
python main.py -m MSCA -d baby
```

Run the complete frozen publication pipeline from the repository root:

```bash
python scripts/reproduce.py --dataset baby --gpu 0
```

The same interface supports `sports` and `elec`.

## Training

Stage-by-stage execution:

```bash
python scripts/reproduce.py --dataset baby --stage msca --gpu 0
python scripts/reproduce.py --dataset baby --stage coliftrec --gpu 0
python scripts/reproduce.py --dataset baby --stage diffusion --gpu 0
python scripts/reproduce.py --dataset baby --stage validation --gpu 0
python scripts/reproduce.py --dataset baby --stage test --gpu 0
```

`--stage all` is the default. Run all three datasets sequentially with:

```bash
bash scripts/reproduce_all.sh 0
```

## Evaluation

The Test stage is evaluation-only. Test data are not used for seed selection, parameter search, checkpoint selection, or post-Test tuning.

## Reproduction

The default runtime workspace is:

```text
runs/reproduction/<dataset>/
```

See `docs/REPRODUCTION.md` for the complete guide. Publication evidence is under `docs/evidence/final/`.

## Configuration

```text
src/configs/dataset/*.yaml
  -> dataset file names / data metadata

src/configs/model/MSCA.yaml and src/configs/model/MSCA/
  -> upstream MSCA backbone parameters

src/configs/model/CoLiftRecDiffusion.yaml
  -> frozen CoLiftRec + Diffusion publication parameters
```

The final method YAML contains only fixed publication values; development search spaces are not part of the public runtime configuration.

## Results

Frozen Test results below are generated from `docs/evidence/final/three_domain_final_results.json`. All three datasets use publication seed **999**.

### Baby

| Method | R@10 | N@10 | R@20 | N@20 | R@50 | N@50 |
|---|---:|---:|---:|---:|---:|---:|
| MSCA | 0.069759 | 0.038059 | 0.103836 | 0.046852 | 0.171405 | 0.060570 |
| MSCA + CoLiftRec | 0.072394 | 0.039797 | 0.108202 | 0.049028 | 0.175978 | 0.062812 |
| MSCA + CoLiftRec + Diffusion | 0.072617 | 0.039913 | 0.108416 | 0.049120 | 0.176445 | 0.062944 |

### Sports

| Method | R@10 | N@10 | R@20 | N@20 | R@50 | N@50 |
|---|---:|---:|---:|---:|---:|---:|
| MSCA | 0.080495 | 0.043785 | 0.120169 | 0.054024 | 0.189231 | 0.068063 |
| MSCA + CoLiftRec | 0.084598 | 0.045690 | 0.124505 | 0.055997 | 0.195191 | 0.070356 |
| MSCA + CoLiftRec + Diffusion | 0.084884 | 0.045984 | 0.125412 | 0.056422 | 0.194709 | 0.070499 |

### Electronics

| Method | R@10 | N@10 | R@20 | N@20 | R@50 | N@50 |
|---|---:|---:|---:|---:|---:|---:|
| MSCA | 0.050197 | 0.028330 | 0.073294 | 0.034295 | 0.116487 | 0.043069 |
| MSCA + CoLiftRec | 0.051303 | 0.028810 | 0.075537 | 0.035069 | 0.119473 | 0.044008 |
| MSCA + CoLiftRec + Diffusion | 0.051502 | 0.029007 | 0.075621 | 0.035255 | 0.119898 | 0.044264 |

For Baby, seed1000 remains preserved as the pre-registered canonical backbone of the dedicated multiseed robustness experiment; it is not used to choose the publication result after Test. The publication table uses the pre-declared unified seed999 policy across all three datasets.

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
│   ├── main.py
│   ├── test.py
│   ├── models/
│   ├── modules/
│   ├── pipelines/
│   └── configs/
├── scripts/
│   ├── reproduce.py
│   └── reproduce_all.sh
├── experiments/
│   └── archive/
├── tests/
└── docs/
    ├── REPRODUCTION.md
    └── evidence/
        ├── final/
        └── archive/
```

`experiments/archive/` preserves development and diagnostic scripts for traceability, but the formal runtime has no dependency on it.

## Acknowledgement

This repository is built upon **MSCA** and the **MMRec** framework. We thank the original authors and maintainers for their work.

- MSCA upstream: `recomall/MSCA`
- Frozen upstream commit: `48455de8efa943e16d49db665e7f2fcb0c6c5e17`
- MMRec: `enoche/MMRec`

The upstream GNU GPL v3.0 license is retained in `LICENSE`.

## Citation

The citation entry for this work will be added after the paper metadata is finalized. For the MSCA backbone, please cite the original MSCA paper described by the upstream project.
