# Source and Method Provenance

## Upstream source

- Upstream repository: `https://github.com/recomall/MSCA`
- Frozen upstream commit: `48455de8efa943e16d49db665e7f2fcb0c6c5e17`
- The upstream `LICENSE` is retained unchanged.
- The repository keeps vanilla MSCA as an independently runnable backbone.

## Current publication pipeline

The current publication-facing pipeline is:

```text
Multimodal Recommender Backbone
-> Full CoLiftRec
-> Preference-Directed Diffusion Boundary Refinement (training only)
-> frozen recommender Test inference
```

All current formal results use `seed = 1000` for Baby, Sports, and Electronics.

Current scientific parameters are frozen in:

```text
src/configs/model/CoLiftRecDiffusion.yaml
```

Dataset-specific MSCA settings remain in:

```text
src/configs/model/MSCA/
```

## Current Diffusion semantics

PDBR is a training-time auxiliary mechanism. It learns user-conditioned local preference directions from BASE / TRUE / SHUFFLED conditions, performs tangent-space refinement around boundary candidates, and applies a bounded auxiliary ranking loss.

Formal Test inference does not sample from or call the Diffusion network:

```text
diffusion_inference_calls = 0
```

## Frozen current configuration

```text
seed          = 1000
lambda_HN     = 0.20
half_margin   = 0.25
theta_max     = 5 degrees
local_t       = [1,2,3,4,5]
warmup        = 10 epochs
formal train  = 60 epochs
active steps  = 2
```

No parameter sweep is part of the current `main` runtime.

## Successful evidence provenance

The current result references originate from the preserved experimental evidence:

```text
Baby:        Round32 M25 seed1000 evidence
Sports:      Round33 frozen seed1000 cross-domain transfer
Electronics: Round33 frozen seed1000 cross-domain transfer
```

The experiment history is preserved for scientific provenance. The formal runtime under `src/` does not import from it.

Machine-readable current parity/result references are copied into:

```text
docs/evidence/current/MAIN_RUNTIME_PARITY_AUDIT.json
docs/evidence/current/THREE_DOMAIN_SEED1000_RESULTS.json
```

## Current three-domain summary

| Dataset | CoLiftRec vs MSCA | PDBR vs CoLiftRec | Full vs MSCA |
|---|---:|---:|---:|
| Baby | +4.48% | +2.40% | +6.99% |
| Sports | +5.50% | +3.20% | +8.87% |
| Electronics | +3.01% | +4.00% | +7.13% |

PDBR is 4/4 positive on the primary Test metrics for all three datasets.

## Historical preservation policy

Historical experiment branches, evidence, checkpoints, and diagnostic outputs are not rewritten by the main-runtime migration. They remain provenance only. The current method documentation describes the seed1000/PDBR runtime; older development methods are not used by the publication pipeline.

## Runtime artifact policy

Generated assets remain local and untracked, including:

- `runs/`
- checkpoints (`*.pt`, `*.pth`)
- cached rankings
- large feature arrays
- experiment evidence/output directories

The current runtime uses only formal code/configuration under `src/` plus the local dataset assets.

## Environment compatibility

The frozen upstream source was authored for Python 3.8.10 and an older PyTorch/CUDA stack. The current RTX 5090 research environment has been validated with the `gume` environment using Python 3.9 and PyTorch 2.7.0+cu128.
