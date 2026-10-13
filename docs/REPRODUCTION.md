# Reproduction

## Current frozen protocol

All current formal experiments use seed1000 and the frozen configuration in `src/configs/model/CoLiftRecDiffusion.yaml`.

```text
CoLiftRec
+ Preference-Directed Diffusion Boundary Refinement
lambda_HN     = 0.20
half_margin   = 0.25
theta_max     = 5 degrees
local_t       = [1,2,3,4,5]
warmup        = 10 epochs
formal train  = 60 epochs
active steps  = 2
Test diffusion calls = 0
```

## Primary CLI

```bash
cd /home/gxy/code/Second-paper
conda activate gume

python src/main.py -m MSCA -d baby --stage full
python src/main.py -m MSCA -d sports --stage full
python src/main.py -m MSCA -d elec --stage full
```

## Stage behavior

### `--stage msca`

Train the seed1000 backbone and select the checkpoint using backbone Validation R@20, then freeze and evaluate Test once.

```bash
python src/main.py -m MSCA -d baby --stage msca
```

### `--stage coliftrec`

Train the baseline seed1000 branch, select using Full CoLiftRec Validation R@20, then evaluate both backbone and Full CoLiftRec on Test.

```bash
python src/main.py -m MSCA -d baby --stage coliftrec
```

### `--stage full`

Create one shared seed1000 initialization and train two fair branches:

```text
baseline branch:
  normal MSCA training
  -> best Full CoLiftRec Validation R@20

full branch:
  same initialization / same normal training plan
  + training-time PDBR boundary supervision
  -> best Full CoLiftRec Validation R@20
```

After both checkpoints are frozen, one Test evaluation reports:

```text
MSCA
MSCA + Full CoLiftRec
MSCA + Full CoLiftRec + PDBR
```

The Diffusion network is not called during Test.

## Validation and Test discipline

Formal training budget is 60 epochs. The final checkpoint is not forced to epoch 60: Validation Full CoLiftRec R@20 selects the best checkpoint. Test never selects the epoch, checkpoint, seed, candidate boundary, CoLiftRec parameter, or PDBR parameter.

## Smoke

```bash
python src/main.py -m MSCA -d baby --stage full --smoke
```

Smoke is engineering-only. It runs a tiny baseline path plus epoch0 warmup and epoch10 active boundary refinement. It performs no formal Test and writes `TEST_ACCESSED = false`.

## Dry-run

```bash
python src/main.py -m MSCA -d baby --stage full --dry-run
python src/main.py -m MSCA -d sports --stage full --dry-run
python src/main.py -m MSCA -d elec --stage full --dry-run
```

Dry-run resolves dataset, seed, backbone, CoLiftRec parameters, PDBR parameters, GPU, and paths without training or evaluation.

## Logs and run directories

Human-readable logs are written under `log/`, for example:

```text
MSCA-CoLiftRec-PreferenceDiffusion-baby-<timestamp>.log
```

Formal full logs include the three Test metric blocks, stage-to-stage relative improvements, selected epochs, and:

```text
Diffusion inference calls = 0
```

Machine-readable runtime artifacts are written under:

```text
runs/publication/<dataset>/<run_id>/
```

A formal run evaluates Test only after Validation selection and records one Test completion marker.

## Deterministic parity audit

Before expensive formal retraining, run:

```bash
env PYTHONPATH=src:. python scripts/audit_main_runtime_parity.py
```

The audit checks:

- seed1000 policy;
- dataset-specific MSCA overrides;
- frozen CoLiftRec parameters;
- frozen M25/PDBR parameters;
- ranking boundary bands and curriculum;
- Test-time Diffusion disabled;
- no `src/` dependency on experiment implementations.

Current audit output:

```text
docs/evidence/current/MAIN_RUNTIME_PARITY_AUDIT.json
```

Current three-domain result reference:

```text
docs/evidence/current/THREE_DOMAIN_SEED1000_RESULTS.json
```

## Unit tests

The repository uses Python standard-library `unittest`, so no extra testing package is required:

```bash
env PYTHONPATH=src:. python -m unittest -v tests.test_preference_diffusion
```

The tests cover tangent orthogonality, BASE/TRUE/SHUFFLED branches, the 25% margin target, 5-degree angular cap, target safety, frozen M25 config, and the no-Diffusion Test evaluator.

## Current result references

| Dataset | CoLiftRec vs MSCA | PDBR vs CoLiftRec | Full vs MSCA |
|---|---:|---:|---:|
| Baby | +4.48% | +2.40% | +6.99% |
| Sports | +5.50% | +3.20% | +8.87% |
| Electronics | +3.01% | +4.00% | +7.13% |

These are the frozen successful seed1000 references used for main-runtime parity. This migration does not initiate a new parameter search.
