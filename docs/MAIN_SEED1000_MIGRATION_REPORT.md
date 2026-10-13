# Main seed1000 Migration Report

## 1. Objective

Migrate the already-validated seed1000 + CoLiftRec + Preference-Directed Diffusion Boundary Refinement method into the official `main` runtime without changing the scientific method or reopening parameter search.

## 2. Previous main runtime

Before migration, `main` used an older Diffusion runtime centered on Test-time multimodal feature editing and an older publication-seed policy. That path is no longer active in the current runner/configuration.

A safety branch was created before modification:

```text
backup/main-before-seed1000-preference-diffusion
```

Pre-migration `main` HEAD:

```text
e2233a784d2a640460142366a76979f093df1314
```

## 3. New current method

```text
Multimodal Recommender Backbone
-> Full CoLiftRec
-> Preference-Directed Diffusion Boundary Refinement during training
-> Validation-selected final recommender
-> frozen Test inference with zero Diffusion calls
```

PDBR learns a user-conditioned local preference direction from BASE / TRUE / SHUFFLED branches, applies tangent-space candidate refinement, and uses a bounded boundary-ranking auxiliary loss.

## 4. seed1000 policy

All current formal runs and ablations use:

```text
seed = 1000
```

for Baby, Sports, and Electronics.

## 5. Frozen M25 configuration

```text
lambda_HN       = 0.20
half_margin     = 0.25
theta_max_deg   = 5.0
local_t         = [1,2,3,4,5]
warmup_epochs   = 10
formal_epochs   = 60
active_steps    = 2
inference       = disabled
```

No new sweep was performed during migration.

## 6. Files replaced / rewired

- `src/configs/overall.yaml`: current seed switched to 1000.
- `src/configs/model/CoLiftRecDiffusion.yaml`: current CoLiftRec + PDBR configuration.
- `src/pipelines/dataset_config.py`: resolves only the current formal method configuration.
- `src/pipelines/runner.py`: official CLI now uses the seed1000/PDBR publication pipeline.
- `src/main.py`: CLI descriptions updated for the current stage semantics.

## 7. Files newly added

- `src/modules/preference_diffusion.py`
- `src/pipelines/preference_pipeline.py`
- `scripts/audit_main_runtime_parity.py`
- `tests/test_preference_diffusion.py`
- `docs/evidence/current/MAIN_RUNTIME_PARITY_AUDIT.json`
- `docs/evidence/current/THREE_DOMAIN_SEED1000_RESULTS.json`
- this report

## 8. Files deprecated from active runtime

Legacy Diffusion implementation files remain in the repository where already tracked for provenance, but the current `runner.py` does not import or execute them. The current runtime contains no import of experiment implementation code.

## 9. Experiment-history untouched audit

The server retains the historical `diffusion_experiments/` directory separately from tracked `main`. Before migration its lightweight manifest was recorded as:

```text
files  = 1908
bytes  = 44100697006
hash   = 809817774df2d6ae2ba4ba51d2fcebb1b5e0ed07c25d8ce5545f69aac676f249
```

The final closing audit must reproduce the same path/size/mtime manifest. No migration code writes into this directory.

## 10. Historical branches

The historical experiment branches are preserved and were neither rebased nor force-pushed. The successful evidence sources remain the seed1000 Baby M25 experiment and the frozen Sports/Electronics cross-domain transfer experiment.

## 11. CLI commands

```bash
python src/main.py -m MSCA -d baby --stage full
python src/main.py -m MSCA -d sports --stage full
python src/main.py -m MSCA -d elec --stage full

python src/main.py -m MSCA -d baby --stage msca
python src/main.py -m MSCA -d baby --stage coliftrec
python src/main.py -m MSCA -d baby --stage full --smoke
python src/main.py -m MSCA -d baby --stage full --dry-run
```

## 12. Dry-run results

Three full-pipeline dry-runs passed for Baby, Sports, and Electronics. They resolved the expected seed1000 policy, dataset-specific CoLiftRec parameters, PDBR M25 parameters, GPU, and data paths. `src/` had no `diffusion_experiments` dependency.

## 13. Smoke result

Baby full smoke passed. It exercised:

- baseline training path;
- Diffusion reconstruction warmup;
- epoch10 active boundary refinement;
- BASE / TRUE / SHUFFLED mechanism;
- bounded calibration.

Observed mechanism checks included:

```text
P(DeltaA > 0) = 0.593017578125
angle_bound_pass = true
target_never_exceeds_positive = true
TEST_ACCESSED = false
```

No Test marker was created by smoke.

## 14. Parity audit

`docs/evidence/current/MAIN_RUNTIME_PARITY_AUDIT.json` records PASS for:

- seed1000;
- three-domain M25 configuration;
- three-domain CoLiftRec parameters;
- three-domain MSCA overrides;
- boundary bands and curriculum;
- zero Diffusion calls in formal Test evaluator;
- no `src/` dependency on experiment implementations.

Unit tests also cover tangent orthogonality, branch semantics, 25% margin, 5-degree cap, target safety, and Test-time Diffusion disablement.

## 15. Three-domain current results

| Dataset | CoLiftRec vs MSCA | PDBR vs CoLiftRec | Full vs MSCA |
|---|---:|---:|---:|
| Baby | +4.48% | +2.40% | +6.99% |
| Sports | +5.50% | +3.20% | +8.87% |
| Electronics | +3.01% | +4.00% | +7.13% |

PDBR is 4/4 positive on the four primary Test metrics for all three datasets.

## 16. Git commits

Migration is intentionally split into focused commits:

```text
b29765e  refactor: add preference-directed diffusion runtime
4f45bc7  config: switch formal pipeline to seed1000 M25
```

Documentation and final test/audit commits are added after their respective gates pass.

## 17. Final main HEAD

The final closing gate requires:

```text
local main == origin/main
working tree clean
```

The exact final commit hash is verified after the closing push; the migration report itself is versioned within that final main history.
