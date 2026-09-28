# Reproduction

The normal user workflow is intentionally MMRec/MSCA-like:

```text
edit YAML
-> python src/main.py ...
-> live terminal + one project-level log
-> Validation-based selection
-> one frozen Test evaluation
-> paper-ready final summary
```

Frozen method parameters remain in `src/configs/model/CoLiftRecDiffusion.yaml`. MSCA keeps its upstream parameters in `src/configs/model/MSCA.yaml` and `src/configs/model/MSCA/`.

## Primary CLI

```bash
# MSCA
python src/main.py -m MSCA -d baby

# MSCA + CoLiftRec
python src/main.py -m MSCA -d baby --stage coliftrec

# MSCA + CoLiftRec + Diffusion
python src/main.py -m MSCA -d baby --stage full
```

The same interface applies to `sports` and `elec`.

A non-smoke, non-dry formal run is:

```text
TRAIN
-> Validation selection
-> freeze current-run checkpoint/config
-> Test exactly once
-> final log summary
```

Test is evaluation-only. It never controls early stopping, best epoch, checkpoint selection, seed, `lambda`, `alpha`, `beta`, `rho`, guidance, or any other scientific choice.

## Stage behavior

### `--stage msca`

```text
MSCA training
-> Validation every eval_step
-> best Validation checkpoint
-> best epoch read from that actual checkpoint
-> current-run asset export
-> MSCA Test
```

### `--stage coliftrec`

```text
MSCA Validation-selected checkpoint
-> current-run assets
-> frozen Full CoLiftRec
-> CoLiftRec Validation
-> one Test evaluation
```

The final log contains both MSCA and MSCA + CoLiftRec Validation/Test metrics and the CoLiftRec Test delta versus MSCA.

### `--stage full`

```text
MSCA Validation-selected checkpoint
-> current-run assets
-> Full CoLiftRec
-> CoLiftRec Validation
-> frozen Diffusion training protocol
-> frozen Diffusion checkpoint selection
-> purification
-> Full Validation
-> one Test evaluation
```

The single Test evaluator reports MSCA, MSCA + CoLiftRec, and MSCA + CoLiftRec + Diffusion on the same users, same Test split, same current-run MSCA checkpoint, and same frozen candidate/scoring protocol.

## Smoke and dry-run

Engineering smoke never accesses Test:

```bash
python src/main.py -m MSCA -d baby --stage full --smoke
```

Expected manifest state:

```text
TEST_ACCESSED = false
TEST_RUN_COUNT = 0
TEST_RUN_COMPLETED = false
```

Dry-run performs only configuration/path resolution:

```bash
python src/main.py -m MSCA -d baby --stage full --dry-run
```

It performs no training, Validation, or Test.

## Project-level log

Every invocation writes one main log to:

```text
Second-paper/log/
```

Typical names:

```text
MSCA-baby-<timestamp>.log
MSCA-CoLiftRec-baby-<timestamp>.log
MSCA-CoLiftRec-Diffusion-baby-<timestamp>.log
```

The same messages are shown live in the terminal.

At the beginning, the log records server, working directory, dataset, stage, GPU, resolved MSCA configuration, CoLiftRec parameters, and Diffusion parameters.

At the end, a formal log contains a `FINAL EXPERIMENT RESULT` block with `Best Epoch`, `Best Validation Score`, explicit `VALIDATION RESULT` and `TEST RESULT` sections, and for Full runs the Diffusion selected-epoch/checkpoint-selection metadata plus all paper-ready metrics.

Metric values are printed to six decimals while machine-readable JSON retains full float precision.

For normal usage, inspect the log file under `./log/`.

## Internal run directory

The internal workspace remains:

```text
runs/reproduction/<dataset>/<run_id>/
├── run_manifest.json
├── resolved_config.yaml
├── msca/
├── coliftrec/
├── diffusion/
├── validation/
├── test/
└── summary.json
```

These are machine-readable reproducibility artifacts. They are not the main human-facing result location.

## Current-run binding and Test guard

MSCA checkpoints generated during normal training are stored under the current run directory. Later stages use the checkpoint bound in `run_manifest.json`; they do not scan historical runs by modification time.

After a successful formal Test:

```text
TEST_RUN_COUNT = 1
TEST_RUN_COMPLETED = true
```

is recorded. A completed formal run directory is protected from later formal or smoke overwrite; a second formal/smoke invocation is rejected before training/Test begins. Dry-run remains allowed because it performs no evaluation.

A fresh ordinary invocation receives a new run ID and therefore represents a new experiment with its own single Test evaluation and its own log.

## Best epoch and Diffusion selection metadata

MSCA `best_epoch` and `best_valid_score` are read from the actual Validation-selected checkpoint state.

For Baby/Sports, frozen Diffusion selection is `checkpoint_selection = final_epoch`, so the formal log records the actual final selected epoch.

For Electronics, frozen Diffusion selection is `checkpoint_selection = best_monitor`, so the formal log records the actual best monitor epoch, best monitor objective, and patience metadata from training evidence.

## Advanced helpers

`scripts/reproduce.py` remains an automation/advanced helper and reuses the same main runner for shared stages.

`src/test.py` remains available for independent evaluation of an existing checkpoint/run.

Ordinary complete experiments do not require either helper.

## Publication parity

The automatic formal Test path in `src/pipelines/runner.py` calls the existing frozen `pipelines.publication_eval.evaluate_test` evaluator rather than maintaining a second Test implementation.

Frozen three-domain parity evidence remains under `docs/evidence/final/ranking_parity_audit.json` and requires, for Baby, Sports, and Electronics:

```text
metric_max_abs_diff = 0
ranked_items_exact = true
```

The publication seed remains `999` for all three domains, including Baby.

See `docs/evidence/final/main_formal_test_log_parity_audit.json` for the final CLI/Test/log integration audit.
