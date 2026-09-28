# Reproduction

The primary user experience follows the original MMRec/MSCA style: modify YAML, run `python src/main.py ...`, receive terminal/file logs, a run manifest, resolved configuration, checkpoints/stage outputs, and Validation results.

Frozen method parameters remain in `src/configs/model/CoLiftRecDiffusion.yaml`; MSCA parameters remain in `src/configs/model/MSCA.yaml` and `src/configs/model/MSCA/`.

## Primary training / Validation CLI

```bash
# MSCA (default stage; backward compatible)
python src/main.py -m MSCA -d baby

# MSCA + Full CoLiftRec
python src/main.py -m MSCA -d baby --stage coliftrec

# Full method
python src/main.py -m MSCA -d baby --stage full
```

`--stage coliftrec` runs MSCA -> current-run assets -> Full CoLiftRec -> Validation. `--stage full` additionally trains the frozen Diffusion purifier, generates purified features, and evaluates the final ranking on Validation.

The same CLI applies to Sports and Electronics:

```bash
python src/main.py -m MSCA -d sports --stage full
python src/main.py -m MSCA -d elec --stage full
```

`src/main.py` never automatically runs Test.

## Legacy source-directory compatibility

```bash
cd src
python main.py -m MSCA -d baby
```

Configuration files are resolved from the source tree instead of the caller's current working directory.

## Run directory and checkpoint binding

Without `--run-dir`, each invocation creates `runs/reproduction/<dataset>/<run_id>/`. A run contains `run_manifest.json`, `resolved_config.yaml`, `msca/`, `coliftrec/`, `diffusion/`, `validation/`, and `summary.json`.

For stage-by-stage execution, reuse one run directory:

```bash
python src/main.py -m MSCA -d baby --stage msca --run-dir runs/reproduction/baby/my_run
python src/main.py -m MSCA -d baby --stage coliftrec --run-dir runs/reproduction/baby/my_run
python src/main.py -m MSCA -d baby --stage full --run-dir runs/reproduction/baby/my_run
```

Later stages load the MSCA checkpoint recorded in `run_manifest.json`; they do not scan checkpoint directories by modification time. `--checkpoint` remains an advanced explicit override.

## Resolved configuration

Every invocation writes `resolved_config.yaml` with dataset, stage, GPU, timestamp/run ID, resolved MSCA configuration, CoLiftRec configuration, Diffusion configuration, and publication seed. The same resolved method parameters are printed at the beginning of the log. Normal parameter changes should be made in YAML rather than Python.

## Logging

Each invocation installs one root logger and writes the same stage messages to the terminal and `runs/logs/*.log`.

Typical log names are `MSCA-baby-<time>.log`, `MSCA-CoLiftRec-baby-<time>.log`, and `MSCA-CoLiftRec-Diffusion-baby-<time>.log`. A full invocation uses one log file and does not install duplicate stage handlers.

## Test policy

The main interface is TRAIN + VALIDATION only. Test remains an explicit action and must not be used for seed selection, parameter search, checkpoint selection, or post-Test tuning.

## Advanced / automation helper

`scripts/reproduce.py` remains available for automation and explicit advanced stages while reusing the same publication runner for shared training stages.

```bash
python scripts/reproduce.py --dataset baby --stage full --gpu 0
python scripts/reproduce.py --dataset baby --stage test --workdir runs/reproduction/baby/my_run --gpu 0
```

## Engineering smoke

`--smoke` is for implementation validation only and is not publication evidence:

```bash
python src/main.py -m MSCA -d baby --stage msca --smoke
python src/main.py -m MSCA -d baby --stage coliftrec --smoke
python src/main.py -m MSCA -d baby --stage full --smoke
```

Diffusion smoke reuses the existing short smoke protocol rather than formal 80-epoch training.

## Publication evidence

Frozen final evidence is under `docs/evidence/final/`, including `config_parity_audit.json`, `ranking_parity_audit.json`, and `three_domain_final_results.json`. Historical evidence remains under `docs/evidence/archive/`; runtime code does not depend on that archive.
