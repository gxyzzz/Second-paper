# Second-paper Terminal Commands

## Project and environment

```bash
cd /home/gxy/code/Second-paper
conda activate gume
nvidia-smi
```

## Formal full pipeline

```bash
python src/main.py -m MSCA -d baby --stage full --gpu-id 0
python src/main.py -m MSCA -d sports --stage full --gpu-id 0
python src/main.py -m MSCA -d elec --stage full --gpu-id 0
```

Each command uses seed1000, frozen dataset-specific CoLiftRec parameters, and Preference-Directed Diffusion Boundary Refinement with `lambda_HN=0.20`, `half_margin=0.25`, and `theta_max=5` degrees.

## Individual stages

### Baby

```bash
python src/main.py -m MSCA -d baby --stage msca --gpu-id 0
python src/main.py -m MSCA -d baby --stage coliftrec --gpu-id 0
python src/main.py -m MSCA -d baby --stage full --gpu-id 0
```

### Sports

```bash
python src/main.py -m MSCA -d sports --stage msca --gpu-id 0
python src/main.py -m MSCA -d sports --stage coliftrec --gpu-id 0
python src/main.py -m MSCA -d sports --stage full --gpu-id 0
```

### Electronics

```bash
python src/main.py -m MSCA -d elec --stage msca --gpu-id 0
python src/main.py -m MSCA -d elec --stage coliftrec --gpu-id 0
python src/main.py -m MSCA -d elec --stage full --gpu-id 0
```

## Smoke

```bash
python src/main.py -m MSCA -d baby --stage full --gpu-id 0 --smoke
```

Smoke exercises warmup plus active boundary refinement with tiny batches and does not access formal Test.

## Dry-run

```bash
python src/main.py -m MSCA -d baby --stage full --gpu-id 0 --dry-run
python src/main.py -m MSCA -d sports --stage full --gpu-id 0 --dry-run
python src/main.py -m MSCA -d elec --stage full --gpu-id 0 --dry-run
```

Stage-specific dry-runs:

```bash
python src/main.py -m MSCA -d baby --stage msca --dry-run
python src/main.py -m MSCA -d baby --stage coliftrec --dry-run
python src/main.py -m MSCA -d baby --stage full --dry-run
```

## Parity and tests

```bash
env PYTHONPATH=src:. python scripts/audit_main_runtime_parity.py
env PYTHONPATH=src:. python -m unittest -v tests.test_preference_diffusion
```

## Background formal runs

```bash
nohup python -u src/main.py -m MSCA -d baby --stage full --gpu-id 0 > log/nohup-baby-full.out 2>&1 &
nohup python -u src/main.py -m MSCA -d sports --stage full --gpu-id 0 > log/nohup-sports-full.out 2>&1 &
nohup python -u src/main.py -m MSCA -d elec --stage full --gpu-id 0 > log/nohup-elec-full.out 2>&1 &
```

## Logs

```bash
ls -lt log | head
tail -f "$(ls -t log/*.log | head -1)"
```

Current full-pipeline log names use:

```text
MSCA-CoLiftRec-PreferenceDiffusion-<dataset>-<timestamp>.log
```

The end of a formal full log reports the three Test metric blocks, relative improvements, and `Diffusion inference calls = 0`.

## Process inspection

```bash
ps -ef | grep "src/main.py" | grep -v grep
nvidia-smi
```

## Git inspection

```bash
git status
git log --oneline -10
git rev-parse HEAD
git ls-remote origin refs/heads/main
```
