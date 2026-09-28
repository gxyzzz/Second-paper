# Reproduction

The publication configuration is fixed in src/configs/model/CoLiftRecDiffusion.yaml.
MSCA keeps its upstream configuration in src/configs/model/MSCA.yaml and src/configs/model/MSCA/.

## One-command reproduction

    python scripts/reproduce.py --dataset baby --gpu 0
    python scripts/reproduce.py --dataset sports --gpu 0
    python scripts/reproduce.py --dataset elec --gpu 0

All three datasets can be run sequentially with:

    bash scripts/reproduce_all.sh 0

## Stage-by-stage reproduction

    python scripts/reproduce.py --dataset baby --stage msca --gpu 0
    python scripts/reproduce.py --dataset baby --stage coliftrec --gpu 0
    python scripts/reproduce.py --dataset baby --stage diffusion --gpu 0
    python scripts/reproduce.py --dataset baby --stage validation --gpu 0
    python scripts/reproduce.py --dataset baby --stage test --gpu 0

The same interface applies to sports and elec.

The default runtime workspace is runs/reproduction/<dataset>/.
Runtime checkpoints, cached rankings, purified arrays, and other generated
artifacts are not tracked by Git.

## Pipeline

    MSCA
      -> current-run MSCA embeddings and Top100 candidates
      -> Full CoLiftRec
      -> frozen Condition-Adaptive Diffusion Semantic Purification
      -> Validation evaluation
      -> Test evaluation

The publication backbone seed is 999 for Baby, Sports, and Electronics.
The Test stage is evaluation-only and must not be used for parameter or seed selection.
