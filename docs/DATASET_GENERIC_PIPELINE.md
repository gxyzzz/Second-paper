# Dataset-generic Second-paper pipeline

The final implementation uses one Python pipeline across Baby, Sports and Electronics (elec).

Dataset-specific scientific parameters and asset paths live under:

- src/configs/second_paper/baby.yaml
- src/configs/second_paper/sports.yaml
- src/configs/second_paper/elec.yaml

Generic entry points:

- python -m pipelines.coliftrec --dataset <id> --assets <dir> --out <dir>
- python -m pipelines.evaluate_coliftrec --dataset <id> --assets <dir> --out <dir>
- python -m pipelines.diffusion_assets --dataset <id> --msca-assets <dir> --out <dir>
- python scripts/diffusion_smoke.py --dataset <id> --msca-assets <dir> --out <dir>

MSCA dataset-specific architecture hyperparameters are loaded from src/configs/model/MSCA/<dataset>.yaml.

Baby full regression against the pre-refactor Phase-2 implementation is exact:
candidate IDs, ranking IDs, metrics, score arrays and background arrays all match exactly with maximum absolute difference 0.

DATASET_GENERIC_PIPELINE = PASS
BABY_GENERIC_REGRESSION = PASS
