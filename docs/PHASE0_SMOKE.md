# Phase 0 / Phase 1 Preparation Smoke Evidence

Date: 2026-09-26
Server: NJUST
Repository: /home/gxy/code/Second-paper
Frozen upstream: recomall/MSCA @ 48455de8efa943e16d49db665e7f2fcb0c6c5e17

## Scope

This smoke validates only the vanilla MSCA repository bootstrap and the ability to start Baby training. It does not implement or run CoLiftRec or Diffusion, and it is not a formal training run.

## Environment

- GPU: NVIDIA GeForce RTX 5090 (GPU 0)
- Existing server environment: /home/gxy/miniconda3/envs/msca
- Python: 3.9.25
- Torch: 2.7.0+cu128
- NumPy: 2.0.2
- Frozen upstream requirements remain unchanged and document Python 3.8.10 / torch 1.11.0+cu113 / NumPy 1.21.5.

No historical checkpoint was loaded.

## Static audit

- python -m compileall -q src: PASS
- imports models.msca, utils.quick_start, utils.topk_evaluator: PASS
- local Baby inputs are canonical interaction/Text/Visual inputs only.
- historical image_adj_10_True.pt / text_adj_10_True.pt were not linked or copied into the new repository before smoke.

## Baby dataset exactness

- users: 19445
- items: 7050
- interactions: 160792
- TRAIN (x_label=0): 118551
- Validation (x_label=1): 20559
- Test (x_label=2): 21682
- Text feature: (7050, 384), float32
- Visual feature: (7050, 4096), float64
- seed: 999

## Training-start smoke

Command:

    cd /home/gxy/code/Second-paper/src
    PYTHONUNBUFFERED=1 /home/gxy/miniconda3/envs/msca/bin/python main.py -m MSCA -d baby

Observed:

- MSCA instantiated successfully.
- trainable parameters: 33,579,072.
- epoch 0 completed in 3.91 s.
- epoch 0 training loss: 777.1020 (finite).
- therefore the minimum "training can start" smoke criterion is PASS.

The first attempt then reached Validation and exposed NumPy 2.x incompatibility in upstream np.float usage. A semantics-preserving compatibility patch (np.float -> builtin float) was applied in src/utils/metrics.py; direct NDCG/MAP metric smoke under NumPy 2.0.2 passes.

## Phase 1 blocker / protocol observation

The frozen upstream trainer.fit() evaluates Test immediately after every Validation epoch. That upstream behavior conflicts with the new repository's Test discipline. No formal Phase 1 training should be launched until Test evaluation is separated from model selection.

This Phase 0 task does not refactor that training protocol. No Test metric from this smoke is used for tuning or selection.

## Result

PHASE0_BOOTSTRAP_SMOKE = PASS
PHASE1_FORMAL_TRAINING = NOT_STARTED
COLIFTREC = NOT_IMPLEMENTED
DIFFUSION = NOT_IMPLEMENTED
