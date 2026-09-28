# Source and Data Provenance

## Upstream source

- Upstream repository: `https://github.com/recomall/MSCA`
- Frozen source commit: `48455de8efa943e16d49db665e7f2fcb0c6c5e17`
- Import method: tracked files were exported from the Git object at the frozen commit (`git archive`), not copied from a mutable sibling working tree.
- The upstream `LICENSE` is retained unchanged.
- The repository keeps the upstream MMRec/MSCA structure and vanilla MSCA as an independently runnable backbone.

## Publication pipeline provenance

The publication-facing pipeline is:

```text
MSCA
-> Full CoLiftRec
-> Condition-Adaptive Diffusion Semantic Purification
```

The unified publication backbone seed is `999` for Baby, Sports, and Electronics.
Frozen method parameters are stored only in:

```text
src/configs/model/CoLiftRecDiffusion.yaml
```

MSCA backbone parameters remain under `src/configs/model/MSCA.yaml` and `src/configs/model/MSCA/`.

The publication-facing frozen evidence is under:

```text
docs/evidence/final/
```

The pre-refactor repository snapshot is recorded in:

```text
docs/evidence/repository_refactor_pre_snapshot.json
```

## Baby robustness history

The dedicated Baby multiseed robustness experiment pre-registered seed1000 as its canonical backbone using MSCA Validation R@20 before opening the four-seed Test evaluation.

That historical decision is preserved and is not rewritten.

For publication tables, seed999 is used as the unified seed across Baby, Sports, and Electronics. The publication-seed policy is therefore separate from the dedicated multiseed robustness experiment and is not selected from its Test outcomes.

## Canonical local data assets

Dataset files are local inputs and are not committed. The publication runtime expects per-dataset assets under `data/<dataset>/`, including:

- interaction file
- text features
- visual features
- metadata text cache

For Baby, the audited canonical assets include:

- `data/baby/baby.inter`
- `data/baby/text_feat.npy`, shape `(7050, 384)`, dtype `float32`
- `data/baby/image_feat.npy`, shape `(7050, 4096)`, dtype `float64`

Baby SHA256 values recorded during the original data audit:

- `text_feat.npy`: `6667f2ad655c9ecc97cb3383f58988864ef51ec0b39c158b15986c66769f2dc4`
- `image_feat.npy`: `36c3be592b98506189a7d5de71b21577cf626f0293b539d861534673b3e9fd70`
- `baby.inter`: `9ef5de9a05949d973928cca113aacba12067ff8f104e310ef28ea42db00c17b1`

## Runtime artifact policy

The final runtime must not depend on historical checkpoints, old Top-100 caches, old purified features, archived development scripts, or sibling repositories.

Generated assets remain local and untracked, including:

- `runs/`
- checkpoints (`*.pt`, `*.pth`)
- cached rankings
- large `*.npy` / `*.npz`
- purified feature arrays

Development scripts are preserved under `experiments/archive/` for traceability only. Formal runtime code under `src/` and `scripts/` must not import from or read that archive.

## Environment compatibility

The frozen upstream source was authored for Python 3.8.10, PyTorch 1.11.0+cu113, and NumPy 1.21.5.

The current RTX 5090 research server has also been validated with Python 3.9.25, PyTorch 2.7.0+cu128, and NumPy 2.0.2. NumPy 2.x removes the deprecated `np.float` alias, so `src/utils/metrics.py` carries only the semantics-preserving `np.float` to builtin `float` compatibility patch already validated in the server workspace. No metric definition or scientific formula is changed.
