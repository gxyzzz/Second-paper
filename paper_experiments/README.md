# Paper Experiments

Publication evidence that is separate from historical method-development experiments.

- `src/`: frozen publication runtime.
- `paper_experiments/`: paper-only ablation, boundary, and efficiency experiments.
- `diffusion_experiments/`: historical method development; read-only in this round.

Protocol:

- seed = 1000 only;
- no PDBR hyperparameter retuning;
- Validation selects checkpoints for new PDBR ablations;
- Test is report-only;
- formal PDBR Test ranking never calls the diffusion network (`diffusion_inference_calls=0`).
