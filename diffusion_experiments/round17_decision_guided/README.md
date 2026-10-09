# Round17 TADGD — Target-Aware Decision-Guided Diffusion

Source: Round16R final commit `1e5a5493d1c6677840d938002514f9ffb8a06fa7`.

Frozen modules: MSCA/backbone, Full CoLiftRec assets, Round16/Round16R D_base generic diffusion basis. Only DecisionNet is trainable in D1/D2/D3.

Variants are preregistered and executed D0 -> D1 -> D2 -> D3. D2 is the primary target-aware-history decision variant. D3 appends frozen 8D CoLift candidate state only to DecisionNet. No CoLift context enters the diffusion generator.

TRAIN uses Round16R clean `history[:-1] -> history[-1]` same-row Top100 supervision. Validation uses full TRAIN history. Test is not used for implementation or Validation selection. The user explicitly authorized one exploratory Baby Test only after the final Validation variant/checkpoints are committed and locked. Sports/Electronics remain closed.
