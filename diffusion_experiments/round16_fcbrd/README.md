# Round16 FCBRD

Frozen-Reference CoLift-Informed Base-Relative Residual Diffusion.

- source: Round15 final `cf9638237cae2a7ff6372b85131920ba8fd33630`
- Stage R: TRAIN-only ZERO/base reconstruction, fixed 5 epochs, epoch5 frozen reference
- A1: frozen reference + absolute pair preference + identity + zero consistency + 5-degree bound
- A2: A1 with base-relative gain objective replacing absolute pair objective
- A3: A2 + 8D frozen CoLift candidate context
- hard pool: frozen Full-CoLiftRec ranks 6-30, TRAIN-only
- Test is never used for training/checkpoint/variant selection. Baby Test may be run only after Validation selection is frozen because the user explicitly authorized it for this round.
- Sports / Electronics remain closed.
