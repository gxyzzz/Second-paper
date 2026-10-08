# Round9 — Confidence-Adaptive Bounded Residual Purification

Active branch: `exp/round9-cabrp-20261008`.

Round9 closes the Round1–8 direct-ranking diffusion line. Diffusion is frozen and used only for multimodal representation purification. Final recommendation remains frozen Full CoLiftRec.

Validation-only protocol:
- Baby backbones: 999/1000/1001/1002.
- A0: historical original fixed-rho purification (`rho_T=.25`, `rho_V=1.0`).
- A1: K=4 diffusion ensemble target + uniform angular SLERP cap at 5/10/15 degrees.
- A2: same cap, scaled item-wise by `sqrt(sample_consistency * condition_confidence)`.
- Condition confidence is calibrated from positive TRAIN-side TRUE-vs-SHUFFLED/NULL reconstruction margins only.
- No Test, Sports, Electronics, ranking loss, ranking score residual, learned gate, or Test-based selection.

Formal stops at `VALIDATION_DECISION`.
