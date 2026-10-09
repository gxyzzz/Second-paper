# Round21 Mechanism Preflight Report

## Verdict

**`ROUND21_MECHANISM_FAIL_FORMAL_BLOCKED`**

Smoke passed, but the preregistered mechanism preflight did not pass on either Baby backbone. Per protocol, B0/C1/D1 formal Validation was not started, no hyperparameter grid was opened, and Test/Sports/Electronics remained closed.

## Gate summary

| Seed | M1 ε learning | M2 norm | M3 modality | M4 user | M5 hardness | M6 curriculum | Overall |
|---:|---|---|---|---|---|---|---|
| 999 | PASS | FAIL | PASS | FAIL | PASS | FAIL | FAIL |
| 1000 | PASS | FAIL | FAIL | FAIL | FAIL | PASS | FAIL |

## 1. Diffusion epsilon prediction

- seed999: zero-baseline MSE ratios low/middle/high = 0.2954 / 0.0366 / 0.0862; non-collapse = True. M1 PASS.
- seed1000: zero-baseline MSE ratios low/middle/high = 0.2945 / 0.0365 / 0.0851; non-collapse = True. M1 PASS.

The denoiser therefore learns the forward epsilon task. This alone is not sufficient for usable negative generation.

## 2. Generated latent norm

- seed999: real-item p99 norm = 0.067074; synthetic p99 = 0.544038; ratio = 8.11x. M2 FAIL.
- seed1000: real-item p99 norm = 0.067027; synthetic p99 = 0.530790; ratio = 7.92x. M2 FAIL.

With the fixed T=25 linear schedule, alpha_bar at the terminal indexed step is 0.776486 (sqrt(1-alpha_bar)=0.472773). Starting generation from unit Gaussian is therefore much broader than the forward terminal distribution of these very small-norm item latents. This is recorded as a plausible diagnostic for the observed scale mismatch; no schedule or normalization change was made.

## 3. Modality guidance

- seed999: V visual 0.015270 vs user-only 0.011533; T text 0.015136 vs user-only 0.013936; TV visual/text 0.019136/0.015147. M3 PASS.
- seed1000: V visual -0.004203 vs user-only -0.001081; T text -0.002674 vs user-only -0.003903; TV visual/text -0.004644/-0.009608. M3 FAIL.

Guidance is not cross-backbone reliable: seed999 shows the intended small directional changes, while seed1000 does not.

## 4. User conditioning

- seed999: mean cos(TRUE-user, SHUFFLED-user) = 0.999882; mean latent distance = 0.006436; mean |score delta| = 0.00006079. M4 FAIL.
- seed1000: mean cos(TRUE-user, SHUFFLED-user) = 0.999879; mean latent distance = 0.006531; mean |score delta| = 0.00005969. M4 FAIL.

The generated negatives are almost invariant to the user condition under the fixed protocol.

## 5. Ranking hardness

- seed999: mean margins V/T/TV = 0.000278216 / 0.000282208 / 0.000273564; P(s_neg>=s_pos) = 0.428/0.418/0.426. M5 PASS.
- seed1000: mean margins V/T/TV = 0.000331014 / 0.000309047 / 0.000318461; P(s_neg>=s_pos) = 0.383/0.396/0.391. M5 FAIL.

TV is marginally hardest on seed999 but not on seed1000, so the intended difficulty hierarchy is not stable.

## 6. Curriculum

- seed999: early/middle/late used-negative mean margins = 0.000278216 / 0.000280212 / 0.000277996. M6 FAIL.
- seed1000: early/middle/late used-negative mean margins = 0.000331014 / 0.000320031 / 0.000319508. M6 PASS.

The easy→hard trend holds on seed1000 but not seed999.

## Required ten questions

1. **Did epsilon prediction learn?** Yes, M1 passes on both seeds.
2. **Did user conditioning work?** No; M4 fails on both seeds.
3. **Did text conditioning work?** Weak/inconsistent; seed999 passes the aggregate modality gate, seed1000 does not.
4. **Did visual conditioning work?** Not reliably across backbones; seed1000 moves in the wrong direction relative to user-only.
5. **Are TV negatives genuinely harder?** Not stably; true on seed999, false on seed1000.
6. **Is the curriculum genuinely easy→hard?** Not stably; M6 fails on seed999.
7. **Are generated latent norms normal?** No; synthetic p99 is about 8x the real-item p99 on both seeds.
8. **Does C1 online curriculum hard-negative improve ranking?** Not evaluated because mechanism gates block formal Validation.
9. **Does D1 beat C1?** Not evaluated because mechanism gates block formal Validation.
10. **How far from the +1% target?** Ranking distance is intentionally unresolved in this round because formal Validation was not scientifically admissible after mechanism failure.

## Protocol discipline

- No Test access.
- No Sports/Electronics access.
- No T/CFG/lambda/M/beta/network grid search after the mechanism failure.
- No B0/C1/D1 formal Validation was started.
- The result is a mechanism-level stop, not a Test-based ranking conclusion.
