# ROUND15_BAURP

Base-Anchored Bounded User Residual Purification.

Frozen protocol:
- source Round14 commit: 394b6acb2a9597124ce2ceb863bf47b3890cfc02
- recommender and Full CoLiftRec frozen
- A0: exact Round14 A3 control
- A1: ZERO/base-only reconstruction anchor, unbounded personalized residual
- A2: ZERO/base-only reconstruction anchor, 5-degree bounded personalized residual, lambda_bound=0.05
- TRUE/SHUFFLED preference uses TRAIN-only frozen Full-CoLiftRec hard negatives rank 6-30 and t=3
- base reconstruction uses random t=1..20
- preference residual subtracts detached ZERO/base prediction
- inference one-step, rho=0.20, seeds 20261301/20261302
- Test/Sports/Electronics closed
