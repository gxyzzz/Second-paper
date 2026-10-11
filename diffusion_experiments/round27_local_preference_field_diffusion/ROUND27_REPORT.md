# Round27 — Local Personalized-Lift Preference Field Diffusion

## Final status

```text
ROUND27_STATUS = PREFLIGHT_BLOCKED
C1_REPRODUCIBILITY = FAIL
TARGET_STABILITY_AUDIT = PASS
SMOKE = PASS
GRADIENT_AUDIT = PASS
PREFLIGHT_PASS = FALSE
FORMAL_ALLOWED = FALSE
FORMAL_VALIDATION = NOT_RUN
BABY_TEST = CLOSED / NOT_ACCESSED
SPORTS_TEST = UNTOUCHED
ELECTRONICS_TEST = UNTOUCHED
SPORTS_VALIDATION = NOT_OPENED
ELECTRONICS_VALIDATION = NOT_OPENED
ROUND27_BABY_METHOD = NOT_FROZEN
```

Round27 was stopped before 60-epoch Formal exactly because the preregistered baseline reproducibility gate remained unresolved. No Validation ranking result for T1-LPF or D4-LPFD was produced, and no Test split was opened.

---

## 1. C1 reproducibility audit

Baby seed999, same code / seed / GPU / environment, epochs 0–15.

Across all attempts:

- initial state hash: exact;
- normal-plan hashes: exact;
- rank-offset hashes: exact;
- epochs 0–9 R20: exact;
- divergence starts exactly when current-boundary hard-negative mining activates at epoch10.

### Attempt 1 — inherited miner

```text
selected-real-item hashes exact = false
R20 exact = false
max |ΔR20| = 0.00037284649
```

### Attempt 2 — float64 mining + stable sort / deterministic item-id tie break

```text
selected-real-item hashes exact = false
R20 exact = false
max |ΔR20| = 0.00028835911
```

### Attempt 3 — attempt 2 + deterministic PyTorch/CUBLAS execution

```text
selected-real-item hashes exact = false
R20 exact = false
max |ΔR20| = 0.00033427616
```

Deterministic attempt active-phase R20:

| epoch | C1_REPRO_A | C1_REPRO_B |
|---:|---:|---:|
| 10 | 0.0336178966 | 0.0335150424 |
| 11 | 0.0348418617 | 0.0349447159 |
| 12 | 0.0356518385 | 0.0354975572 |
| 13 | 0.0371775092 | 0.0369203737 |
| 14 | 0.0422281404 | 0.0425624166 |
| 15 | 0.0496512753 | 0.0498000465 |

Therefore:

```text
C1_REPRODUCIBILITY = FAIL
```

The Round26 C1 path sensitivity is real and reproducible as a phenomenon: tiny numerical differences are amplified by online current-boundary mining once the hard-negative auxiliary path begins.

---

## 2. Numerical source audit

Two independent deterministic epoch0–9 warmup runs were saved and compared.

```text
normal plan exact = true
warmup state hash exact = false
max parameter abs diff = 9.9186e-7
max parameter relative diff = 5.3212e-4
```

For a fixed TRAIN batch after warmup:

```text
cross-run score abs diff median = 4.5620e-10
cross-run score abs diff p99    = 3.5445e-9
cross-run score abs diff max    = 2.2333e-8
```

Normal adjacent Top100 rank gaps were much larger:

```text
median = 2.0231e-5
p10    = 2.3943e-6
```

At this static epoch9 snapshot the Top100 set and order were already 100% identical; divergence emerges through the online feedback loop after recommender updates begin.

Additional TRAIN-only tie-bucket diagnostics showed that forcing wider numerical buckets does not provide a clean exact solution: thresholds large enough to alter several percent of ranked slots still failed to guarantee exact ordering. Therefore no wider tie rule was adopted, because the guide explicitly forbids changing normal score geometry merely to pass the gate.

---

## 3. LPF target stability audit

TRAIN-only, seed999, 1024 events, Easy/Medium/Hard boundary targets.

### History and confidence

```text
history >= 3 fraction     = 84.5703%
history == 5 fraction     = 61.3281%
no-refinement fraction    = 15.4297%
rho median                = 0.6051
confidence median         = 0.5230
confidence p10            = 0.0
confidence max            = 0.8369
current positive excluded = true
Validation/Test history   = not accessed
```

### Stability versus Round26 single-positive surrogate

```text
LPF bootstrap cosine median              = 0.10241
single-positive direction cosine median  = 0.05626
LPF - single median                      = +0.04615
```

Thus:

```text
LPF_MORE_STABLE_THAN_SINGLE_POSITIVE = TRUE
```

However, the absolute split-half directional agreement remains modest. This is evidence that LPF improves target stability, not evidence that the target is a high-confidence ground truth direction.

### Personalized lift participation

```text
personalized lift nonzero fraction = 100%
```

Frozen CoLiftRec parameters used in target construction:

```text
lambda_T = 1.00
lambda_A = 0.75
lambda_V = 0.25
alpha_T  = 0.25
alpha_A  = 0.15
alpha_V  = 0.025
```

Therefore personalized lift genuinely participates in the target rather than being bypassed.

---

## 4. Smoke and gradient topology

### Smoke

```text
TRAIN history only                 = PASS
current positive excluded          = PASS
Top-K local history                = PASS
CoLift frozen lambda/alpha         = PASS
LPF weights sum to 1               = PASS
rho/confidence range               = PASS
history < 3 -> no refinement       = PASS
all diffusion branches reconstruct = PASS
L_field finite                     = PASS
real-negative gradient             = nonzero
T1/D4 delta detached               = PASS
max angle                          = 4.999946 deg
Test flags                         = false
```

### Gradient audit

```text
C1 real-negative grad norm = 0.0063313707
T1 real-negative grad norm = 0.0063313735
D4 real-negative grad norm = 0.0063313707
ranking BPR -> Diffusion grad sum = 0
```

Hence the intended topology is correct: the real catalog negative remains graph-connected, while the synthetic refinement delta is detached and ranking loss does not leak into Diffusion.

---

## 5. TRAIN-only preflight mechanism

Baby seed999, 1024 TRAIN events, 250 online Diffusion updates.

Warmup reconstruction improved:

```text
L_rec epoch0 = 1.03990
L_rec epoch9 = 0.90726
```

### LPF statistics

```text
support-size median      = 5
history >= 3 fraction    = 85.0586%
no-refinement fraction   = 14.9414%
rho median               = 0.62088
confidence median        = 0.53304
personalized lift median = 0.00296
```

### Diffusion alignment to LPF

```text
A_true median      = 0.60469
A_shuf median      = 0.00000
DeltaA median      = 0.58744
P(DeltaA > 0)      = 84.5703%
raw residual norm median = 0.001804
near-zero residual fraction = 0
```

This is a strong mechanism result: conditional Diffusion learned a direction substantially closer to the LPF target for the true user than for a shuffled user.

### Finite refinement

```text
applied refinement fraction = 33.8867%
target reachable fraction   = 22.0703%
cap-hit fraction             = 11.8164%
angle median                 = 0.01978 deg
angle p90                    = 4.99991 deg
angle max                    = 5.00006 deg
actual score shift median    = 0
actual score shift p90       = 0.001342
```

The 5-degree trust region is respected and cap saturation is not the dominant regime.

A logging-only audit condition was corrected after preflight: already-hard negatives (`positive margin <= 0`) correctly receive no refinement and must be exempt from the `target <= positive` check. This correction changes no training output or mechanism statistic.

### Preflight verdict

```text
smoke_pass = true
gradient_audit_pass = true
RANKING_SCORE_GRAD_TO_DIFFUSION = 0
PREFLIGHT_PASS = false
BLOCK_REASONS = [C1_REPRODUCIBILITY_UNRESOLVED]
FORMAL_ALLOWED = false
```

---

## 6. Required Round27 questions

1. **Is C1 same-code same-seed truly deterministic?**  
   No. Three audits, including float64 stable mining and deterministic execution, still diverged from epoch10.

2. **Is the Round26 C1 path sensitivity real?**  
   Yes. The divergence begins exactly when online boundary mining activates, while initialization, normal plan, rank-offset plan, and epochs0–9 metrics are exact.

3. **Is LPF more stable than the single-positive proxy?**  
   Yes, diagnostically: bootstrap cosine median improves from 0.0563 to 0.1024 (+0.0462), although absolute consistency is still modest.

4. **Does CoLift personalized lift genuinely participate in the target?**  
   Yes. Frozen lambda/alpha are used and personalized lift is nonzero for 100% of audited local-history entries.

5. **How consistent are multiple history directions?**  
   Direction concentration is moderate (`rho` median about 0.61), while split-half LPF cosine median is only about 0.10. The field is more stable than a single-positive surrogate but remains multi-interest/noisy.

6. **Can confidence suppress uncertain events?**  
   Mechanistically yes: confidence is non-degenerate, p10 is 0, and about 15% of audited targets receive no refinement due insufficient support. This is a mechanism observation, not a ranking-utility claim.

7. **Is TRUE Diffusion closer to LPF than SHUFFLED?**  
   Yes in preflight: median `A_true=0.6047`, `A_shuf=0`, median `DeltaA=+0.5874`, with `P(DeltaA>0)=84.57%`.

8. **Does T1-LPF beat C1?**  
   Not evaluated. Formal was preregistered-blocked before ranking comparison.

9. **Does D4-LPFD beat C1 on both seeds?**  
   Not evaluated.

10. **Does D4 beat B0 on both seeds?**  
    Not evaluated.

11. **Does D4 reach or exceed T1?**  
    Not evaluated.

12. **Is Round27 qualified for Sports / Electronics Validation?**  
    No. Baby Formal was not allowed, so the method cannot be frozen or transferred cross-domain.

---

## 7. Scientific interpretation

Round27 produced two useful findings but no valid ranking claim.

First, the new LPF supervision is scientifically better motivated than the Round26 single-positive direction and is measurably more stable. The conditional Diffusion learner also shows strong true-user alignment to that LPF target without gradient leakage, residual collapse, or trust-region saturation.

Second, the current-boundary C1 baseline itself is path-sensitive under online mining. The observed numerical state differences are tiny, but the online hard-negative feedback loop amplifies them into different selected item IDs and different R20 trajectories. Because the preregistered protocol requires exact C1 reproducibility before Formal, this unresolved baseline instability invalidates a clean T1/D4 ranking comparison in Round27.

Therefore the correct final status is not `LPF FAIL` and not `DIFFUSION FAIL`. It is:

```text
ROUND27_MECHANISM = HEALTHY / PROMISING
ROUND27_BASELINE_REPRODUCIBILITY = UNRESOLVED
ROUND27_FORMAL = BLOCKED
ROUND27_RANKING_CONCLUSION = NOT ESTABLISHED
```

No Test result was used, and no cross-domain Validation was opened.
