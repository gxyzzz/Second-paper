# Paper Experiment Summary

Status: COMPLETE for the currently approved paper-experiment scope.

Protocol: seed=1000; frozen CoLiftRec/PDBR settings; Validation selects checkpoints; Test is report-only; PDBR is training-time only and `diffusion_inference_calls=0` during ranking.

## 1. Main three-domain results

U4 is the mean relative change over R@10, N@10, R@20 and N@20.

| Dataset | CoLiftRec vs MSCA | PDBR vs CoLiftRec | Full vs MSCA | Full primary |
|---|---:|---:|---:|---:|
| Baby | +4.48% | +2.40% | +6.99% | 4/4 |
| Sports | +5.50% | +3.20% | +8.87% | 4/4 |
| Electronics | +3.01% | +4.00% | +7.13% | 4/4 |

Full Test metrics:

| Dataset | R@10 | N@10 | R@20 | N@20 | R@50 | N@50 |
|---|---:|---:|---:|---:|---:|---:|
| Baby | 0.072726 | 0.039750 | 0.110479 | 0.049464 | 0.179375 | 0.063351 |
| Sports | 0.086167 | 0.046832 | 0.126438 | 0.057239 | 0.197375 | 0.071669 |
| Electronics | 0.053055 | 0.029549 | 0.078976 | 0.036249 | 0.123687 | 0.045357 |

Source: `docs/evidence/current/THREE_DOMAIN_SEED1000_RESULTS.json`.

## 2. CoLiftRec internal ablation

Values below are U4 changes relative to Full CoLiftRec; positive means the ablated variant is better than Full.

| Variant | Baby | Sports | Electronics |
|---|---:|---:|---:|
| w/o Generic Background Removal | +0.281% | -0.376% | +0.130% |
| w/o Text Lift | -1.281% | -1.567% | -0.460% |
| w/o Attribute Lift | -1.456% | -3.212% | -2.078% |
| w/o Visual Lift | +0.357% | -0.463% | +0.004% |

Interpretation supported by the evidence:

- Attribute lift is the strongest and most consistent modality contribution across all three domains.
- Text lift is also consistently positive across all three domains.
- Visual lift is weak/mixed: useful on Sports, approximately neutral on Electronics, and the Baby ablation is slightly better than Full.
- Generic-background removal is mixed across domains and should not be described as a uniformly necessary component based on this ablation alone.

Source: `paper_experiments/coliftrec_ablation/evidence/COLIFTREC_ABLATION_SUMMARY.csv`.

## 3. PDBR internal ablation

Values are U4 relative to the frozen CoLiftRec baseline.

| Variant | Baby | Sports | Electronics |
|---|---:|---:|---:|
| P1 Full PDBR | +2.404% | +3.203% | +3.997% |
| P2 Shuffled User | +1.261% | +3.553% | +3.668% |
| P3 w/o Preference Alignment | +2.536% | +3.925% | +4.008% |
| P4 w/o Angular Bound | +1.106% | +3.371% | +3.762% |
| P5 w/o BASE Residual | +1.548% | — | — |
| P6 w/o Boundary Curriculum | +1.312% | — | — |

Evidence notes:

- Baby strongly supports correct user identity, the angular bound, BASE residual, and the progressive boundary curriculum.
- Electronics shows small drops for shuffled user and no angular bound.
- Sports does not show the same monotonic ablation pattern.
- P3 is not worse than Full on any domain and is notably better on Sports. Therefore the current results do not support a claim that the explicit preference-alignment loss is necessary for ranking gains.
- If a PDBR ablation table is used in the paper, the full table should be reported rather than selectively hiding P3.

Source: `paper_experiments/pdbr_ablation/evidence/PDBR_ABLATION_SUMMARY.csv`.

## 4. Boundary-stratified Test analysis

Groups are defined using the frozen CoLiftRec baseline only, avoiding post-treatment grouping.

| Dataset | Top10 rescue: baseline 11-20 -> Full <=10 | Top20 rescue: baseline 21-30 -> Full <=20 | Top100 entry: baseline >100 -> Full <=100 |
|---|---:|---:|---:|
| Baby | 19.37% | 30.98% | 2.80% |
| Sports | 23.98% | 32.86% | 3.90% |
| Electronics | 23.93% | 33.42% | 3.07% |

This is the cleanest cross-domain mechanism result: PDBR consistently rescues a substantial fraction of positives immediately outside Top10/Top20 cutoffs. It is a redistribution effect rather than a monotonic improvement for every user: the already-Top10 G0 group loses some R@10, so the paper should describe boundary rescue precisely rather than claim universal rank improvement.

Source: `paper_experiments/boundary_analysis/evidence/BOUNDARY_ANALYSIS_SUMMARY.json`.

## 5. Efficiency and complexity

### Parameters

PDBR adds 164,416 trainable parameters during training; CoLiftRec adds zero trainable parameters. PDBR parameters are not deployed for ranking inference.

| Dataset | Backbone train params | Full train params | PDBR train-param increase | Full deploy params |
|---|---:|---:|---:|---:|
| Baby | 33,579,072 | 33,743,488 | +0.490% | 33,579,072 |
| Sports | 85,991,872 | 86,156,288 | +0.191% | 85,991,872 |
| Electronics | 298,889,728 | 299,054,144 | +0.055% | 298,889,728 |

### Training benchmark

RTX 5090; 20 warmup batches + 100 timed batches; batch size 2048. PDBR is measured with all 3 boundary bands active and `active_steps=2`, so this is a **full-active upper-bound**, not whole-training average overhead.

| Dataset | MSCA ms/batch | Full-active PDBR ms/batch | Time overhead | Allocated-VRAM overhead |
|---|---:|---:|---:|---:|
| Baby | 47.54 | 179.66 | +277.93% | +3.07% |
| Sports | 50.84 | 184.67 | +263.27% | +1.26% |
| Electronics | 72.39 | 217.80 | +200.89% | +1.11% |

PDBR substantially increases compute in the fully active training phase but adds little peak allocated VRAM.

### Validation inference benchmark

Median latency over 5 repeats. PDBR network is never called during inference.

| Dataset | MSCA ms/user | CoLiftRec ms/user | Full ms/user | Incremental Full vs CoLiftRec | Diffusion calls |
|---|---:|---:|---:|---:|---:|
| Baby | 0.00232 | 1.64918 | 1.65831 | +0.55% | 0 |
| Sports | 0.00237 | 1.68529 | 1.70727 | +1.30% | 0 |
| Electronics | 0.00309 | 1.87051 | 1.86807 | -0.13% | 0 |

The defensible efficiency statement is: **PDBR adds no diffusion-network inference stage and negligible incremental latency beyond CoLiftRec**. Do not claim that the complete Full method has zero inference overhead relative to raw MSCA, because CoLiftRec itself performs post-hoc content reranking.

Sources: `paper_experiments/efficiency/evidence/PARAMETER_COUNT.json`, `TRAINING_BENCHMARK.csv`, `TRAINING_OVERHEAD.json`, `INFERENCE_BENCHMARK.csv`, `ENVIRONMENT.json`, and `EFFICIENCY_SUMMARY.csv`.

## 6. Paper-facing evidence candidates

Strong candidates for the manuscript, without changing or hiding the complete repository evidence:

1. Main three-domain Test table: Full vs MSCA and module-wise U4 gains.
2. Boundary rescue analysis: especially the stable 30.98%-33.42% Top20 rescue rates across domains.
3. Efficiency: only +0.055%-0.490% training parameters for PDBR, no deployed PDBR parameters, zero diffusion inference calls, and negligible incremental latency over CoLiftRec.
4. CoLiftRec modality ablation: Attribute and Text have consistent three-domain contributions.
5. PDBR Baby ablations P2/P4/P5/P6 are mechanistically supportive, but if presented as a formal ablation study the contradictory P3 and Sports results must also remain visible.

## 7. Completion state

- CoLiftRec internal ablation: COMPLETE.
- PDBR internal ablation: COMPLETE.
- Boundary-stratified ranking analysis: COMPLETE.
- Parameter count: COMPLETE.
- Training efficiency benchmark: COMPLETE.
- Inference efficiency benchmark: COMPLETE.
- Final evidence summaries: COMPLETE.

No new method search, hyperparameter tuning, seed selection, or Test-based redesign was performed in this paper-experiment completion round.
