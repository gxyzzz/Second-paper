# Round10 — Collaborative-Condition Differential Purification

Protocol: `ROUND10_CCDP_V1`.

Source branch/commit: `exp/round9-cabrp-20261008` / `a1104fd3c87a70accbed4cabb9f1eac0e08e42f3`.

The round keeps the frozen Baby backbones, CoLiftRec and original diffusion generator. It changes only the purification residual definition: paired TRUE/NULL DDIM trajectories share the exact same initial noisy state, and B3 applies only the tangent component of `y_true - y_null` to raw Text/Visual features. No ranking residual, ranking loss, preference loss, Test, Sports or Electronics is allowed.

Formal eta selection is lexicographic and frozen before Validation inspection: (1) maximize number of positive backbones, (2) maximize worst-seed U, (3) maximize mean U, (4) choose the smaller eta on an exact tie.

For the guide's qualitative `PARTIAL_SIGNAL` language, the implementation freezes the following deterministic interpretation before formal results: branch (a) requires >=3/4 positive and strictly improved worst-seed U versus B1; branch (b) requires B3 to beat B2 on >=3/4 backbones and also have larger four-backbone mean U. No post-hoc tolerance is introduced.
