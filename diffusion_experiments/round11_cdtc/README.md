# Round11 — Conditional Diffusion Trajectory Calibration

Protocol: `ROUND11_CDTC_V1`.

Source: `exp/round10-ccdp-20261008` at `d31be7fdfba2f434d5a6a4fbbd8c2fe00c41061f`.

Round11 freezes the Round10 condition-differential tangent residual and CoLiftRec. It screens only guidance, beta and t_edit with OFAT, using eta=0.10 and a fixed 20-degree safety cap. The cap is not a search dimension.

Hard-shell diagnostics are frozen before formal results as follows. Each Validation `(user, positive-item)` pair is a unit. A positive absent from candidate Top100 has rank 101. `positive_item_rank_change = baseline_rank - config_rank`, so positive values mean improvement. Top10 shell uses baseline rank 8–12; harmful exits are 8–10 -> >10 and beneficial entries are 11–12 -> <=10. Top20 analogously uses 18–22. `E_rank = U / mean_applied_angle_deg`, where the denominator is the arithmetic mean of Text and Visual mean raw-to-final applied angular movement for that backbone.

Family selection is lexicographic: positive backbone count, worst-seed U, mean U, then conservative tie-break (smaller guidance; smaller beta / closer to collaborative beta=0; smaller t_edit). Boundary diagnostics and E_rank are supporting evidence, not a replacement for this selection rule. After screening, exactly one `C*` is formed from the three selected family parameters and evaluated on all four Baby backbones. No additional combinations are permitted.

For the Round11 boundary-based PARTIAL_SIGNAL alternative, `clear positive NetCross@10 / NetCross@20 across >=3 backbones` is frozen as: on at least three backbones, both `NetCross10 > 0` and `NetCross20 > 0` for the final C* configuration. If no registered PASS/PARTIAL criterion is met, the executor reports `NO_PASS` rather than inventing an extra success class.
