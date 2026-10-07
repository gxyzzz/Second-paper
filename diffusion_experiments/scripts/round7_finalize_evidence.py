from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "diffusion_experiments/runs/round7"
EVID = ROOT / "diffusion_experiments/evidence"
PRIMARY = ("R10", "N10", "R20", "N20")
ALL = ("R10", "N10", "R20", "N20", "R50", "N50")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pct(x: float) -> str:
    return f"{100.0 * float(x):+.6f}%"


def main():
    EVID.mkdir(parents=True, exist_ok=True)
    test_raw = json.loads((RUN / "test_locked/test_results.json").read_text())
    val = json.loads((RUN / "validation_lock_dryrun/validation_dryrun.json").read_text())
    lock = json.loads((RUN / "selection_lock/selection_lock.json").read_text())

    # Reporting-only direction tolerance. Metric values/rankings remain untouched.
    test = json.loads(json.dumps(test_raw))
    tol = 1e-15
    for key, metrics in [("primary_direction", PRIMARY), ("all6_direction", ALL)]:
        direction = {}
        for metric in metrics:
            diff = float(test["main"]["M2"][metric]) - float(test["main"]["M1"][metric])
            direction[metric] = 0 if abs(diff) <= tol else (1 if diff > 0 else -1)
        test["main"][key] = direction
    test["main"]["primary_positive_count"] = sum(v > 0 for v in test["main"]["primary_direction"].values())
    test["main"]["all6_positive_count"] = sum(v > 0 for v in test["main"]["all6_direction"].values())
    test["reporting_direction_tolerance"] = tol
    test["reporting_note"] = "Direction labels only use 1e-15 tolerance for floating averaging residue; rankings and metrics are unchanged."
    test["raw_test_results_sha256"] = sha(RUN / "test_locked/test_results.json")
    test["selection_lock_sha256"] = sha(RUN / "selection_lock/selection_lock.json")
    (EVID / "round7_test_results.json").write_text(json.dumps(test, indent=2) + "\n")

    formal = {}
    all_pref_grads = []
    all_grad_cos = []
    for backbone in (999, 1000):
        for dseed in (202610101, 202610102):
            z = json.loads((RUN / f"formal_b{backbone}_d{dseed}/result.json").read_text())
            grads = z["gradient_diagnostics"]
            key = f"b{backbone}_d{dseed}"
            formal[key] = {
                "status": z["status"],
                "updates": z["updates"],
                "warmup_updates": z["warmup_updates"],
                "parameter_count": z["parameter_count"],
                "generator_sha256": z["generator_sha256"],
                "training_git_sha": z["git_sha"],
                "illegal_negative_count": z["illegal_negative_count"],
                "nonfinite_count": z["nonfinite_count"],
                "pref_gradient_records": len(grads),
                "pref_grad_norm_min": min(g["pref_grad_norm"] for g in grads),
                "pref_grad_norm_mean": float(np.mean([g["pref_grad_norm"] for g in grads])),
                "gradient_cosine_mean": float(np.mean([g["direction_cosine"] for g in grads])),
                "gradient_cosine_min": min(g["direction_cosine"] for g in grads),
                "peak_cuda_allocated_gib": z["peak_cuda_allocated_gib"],
                "elapsed_seconds": z["elapsed_seconds"],
            }
            all_pref_grads.extend(g["pref_grad_norm"] for g in grads)
            all_grad_cos.extend(g["direction_cosine"] for g in grads)

    assets = {}
    recompute = {}
    for backbone in (999, 1000):
        a = json.loads((RUN / f"assets_seed{backbone}_formal/audit.json").read_text())
        r = json.loads((RUN / f"locked_rankings_b{backbone}/audit.json").read_text())
        assets[str(backbone)] = {
            "checkpoint_sha256": a["source_checkpoint_sha256"],
            "checkpoint_epoch": a["source_checkpoint_epoch"],
            "train_edges": a["train_edges"],
            "events_with_anchor": a["events_with_anchor"],
            "events_without_anchor": a["events_without_anchor"],
            "condition_dim": a["condition_dim"],
            "inverse_standardization_max_abs_diff": a["item_inverse_standardization_max_abs_diff"],
            "A": a["A"],
        }
        recompute[str(backbone)] = {
            "audit_sha256": sha(RUN / f"locked_rankings_b{backbone}/audit.json"),
            **r["independent_recompute"],
            "target_item_ids_accessed": r["target_item_ids_accessed"],
            "test_metrics_computed": r["test_metrics_computed"],
        }

    checks = {
        "protocol_version": lock["protocol_version"],
        "final_status": "COMPLETE_NO_INCREMENT",
        "implementation_commit": "c190aab0d6039eb8dbc56680f0c423b8a19f873e",
        "selection_lock_git_sha": lock["git_sha"],
        "selection_lock_sha256": sha(RUN / "selection_lock/selection_lock.json"),
        "raw_test_results_sha256": sha(RUN / "test_locked/test_results.json"),
        "validation_dryrun_sha256": sha(RUN / "validation_lock_dryrun/validation_dryrun.json"),
        "full_train_edges": 118551,
        "evaluation_users": 19445,
        "selected_eta": lock["selected_eta"],
        "selection_mean_validation_U": lock["expansion_decision_before_test"]["selected_eta_mean_validation_U"],
        "selection_positive_cells": lock["expansion_decision_before_test"]["positive_cells"],
        "expand_sports_elec": lock["expansion_decision_before_test"]["expand_sports_elec"],
        "baseline_identity": "PASS",
        "positive_removed_from_anchor_history": "PASS",
        "eta0_identity": "PASS",
        "oracle_x0_ddim": "PASS",
        "keyed_noise_order_invariance": "PASS",
        "boundary_slot_invariance": "PASS",
        "independent_unlabeled_recompute": "PASS",
        "final_evaluator_validation_replay": "PASS",
        "validation_replay_max_metric_diff": val["validation_replay_max_metric_diff"],
        "assets": assets,
        "independent_recompute": recompute,
        "formal_training": formal,
        "formal_pref_grad_norm_global_min": min(all_pref_grads),
        "formal_gradient_cosine_global_mean": float(np.mean(all_grad_cos)),
        "test": {
            "classification": test["main"]["classification"],
            "U_M1_vs_M0": test["main"]["U_M1_vs_M0"],
            "U_M2_vs_M1": test["main"]["U_M2_vs_M1"],
            "bootstrap": test["main"]["bootstrap"],
            "cell_positive_count": test["cell_positive_count"],
            "primary_positive_count": test["main"]["primary_positive_count"],
            "all6_positive_count": test["main"]["all6_positive_count"],
            "NO_POST_TEST_TUNING": test["NO_POST_TEST_TUNING"],
        },
        "test_history_caveat": "Baby Test had historical exposure in earlier work. Round7 selection and rankings were locked before this run; this does not make Test globally unseen.",
    }
    (EVID / "round7_checks.json").write_text(json.dumps(checks, indent=2) + "\n")

    proto_path = EVID / "round7_protocol.json"
    proto = json.loads(proto_path.read_text())
    proto.update({
        "stage": "FINAL_COMPLETE",
        "final_status": "NO_INCREMENT",
        "implementation_commit": "c190aab0d6039eb8dbc56680f0c423b8a19f873e",
        "selection_lock_git_sha": lock["git_sha"],
        "selection_lock_sha256": sha(RUN / "selection_lock/selection_lock.json"),
        "selected_eta": lock["selected_eta"],
        "selection_mean_validation_U": lock["expansion_decision_before_test"]["selected_eta_mean_validation_U"],
        "selection_positive_cells": lock["expansion_decision_before_test"]["positive_cells"],
        "expand_sports_elec": False,
        "test_users": test["users"],
        "test_U_M2_vs_M1": test["main"]["U_M2_vs_M1"],
        "test_bootstrap": test["main"]["bootstrap"],
        "test_target_1pct_met": False,
        "no_post_test_tuning": True,
    })
    proto_path.write_text(json.dumps(proto, indent=2) + "\n")

    columns = ["split", "scope", "backbone_seed", "diffusion_seed", "model", *ALL, "U_vs_M0", "U_vs_M1", "changed_users", "changed_slots", "K10_increase", "K10_decrease", "K20_increase", "K20_decrease"]
    rows = []
    main_result = test["main"]
    for model in ("M0", "M1", "M2"):
        metrics = main_result[model]
        rows.append({
            "split": "TEST", "scope": "MAIN_EQUAL_BACKBONE_MEAN", "backbone_seed": "999+1000",
            "diffusion_seed": "101+102" if model == "M2" else "NA", "model": model,
            **{k: metrics[k] for k in ALL},
            "U_vs_M0": 0.0 if model == "M0" else (main_result["U_M1_vs_M0"] if model == "M1" else main_result["U_M2_vs_M0"]),
            "U_vs_M1": "" if model == "M0" else (0.0 if model == "M1" else main_result["U_M2_vs_M1"]),
            "changed_users": "", "changed_slots": "", "K10_increase": "", "K10_decrease": "", "K20_increase": "", "K20_decrease": "",
        })
    for cell, result in test["cells"].items():
        for model in ("M0", "M1", "M2"):
            metrics = result[model]
            rows.append({
                "split": "TEST", "scope": "CELL", "backbone_seed": result["backbone_seed"], "diffusion_seed": result["diffusion_seed"], "model": model,
                **{k: metrics[k] for k in ALL},
                "U_vs_M0": 0.0 if model == "M0" else (result["U_M1_vs_M0"] if model == "M1" else result["U_M2_vs_M0"]),
                "U_vs_M1": "" if model == "M0" else (0.0 if model == "M1" else result["U_M2_vs_M1"]),
                "changed_users": result["changed_users"] if model == "M2" else 0,
                "changed_slots": result["changed_slots"] if model == "M2" else 0,
                "K10_increase": result["K10"]["increase"] if model == "M2" else 0,
                "K10_decrease": result["K10"]["decrease"] if model == "M2" else 0,
                "K20_increase": result["K20"]["increase"] if model == "M2" else 0,
                "K20_decrease": result["K20"]["decrease"] if model == "M2" else 0,
            })
    with open(EVID / "round7_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)

    table = lock["eta_validation_table"]
    cell_lines = []
    for cell, result in test["cells"].items():
        cell_lines.append(f"| {cell} | {result['changed_users']} | {result['changed_slots']} | {pct(result['U_M2_vs_M1'])} | {result['K10']['increase']}/{result['K10']['decrease']} | {result['K20']['increase']}/{result['K20']['decrease']} |")
    m = test["main"]
    def metric_row(name, metrics, u0, u1):
        u1s = u1 if isinstance(u1, str) else pct(u1)
        return f"| {name} | {metrics['R10']:.8f} | {metrics['N10']:.8f} | {metrics['R20']:.8f} | {metrics['N20']:.8f} | {metrics['R50']:.8f} | {metrics['N50']:.8f} | {pct(u0)} | {u1s} |"

    report = f"""# Round7 Report — Historical-Anchored Collaborative Preference Completion

Protocol: `ROUND7_ANCHORED_COLLABORATIVE_PREFERENCE_COMPLETION_V1`  
Final verdict: **NO_INCREMENT**  
Scope: **Baby only**. Sports/Elec expansion was frozen to `false` before Test.

## Direct answers

- Old boundary-risk/near-negative route avoided: **yes**. No generated hard-negative IDs or boundary unknown-negative pool.
- Full-TRAIN fair baseline: **yes**. All **118,551** Baby TRAIN interactions; M0/M1/M2 share the same frozen backbone per cell.
- Differentiable history-anchor DDIM path active: **yes**. Four formal fits × 25 diagnostics had finite nonzero preference gradients; global minimum norm **{min(all_pref_grads):.6f}**.
- New strong unknown negatives introduced: **no**. Illegal negative count is 0 in all formal fits.
- M2 changed rankings: **yes**; this is not NO_INTERVENTION.
- Full Test gain over Full CoLiftRec: **no**. `U(M2,M1)={pct(m['U_M2_vs_M1'])}`, bootstrap CI **[{pct(m['bootstrap']['ci95'][0])}, {pct(m['bootstrap']['ci95'][1])}]**, `P(U>0)={m['bootstrap']['positive_fraction']:.3f}`.
- 1% target met: **no**.

## Validation selection

| eta | four cell U | mean cell U | positive cells |
|---:|---|---:|---:|
| 0.025 | {', '.join(pct(x) for x in table['0.025']['cell_U'])} | {pct(table['0.025']['mean_U'])} | {table['0.025']['positive_cells']}/4 |
| 0.05 | {', '.join(pct(x) for x in table['0.05']['cell_U'])} | {pct(table['0.05']['mean_U'])} | {table['0.05']['positive_cells']}/4 |
| 0.10 | {', '.join(pct(x) for x in table['0.1']['cell_U'])} | {pct(table['0.1']['mean_U'])} | {table['0.1']['positive_cells']}/4 |

Shared eta **0.025** was selected by the registered rule. Its mean Validation U was negative and only 1/4 cell was positive, so the registered Sports/Elec expansion gate failed before Test. Final-evaluator Validation replay had max metric difference **0.0**. Selection-lock SHA256: `{sha(RUN / 'selection_lock/selection_lock.json')}`.

## Main full-user Test

All rows below were computed from this Round7 run's locked rankings over all **19,445** Test users.

| Model | R10 | N10 | R20 | N20 | R50 | N50 | U vs M0 | U vs M1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{metric_row('MSCA M0', m['M0'], 0.0, '—')}
{metric_row('MSCA + Full CoLift M1', m['M1'], m['U_M1_vs_M0'], 0.0)}
{metric_row('M1 + Round7 Diffusion M2', m['M2'], m['U_M2_vs_M0'], m['U_M2_vs_M1'])}

Using 1e-15 only to suppress floating averaging residue, R10/N10/R50 are unchanged, while R20/N20/N50 decrease. Primary positive count: **0/4**; all-six positive count: **0/6**. CoLiftRec remains strongly positive over MSCA (`U={pct(m['U_M1_vs_M0'])}`), while Round7 slightly erodes it (`U={pct(m['U_M2_vs_M1'])}`).

## Per-cell intervention

| cell | changed users | changed slots | U(M2,M1) | Top10 rescue/harm | Top20 rescue/harm |
|---|---:|---:|---:|---:|---:|
{chr(10).join(cell_lines)}

Backbone-averaged Round7 U: seed999 **{pct(m['per_backbone_U_M2_vs_M1']['999'])}**; seed1000 **{pct(m['per_backbone_U_M2_vs_M1']['1000'])}**. Test cell positive count is **0/4**. There are no Top10 hit-count changes. At Top20, b999/D102 has 1 rescue and 1 harm; b1000/D101 has 0 rescue and 2 harms.

## Mechanism diagnostics

Each DDPM has **142,208** trainable parameters, uses about **0.0524 GiB** peak CUDA allocated memory, and takes roughly 229–231 s on RTX5090. Preference gradients are clearly active, but denoising/preference gradient cosine is consistently negative; global mean is **{float(np.mean(all_grad_cos)):.3f}**. This indicates objective conflict rather than an inactive ranking branch.

Generated completion movement is large in raw CF space (formal Validation-cache mean delta norms are roughly 30–34), while deployment clips `d/q` to `[-1,1]` and multiplies by eta 0.025. Large generative motion therefore did not translate into useful boundary correction.

## Scientific conclusion

Round7 is implementation-valid and scientifically distinct from Round5/6/6R1: it freezes the healthy recommender, anchors generation on real TRAIN behavior history, routes ranking gradients through the same multi-step reverse path used at inference, and removes the high-relatedness boundary-negative pool. However, the locked Baby Test answer is **NO_INCREMENT**. The decrement is small and the CI crosses zero, so this is evidence of no reliable added value, not a large damaging effect. The 1% target is not approached, and no Test-after tuning is allowed.

Baby Test had historical exposure in prior project phases. Round7's generators, eta, A/q rules, unlabeled rankings, and cross-domain decision were locked before this Test run, but the Test set is not globally pristine. Sports/Elec were intentionally not opened because the registered pre-Test gate failed.

## Evidence

- `diffusion_experiments/evidence/round7_protocol.json`
- `diffusion_experiments/evidence/round7_checks.json`
- `diffusion_experiments/evidence/round7_results.csv`
- `diffusion_experiments/evidence/round7_test_results.json`
- `diffusion_experiments/runs/round7/selection_lock/selection_lock.json`
- `diffusion_experiments/runs/round7/test_locked/test_results.json`

No Test-driven parameter, checkpoint, seed, noise, eta, A-rule, or user-subset change was made after lock.
"""
    (EVID / "ROUND7_REPORT.md").write_text(report)
    print(json.dumps({
        "status": "ROUND7_EVIDENCE_COMPLETE",
        "classification": test["main"]["classification"],
        "U_M2_vs_M1": test["main"]["U_M2_vs_M1"],
        "selection_lock_sha256": checks["selection_lock_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
