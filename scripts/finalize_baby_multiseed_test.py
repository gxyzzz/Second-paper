from __future__ import annotations
import json
from pathlib import Path
import numpy as np

SEEDS=[999,1000,1001,1002]
NEW=[1000,1001,1002]
PAPER=Path("docs/evidence/paper")
FREEZE=PAPER/"BABY_CANONICAL_BACKBONE_FROZEN_BEFORE_MULTISEED_TEST.json"

def load(seed):
    return json.load(open(f"runs/test/baby_multiseed/seed{seed}/summary.json"))

def main():
    PAPER.mkdir(parents=True,exist_ok=True)
    freeze=json.load(open(FREEZE))
    zs={s:load(s) for s in SEEDS}
    assert freeze["canonical_seed"]==1000
    assert all(zs[s]["BABY_DIFFUSION_TEST_RUN_COUNT"]==1 for s in SEEDS)
    assert all(zs[s]["TEST_USED_FOR_SELECTION"] is False for s in SEEDS)
    assert all(zs[s]["NO_POST_TEST_TUNING"] is True for s in SEEDS)

    rows=[]
    boots={}
    for s in SEEDS:
        z=zs[s]
        rows.append({
            "msca_seed":s,
            "canonical_backbone":bool(s==1000),
            "msca_best_epoch":z["MSCA"]["best_epoch"],
            "msca_checkpoint_sha256":z["MSCA"]["checkpoint_sha256"],
            "MSCA":z["MSCA"]["metrics"],
            "COLIFTREC":z["COLIFTREC"]["metrics"],
            "COLIFTREC_vs_MSCA":z["COLIFTREC"]["vs_MSCA"],
            "DIFFUSION":z["DIFFUSION"]["metrics"],
            "DIFFUSION_vs_COLIFTREC":z["DIFFUSION"]["vs_COLIFTREC"],
            "DIFFUSION_TEST_STATUS":z["DIFFUSION"]["TEST_STATUS"],
            "DIFFUSION_TEST_PASS":z["DIFFUSION"]["TEST_PASS"],
            "DIFFUSION_TEST_RUN_COUNT":z["BABY_DIFFUSION_TEST_RUN_COUNT"],
        })
        boots[str(s)]={
            "COLIFTREC_vs_MSCA":z["COLIFTREC"]["bootstrap_vs_MSCA"],
            "DIFFUSION_vs_COLIFTREC":z["DIFFUSION"]["bootstrap_vs_COLIFTREC"],
        }

    results={
        "phase":"BABY_MULTI_BACKBONE_FROZEN_TEST_RESULTS",
        "dataset":"baby",
        "canonical_seed":1000,
        "canonical_seed_frozen_before_test":True,
        "canonical_selection_metric":"MSCA_VALIDATION_R20",
        "rows":rows,
        "TEST_USED_FOR_SELECTION":False,
        "NO_POST_TEST_TUNING":True,
        "BABY_METHOD_SEARCH":"CLOSED",
        "BABY_PARAMETER_SEARCH":"CLOSED",
        "BABY_SEED_SEARCH":"CLOSED",
    }
    (PAPER/"baby_multiseed_test_results.json").write_text(json.dumps(results,indent=2)+"\n")

    bootstrap={
        "phase":"BABY_MULTI_BACKBONE_TEST_PAIRED_BOOTSTRAP",
        "dataset":"baby",
        "bootstrap_seed":20262002,
        "resamples":1000,
        "paired_user_resampling":True,
        "seeds":boots,
        "TEST_USED_FOR_SELECTION":False,
        "NO_POST_TEST_TUNING":True,
    }
    (PAPER/"baby_multiseed_test_bootstrap.json").write_text(json.dumps(bootstrap,indent=2)+"\n")

    can=zs[1000]
    can_status=can["DIFFUSION"]["TEST_STATUS"]
    canonical={
        "phase":"BABY_CANONICAL_TEST",
        "dataset":"baby",
        "canonical_seed":1000,
        "canonical_frozen_before_test":True,
        "selection_metric":"MSCA_VALIDATION_R20",
        "selection_does_not_use_diffusion":True,
        "selection_does_not_use_test":True,
        "freeze_evidence":str(FREEZE),
        "MSCA":can["MSCA"],
        "COLIFTREC":can["COLIFTREC"],
        "DIFFUSION":can["DIFFUSION"],
        "BABY_CANONICAL_DIFFUSION_TEST":can_status,
        "BABY_SEED1000_DIFFUSION_TEST_RUN_COUNT":1,
        "TEST_USED_FOR_SELECTION":False,
        "NO_POST_TEST_TUNING":True,
        "BABY_METHOD_SEARCH":"CLOSED",
        "BABY_PARAMETER_SEARCH":"CLOSED",
        "BABY_SEED_SEARCH":"CLOSED",
    }
    (PAPER/"baby_seed1000_canonical_test.json").write_text(json.dumps(canonical,indent=2)+"\n")

    all_u=np.array([zs[s]["DIFFUSION"]["vs_COLIFTREC"]["mean_relative_primary_gain"] for s in SEEDS],float)
    new_u=np.array([zs[s]["DIFFUSION"]["vs_COLIFTREC"]["mean_relative_primary_gain"] for s in NEW],float)
    pass4=sum(bool(zs[s]["DIFFUSION"]["TEST_PASS"]) for s in SEEDS)
    pos4=sum(zs[s]["DIFFUSION"]["vs_COLIFTREC"]["mean_relative_primary_gain"]>0 for s in SEEDS)
    pass3=sum(bool(zs[s]["DIFFUSION"]["TEST_PASS"]) for s in NEW)
    pos3=sum(zs[s]["DIFFUSION"]["vs_COLIFTREC"]["mean_relative_primary_gain"]>0 for s in NEW)
    all_colift_4=all(zs[s]["COLIFTREC"]["vs_MSCA"]["primary_4of4"] for s in SEEDS)
    all_colift_6=all(zs[s]["COLIFTREC"]["vs_MSCA"]["all_6of6"] for s in SEEDS)
    final={
        "phase":"BABY_MULTISEED_TEST_FINAL_SUMMARY",
        "dataset":"baby",
        "provenance":{
            "pre_test_canonical_freeze_git_commit":"7e36ae2d3dd6f29078fcf8fe14285e5be69cb331",
            "pre_test_evaluator_git_commit":"beceeb17da0adcdb5633c103316f524c14eb3d7b",
            "freeze_evidence":"docs/evidence/paper/BABY_CANONICAL_BACKBONE_FROZEN_BEFORE_MULTISEED_TEST.json",
        },
        "CANONICAL_RESULT":{
            "seed":1000,
            "selection_basis":"highest MSCA Validation R20 among frozen seeds before Test",
            "BABY_CANONICAL_DIFFUSION_TEST":can_status,
            "diffusion_primary_positive_count":can["DIFFUSION"]["vs_COLIFTREC"]["primary_positive_count"],
            "diffusion_mean_relative_primary_gain":can["DIFFUSION"]["vs_COLIFTREC"]["mean_relative_primary_gain"],
            "bootstrap_P_mean_relative_gain_gt_0":can["DIFFUSION"]["bootstrap_vs_COLIFTREC"]["mean_relative_primary_gain"]["P_gt_0"],
            "bootstrap_CI95_mean_relative_gain":can["DIFFUSION"]["bootstrap_vs_COLIFTREC"]["mean_relative_primary_gain"]["ci95"],
        },
        "ROBUSTNESS_ANALYSIS":{
            "seed_status":{str(s):zs[s]["DIFFUSION"]["TEST_STATUS"] for s in SEEDS},
            "Diffusion_Test_PASS_count":int(pass4),
            "Diffusion_Test_total":4,
            "Diffusion_Test_U_positive_count":int(pos4),
            "four_seed_mean_relative_primary_gain":float(all_u.mean()),
            "four_seed_std_relative_primary_gain":float(all_u.std(ddof=1)),
            "new_seed_1000_1001_1002":{
                "PASS_count":int(pass3),
                "total":3,
                "U_positive_count":int(pos3),
                "mean_relative_primary_gain":float(new_u.mean()),
                "std_relative_primary_gain":float(new_u.std(ddof=1)),
            },
            "backbone_sensitivity_observed":True,
        },
        "COLIFTREC_TEST_ROBUSTNESS":{
            "all_four_seed_primary_4of4_positive":bool(all_colift_4),
            "all_four_seed_6of6_positive":bool(all_colift_6),
        },
        "TEST_RUN_COUNTS":{f"BABY_SEED{s}_DIFFUSION_TEST_RUN_COUNT":1 for s in SEEDS},
        "TEST_USED_FOR_SELECTION":False,
        "NO_POST_TEST_TUNING":True,
        "BABY_METHOD_SEARCH":"CLOSED",
        "BABY_PARAMETER_SEARCH":"CLOSED",
        "BABY_SEED_SEARCH":"CLOSED",
        "scientific_note":"Canonical seed1000 remains canonical because it was frozen by MSCA Validation R20 before Test; multiseed Test outcomes are robustness evidence only.",
    }
    (PAPER/"BABY_MULTISEED_TEST_FINAL_SUMMARY.json").write_text(json.dumps(final,indent=2)+"\n")
    print(json.dumps(final,sort_keys=True))

if __name__=="__main__":
    main()
