from __future__ import annotations
import json
from pathlib import Path
import numpy as np

SEEDS=[999,1000,1001,1002]
NEW=[1000,1001,1002]
PAPER=Path("docs/evidence/paper")
SOURCE_COMMIT="48455de8efa943e16d49db665e7f2fcb0c6c5e17"

def load(s):
    return json.load(open(f"runs/backbone_robustness/baby_seed{s}/evidence/fixed_diffusion_validation.json"))

def main():
    PAPER.mkdir(parents=True,exist_ok=True)
    zs={s:load(s) for s in SEEDS}
    msca={
      "phase":"BABY_MSCA_MULTI_SEED_VALIDATION",
      "dataset":"baby","official_msca_source_commit":SOURCE_COMMIT,
      "selection_protocol":"Validation-only Recall@20 early stopping/model selection",
      "training_time_test_disabled":True,
      "seeds":{},
      "TEST_ACCESSED":False,"VALIDATION_ONLY":True,
    }
    rows=[]
    for s,z in zs.items():
        m=z["MSCA"]; d=z["DIFFUSION"]; c=z["COLIFTREC"]
        msca["seeds"][str(s)]={
          "reused_existing_clean_checkpoint":bool(s==999),
          "best_epoch":m["best_epoch"],"checkpoint_sha256":m["checkpoint_sha256"],
          "validation_metrics":m["metrics"],
        }
        rows.append({
          "msca_seed":s,"msca_best_epoch":m["best_epoch"],
          "msca_checkpoint_sha256":m["checkpoint_sha256"],
          "MSCA":m["metrics"],"COLIFTREC":c["metrics"],
          "COLIFTREC_delta_vs_MSCA":c["delta_vs_msca"],
          "COLIFTREC_primary_positive_count":c["primary_positive_count"],
          "DIFFUSION":d["metrics"],"DIFFUSION_delta_vs_COLIFTREC":d["delta_vs_coliftrec"],
          "DIFFUSION_U":d["U"],"DIFFUSION_primary_positive_count":d["primary_positive_count"],
          "DIFFUSION_primary_4of4":d["primary_4of4"],"DIFFUSION_PASS":d["PASS"],
          "bootstrap":d["bootstrap"],
          "diffusion_checkpoint_sha256":d["checkpoint_sha256"],
          "condition_sha256":d["condition_sha256"],
        })
    (PAPER/"baby_msca_multiseed_validation.json").write_text(json.dumps(msca,indent=2)+"\n")

    new_u=np.array([zs[s]["DIFFUSION"]["U"] for s in NEW],dtype=float)
    all_u=np.array([zs[s]["DIFFUSION"]["U"] for s in SEEDS],dtype=float)
    pass_count=sum(bool(zs[s]["DIFFUSION"]["PASS"]) for s in NEW)
    upos=sum(zs[s]["DIFFUSION"]["U"]>0 for s in NEW)
    colift_ok=all(zs[s]["COLIFTREC"]["primary_positive_count"]>=3 for s in SEEDS)
    robust=bool(pass_count>=2 and new_u.mean()>0)
    summary={
      "phase":"BABY_DIFFUSION_BACKBONE_ROBUSTNESS",
      "dataset":"baby",
      "hypothesis":"BABY_DIFFUSION_BACKBONE_SENSITIVITY",
      "provenance":{
        "official_msca_source_commit":SOURCE_COMMIT,
        "native_feature_sha":zs[999]["native_feature_sha"],
        "dataset_interaction_sha256":zs[999]["native_feature_sha"]["interaction"],
      },
      "fixed_protocol":{
        "msca_seeds":SEEDS,
        "new_from_scratch_msca_seeds":NEW,
        "coliftrec":{"lambda_T":1.0,"lambda_A":0.75,"lambda_V":0.25,
                     "alpha_T":0.25,"alpha_A":0.15,"alpha_V":0.025},
        "diffusion":{"beta":0.5,"training_seed":20261501,"epochs":80,
                     "checkpoint_selection":"final_epoch","train_item_count":7050,
                     "batch":512,"lambda_ctr":0.1,"p_uncond":0.15,
                     "optimizer":"AdamW","lr":0.001,"weight_decay":0.0001,
                     "t_edit":3,"guidance":2.0,"rho_T":0.25,"rho_V":1.0,
                     "purification_seeds":[20261001,20261002,20261003,20261004]},
      },
      "rows":rows,
      "new_seed_PASS_count":int(pass_count),
      "new_seed_U_positive_count":int(upos),
      "new_seed_mean_delta_U":float(new_u.mean()),
      "new_seed_std_delta_U":float(new_u.std(ddof=1)),
      "four_seed_mean_delta_U":float(all_u.mean()),
      "four_seed_std_delta_U":float(all_u.std(ddof=1)),
      "robustness_gate":{
        "required_new_seed_PASS_count":2,
        "required_new_seed_mean_delta_U_positive":True,
        "observed_new_seed_PASS_count":int(pass_count),
        "observed_new_seed_mean_delta_U":float(new_u.mean()),
        "PASS":robust,
      },
      "coliftrec_cross_backbone_stable_primary_ge_3of4":bool(colift_ok),
      "coliftrec_all_four_seed_primary_4of4":bool(all(zs[s]["COLIFTREC"]["primary_positive_count"]==4 for s in SEEDS)),
      "interpretation_flags":{
        "at_least_one_new_backbone_recovers_diffusion_PASS":bool(pass_count>=1),
        "multi_backbone_diffusion_robustness_established":robust,
        "seed999_can_be_explained_as_single_checkpoint_sensitivity":False,
      },
      "VALIDATION_ONLY":True,"BABY_TEST":"CLOSED","TEST_ACCESSED":False,"TEST_USED_FOR_SELECTION":False,
    }
    (PAPER/"baby_backbone_robustness_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps({
      "new_seed_PASS_count":pass_count,"new_seed_U_positive_count":upos,
      "new_mean_U":summary["new_seed_mean_delta_U"],"four_mean_U":summary["four_seed_mean_delta_U"],
      "four_std_U":summary["four_seed_std_delta_U"],"robustness_PASS":robust,
      "coliftrec_stable":colift_ok},sort_keys=True))
if __name__=="__main__": main()
