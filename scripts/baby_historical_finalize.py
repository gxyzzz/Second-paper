from __future__ import annotations
import json, shutil
from pathlib import Path
from diffusion_validate import btag, gtag, sha256

ROOT=Path("runs/diffusion_rescue/baby_historical_seed_replay")
E=ROOT/"evidence"
BETAS=[0.0,0.25,0.5,0.75,1.0]
SEEDS={0.0:20261001,0.25:20261251,0.5:20261501,0.75:20261751,1.0:20261101}
PROTOCOL="HISTORICAL_M31B_EXACT"
NOISE=[20261001,20261002,20261003,20261004]

def meta_path(beta):
    if beta==1.0: return Path("runs/diffusion_repro/baby_m31_formal/evidence/beta_1p0_training.json")
    tag={0.0:"beta_0p0",0.25:"beta_0p25",0.5:"beta_0p5",0.75:"beta_0p75"}[beta]
    ev={0.0:"beta_0p0_training.json",0.25:"beta_0p2_training.json",0.5:"beta_0p5_training.json",0.75:"beta_0p8_training.json"}[beta]
    return ROOT/tag/"evidence"/ev

def audit():
    by={}
    for b in BETAS:
        mp=meta_path(b); m=json.loads(mp.read_text())
        req={"training_seed":SEEDS[b],"epochs":80,"final_epoch":80,"train_item_count":7050,
             "training_protocol":"m31_fixed_all_items","train_scope":"all_catalog_items",
             "checkpoint_selection":"final_epoch","VALIDATION_TARGET_USED":False,
             "TEST_TARGET_USED":False,"TEST_ACCESSED":False}
        for k,v in req.items():
            if m.get(k)!=v: raise RuntimeError(("training provenance",b,k,m.get(k),v))
        ck=ROOT/"checkpoints"/f"baby_beta_{btag(b)}.pt"
        cond=ROOT/"assets"/f"condition_beta_{btag(b)}.npy"
        if sha256(ck)!=m["checkpoint_sha256"] or sha256(cond)!=m["condition_sha256"]:
            raise RuntimeError(("sha provenance",b))
        n=0
        for t in [2,3,5,7,10]:
            for g in [1.0,1.25,1.5,1.75,2.0]:
                p=ROOT/"purified"/f"beta_{btag(b)}"/f"t{t}_g{gtag(g)}.json"
                z=json.loads(p.read_text())
                for k,v in {"training_seed":SEEDS[b],"seed_protocol":PROTOCOL,
                            "checkpoint_sha256":m["checkpoint_sha256"],
                            "condition_sha256":m["condition_sha256"],
                            "purification_seeds":NOISE,"TEST_ACCESSED":False}.items():
                    if z.get(k)!=v: raise RuntimeError(("purified provenance",b,t,g,k,z.get(k),v))
                n+=1
        by[str(b)]={"training_seed":SEEDS[b],"checkpoint_sha256":m["checkpoint_sha256"],
                    "condition_sha256":m["condition_sha256"],"purified_pair_count":n,
                    "training_evidence":str(mp)}
    out={"phase":"BABY_HISTORICAL_SEED_PROVENANCE_AUDIT","seed_protocol":PROTOCOL,
         "beta_provenance":by,"purification_seeds":NOISE,"PASS":True,"TEST_ACCESSED":False}
    (E/"historical_seed_provenance_audit.json").write_text(json.dumps(out,indent=2)+"\n")
    return by

def annotate_grid(by):
    gp=E/"baby_validation_grid.json"; grid=json.loads(gp.read_text())
    if grid.get("diffusion_candidate_count")!=3125 or grid.get("TEST_ACCESSED") is not False:
        raise RuntimeError("generic combine grid invalid")
    for r in grid["rows"]:
        if r.get("kind")=="DIFFUSION":
            b=float(r["beta"]); p=by[str(b)]
            r["training_seed"]=SEEDS[b]; r["seed_protocol"]=PROTOCOL
            r["checkpoint_sha256"]=p["checkpoint_sha256"]; r["condition_sha256"]=p["condition_sha256"]
    grid["phase"]="BABY_HISTORICAL_SEED_VALIDATION_GRID"
    grid["seed_protocol"]=PROTOCOL
    grid["beta_training_seed_map"]={str(k):v for k,v in SEEDS.items()}
    hp=E/"historical_baby_validation_grid.json"; hp.write_text(json.dumps(grid,indent=2)+"\n")
    return grid

def main():
    by=audit()
    grid=annotate_grid(by)
    summary=json.loads((E/"baby_a2_full_grid_summary.json").read_text())
    summary["phase"]="BABY_HISTORICAL_SEED_FULL_BETA_VALIDATION_GRID"
    summary["seed_protocol"]=PROTOCOL; summary["beta_training_seed_map"]={str(k):v for k,v in SEEDS.items()}
    (E/"historical_baby_full_grid_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    if summary["eligible_count"] == 0:
        result={"phase":"BABY_HISTORICAL_SEED_RESCUE_FAIL","reason":"FULL_GRID_ELIGIBLE_ZERO",
                "seed_protocol":PROTOCOL,"BABY_DIFFUSION_TEST":"CLOSED","TEST_USED_FOR_SELECTION":False}
        (E/"BABY_HISTORICAL_SEED_RESCUE_FAIL.json").write_text(json.dumps(result,indent=2))
        print(json.dumps({"eligible_count":0,"result_phase":result["phase"],"reason":result["reason"],
                          "BABY_DIFFUSION_TEST":"CLOSED"},sort_keys=True))
        return
    pool=json.loads((E/"BABY_DIFFUSION_FROZEN_CANDIDATE_POOL.json").read_text())
    for c in pool["configs"]:
        if c.get("kind")=="DIFFUSION":
            b=float(c["beta"]); p=by[str(b)]
            c["training_seed"]=SEEDS[b]; c["seed_protocol"]=PROTOCOL
            c["checkpoint_sha256"]=p["checkpoint_sha256"]; c["condition_sha256"]=p["condition_sha256"]
    pool["phase"]="BABY_HISTORICAL_SEED_FROZEN_CANDIDATE_POOL"; pool["seed_protocol"]=PROTOCOL
    pp=E/"BABY_HISTORICAL_SEED_FROZEN_CANDIDATE_POOL.json"; pp.write_text(json.dumps(pool,indent=2)+"\n")
    cross=json.loads((E/"baby_a3_crossfit_summary.json").read_text())
    cross["seed_protocol"]=PROTOCOL
    (E/"historical_baby_crossfit_summary.json").write_text(json.dumps(cross,indent=2)+"\n")
    passed=bool(cross["BABY_DIFFUSION_UPGRADE_PASS"])
    if passed:
        elig=[r for r in grid["rows"][1:] if r["U"]>0 and r["primary_positive_count"]>=3 and r["sum_primary_delta"]>0]
        final=max(elig,key=lambda x:x["U"]); b=float(final["beta"])
        frozen={"phase":"BABY_HISTORICAL_SEED_VALIDATION_FROZEN_CONFIG","dataset":"baby",
                "seed_protocol":PROTOCOL,"selection_rule":"CROSSFIT_PASS_THEN_FULL_VALIDATION_MAX_U_ELIGIBLE",
                "final_config":final,"training_seed":SEEDS[b],
                "checkpoint_sha256":by[str(b)]["checkpoint_sha256"],
                "condition_sha256":by[str(b)]["condition_sha256"],
                "purification_seeds":NOISE,"crossfit_summary":cross,
                "candidate_pool_sha256":sha256(pp),"BABY_DIFFUSION_TEST":"CLOSED",
                "TEST_USED_FOR_SELECTION":False}
        if b==0.5:
            frozen["seed_robustness_preregistered_seeds"]=[20261501,20261502,20261503]
            frozen["NEXT_REQUIRED"]="SEED_ROBUSTNESS"
        else:
            frozen["NEXT_REQUIRED"]="ADVISOR_SEED_ROBUSTNESS_SCHEDULE_REQUIRED_FOR_NON_BETA_0P5"
        (E/"BABY_HISTORICAL_SEED_VALIDATION_FROZEN_CONFIG.json").write_text(json.dumps(frozen,indent=2)+"\n")
        result=frozen
    else:
        result={"phase":"BABY_HISTORICAL_SEED_RESCUE_FAIL","reason":"CROSSFIT_FAIL",
                "seed_protocol":PROTOCOL,"crossfit_summary":cross,
                "BABY_DIFFUSION_TEST":"CLOSED","TEST_USED_FOR_SELECTION":False}
        (E/"BABY_HISTORICAL_SEED_RESCUE_FAIL.json").write_text(json.dumps(result,indent=2)+"\n")
    for stale in ["BABY_A3_VALIDATION_FROZEN_CONFIG.json","BABY_DIFFUSION_CONDITION_SEARCH_FAIL.json"]:
        (E/stale).unlink(missing_ok=True)
    compact={"eligible_count":summary["eligible_count"],
             "beta_summary":{k:{"eligible":v["eligible_config_count"],"4of4":v["primary_4of4_count"],
                                "best_U":v["best_U"],"mean_U":v["mean_U"]} for k,v in summary["beta_summary"].items()},
             "crossfit":{"OOF_mean_delta_U":cross["OOF_mean_delta_U"],"P_delta_U_gt_0":cross["P_delta_U_gt_0"],
                         "positive":cross["OOF_positive_count"],"repeat_positive":cross["repeat_positive_mean_count"],
                         "PASS":passed},
             "result_phase":result["phase"],"BABY_DIFFUSION_TEST":"CLOSED"}
    print(json.dumps(compact,sort_keys=True))

if __name__=="__main__": main()
