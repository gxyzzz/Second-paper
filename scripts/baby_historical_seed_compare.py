from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from pipelines.dataset_config import load_dataset_config

OLD=Path("runs/diffusion_rescue/baby_condition_rescue")
NEW=Path("runs/diffusion_rescue/baby_historical_seed_replay")
PRIMARY=["R10","N10","R20","N20"]

def find_cfg(grid):
    xs=[r for r in grid["rows"] if r.get("kind")=="DIFFUSION"
        and abs(float(r["beta"])-0.5)<1e-12 and int(r["t_edit"])==3
        and abs(float(r["guidance"])-2.0)<1e-12
        and abs(float(r["rho_T"])-0.25)<1e-12 and abs(float(r["rho_V"])-1.0)<1e-12]
    if len(xs)!=1: raise RuntimeError(("config match",len(xs)))
    return xs[0]

def disp(raw_path,pur_path,batch=512):
    a=np.load(raw_path,mmap_mode="r",allow_pickle=False)
    b=np.load(pur_path,mmap_mode="r",allow_pickle=False)
    l2=[]; cs=[]
    for st in range(0,len(a),batch):
        x=np.asarray(a[st:st+batch],np.float32); y=np.asarray(b[st:st+batch],np.float32)
        x=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
        y=y/np.maximum(np.linalg.norm(y,axis=1,keepdims=True),1e-12)
        l2.append(np.linalg.norm(y-x,axis=1)); cs.append(np.sum(x*y,axis=1))
    l=np.concatenate(l2); c=np.concatenate(cs)
    return {"normalized_l2_mean":float(l.mean()),"normalized_l2_std":float(l.std()),
            "cosine_mean":float(c.mean()),"cosine_std":float(c.std())}

def train(meta):
    e=meta["epoch_log"][-1]; m=meta["mechanism"]
    return {"training_seed":meta["training_seed"],"checkpoint_sha256":meta["checkpoint_sha256"],
            "final_train_loss":e["train_loss"],"final_train_rec":e["train_rec"],
            "true_balanced":m["true_balanced"],"shuffled_balanced":m["shuf_balanced"],
            "null_balanced":m["null_balanced"],
            "TRUE_LT_SHUFFLED":m["true_better_shuffled_balanced"],
            "TRUE_LT_NULL":m["true_better_null_balanced"]}

def cross(z):
    if z is None:
        return {"OOF_mean_delta_U":None,"P_delta_U_gt_0":None,"positive_folds":None,
                "total_folds":None,"repeat_positive_mean_count":None,"PASS":False}
    return {"OOF_mean_delta_U":z["OOF_mean_delta_U"],"P_delta_U_gt_0":z["P_delta_U_gt_0"],
            "positive_folds":z["OOF_positive_count"],"total_folds":z["OOF_total"],
            "repeat_positive_mean_count":z["repeat_positive_mean_count"],
            "PASS":z["BABY_DIFFUSION_UPGRADE_PASS"]}

def main():
    og=json.loads((OLD/"evidence/baby_validation_grid.json").read_text())
    ng=json.loads((NEW/"evidence/historical_baby_validation_grid.json").read_text())
    os=json.loads((OLD/"evidence/baby_a2_full_grid_summary.json").read_text())
    ns=json.loads((NEW/"evidence/historical_baby_full_grid_summary.json").read_text())
    oc=json.loads((OLD/"evidence/baby_a3_crossfit_summary.json").read_text())
    ncp=NEW/"evidence/historical_baby_crossfit_summary.json"
    nc=json.loads(ncp.read_text()) if ncp.exists() else None
    om=json.loads((OLD/"evidence/beta_0p5_training.json").read_text())
    nm=json.loads((NEW/"beta_0p5/evidence/beta_0p5_training.json").read_text())
    o=find_cfg(og); n=find_cfg(ng)
    cfg=load_dataset_config("baby"); paths=cfg["resolved_paths"]
    out={"phase":"BABY_DIFFUSION_SEED_COMPARISON","dataset":"baby",
         "fixed_config":{"beta":0.5,"t_edit":3,"guidance":2.0,"rho_T":0.25,"rho_V":1.0,
                         "purification_seeds":[20261001,20261002,20261003,20261004]},
         "seed_20261101":{"training":train(om),
            "ranking":{"metrics":o["metrics"],"delta_vs_full_coliftrec":o["delta_vs_no_diffusion"],
                       "U":o["U"],"primary_positive_count":o["primary_positive_count"]},
            "purified_displacement":{"text":disp(paths["text_feature"],OLD/"purified/beta_0p5/t3_g2p0_text.npy"),
                                     "visual":disp(paths["visual_feature"],OLD/"purified/beta_0p5/t3_g2p0_visual.npy")},
            "eligible_count":os["eligible_count"],"crossfit":cross(oc)},
         "seed_20261501":{"training":train(nm),
            "ranking":{"metrics":n["metrics"],"delta_vs_full_coliftrec":n["delta_vs_no_diffusion"],
                       "U":n["U"],"primary_positive_count":n["primary_positive_count"]},
            "purified_displacement":{"text":disp(paths["text_feature"],NEW/"purified/beta_0p5/t3_g2p0_text.npy"),
                                     "visual":disp(paths["visual_feature"],NEW/"purified/beta_0p5/t3_g2p0_visual.npy")},
            "eligible_count":ns["eligible_count"],"crossfit":cross(nc)},
         "historical_minus_old":{"R10":float(n["metrics"]["R10"]-o["metrics"]["R10"]),
                                 "N10":float(n["metrics"]["N10"]-o["metrics"]["N10"]),
                                 "R20":float(n["metrics"]["R20"]-o["metrics"]["R20"]),
                                 "N20":float(n["metrics"]["N20"]-o["metrics"]["N20"]),
                                 "U":float(n["U"]-o["U"])},
         "VALIDATION_ONLY":True,"TEST_ACCESSED":False}
    (NEW/"evidence/baby_diffusion_seed_comparison.json").write_text(json.dumps(out,indent=2))
    p=Path("docs/evidence/paper/baby_diffusion_seed_comparison.json")
    p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(out,indent=2))
    print(json.dumps({"historical_minus_old":out["historical_minus_old"],
                      "old_crossfit":out["seed_20261101"]["crossfit"],
                      "historical_crossfit":out["seed_20261501"]["crossfit"]},sort_keys=True))

if __name__=="__main__": main()
