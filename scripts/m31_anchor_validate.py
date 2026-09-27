from __future__ import annotations
import argparse, json, hashlib, gc
from pathlib import Path
import numpy as np
import torch

from modules.diffusion import NativeTVX0Denoiser, purify_indices
from modules.ranking import metrics_at, rank_by_score
from pipelines.dataset_config import load_dataset_config
from diffusion_validate import build_context, semantic_lift, lift_for_rho

ROOT=Path(__file__).resolve().parents[1]
PRIMARY=["R10","N10","R20","N20"]
ALL=PRIMARY+["R50","N50"]
DEVICE="cuda"

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for x in iter(lambda:f.read(1<<20),b""): h.update(x)
    return h.hexdigest()

def btag(x): return ("%.1f"%float(x)).replace(".","p")

def load_model(diff_dir,dataset,beta):
    p=Path(diff_dir)/"checkpoints"/f"{dataset}_beta_{btag(beta)}.pt"
    z=torch.load(p,map_location="cpu",weights_only=False)
    m=NativeTVX0Denoiser(int(z["D"]),cond_dim=64,hidden=int(z["hidden"]),time_dim=64).to(DEVICE)
    m.load_state_dict(z["state_dict"],strict=True); m.eval()
    return m,p,z

def run(dataset,msca_assets,coliftrec_dir,diff_dir):
    cfg=load_dataset_config(dataset); ds=cfg["dataset"]; dcfg=cfg["diffusion"]
    if dcfg["training_protocol"]!="m31_fixed_all_items":
        raise RuntimeError("anchor validation requires m31_fixed_all_items")
    beta=float(dcfg["beta_candidates"][0]); t=int(dcfg["t_edit_candidates"][0]); g=float(dcfg["guidance_candidates"][0])
    rho_t=float(dcfg["rho_text_candidates"][0]); rho_v=float(dcfg["rho_visual_candidates"][0])
    seeds=[int(x) for x in dcfg["purification_seeds"]]
    if not (beta==1.0 and t==5 and g==1.5 and len(seeds)==4):
        raise RuntimeError("M31 stable anchor config changed")
    diff=Path(diff_dir); evid=diff/"evidence"; pur=diff/"purified_anchor"; tmp=diff/"tmp_anchor"
    evid.mkdir(parents=True,exist_ok=True); pur.mkdir(parents=True,exist_ok=True); tmp.mkdir(parents=True,exist_ok=True)

    train_meta=json.loads((evid/f"beta_{btag(beta)}_training.json").read_text())
    required={
        "training_protocol":"m31_fixed_all_items",
        "train_scope":"all_catalog_items",
        "checkpoint_selection":"final_epoch",
        "training_seed":20261101,
        "epochs":80,
        "final_epoch":80,
        "VALIDATION_TARGET_USED":False,
        "TEST_TARGET_USED":False,
        "ALL_CATALOG_SIDE_INFORMATION":True,
        "TEST_ACCESSED":False,
    }
    for k,v in required.items():
        if train_meta.get(k)!=v: raise RuntimeError(("training protocol mismatch",k,train_meta.get(k),v))

    model,ck,z=load_model(diff,ds,beta)
    if sha256(ck)!=train_meta["checkpoint_sha256"]: raise RuntimeError("checkpoint sha mismatch")
    paths=cfg["resolved_paths"]
    raw_t=np.load(paths["text_feature"],mmap_mode="r",allow_pickle=False)
    raw_v=np.load(paths["visual_feature"],mmap_mode="r",allow_pickle=False)
    cond=np.load(diff/"assets"/f"condition_beta_{btag(beta)}.npy",mmap_mode="r",allow_pickle=False)
    ids=np.arange(len(raw_t),dtype=np.int64)
    tp=pur/f"beta_{btag(beta)}_t{t}_g{str(g).replace('.','p')}_text.npy"
    vp=pur/f"beta_{btag(beta)}_t{t}_g{str(g).replace('.','p')}_visual.npy"
    if not (tp.exists() and vp.exists()):
        ot,ov=purify_indices(model,raw_t,raw_v,cond,ids,t_edit=t,guidance=g,seeds=tuple(seeds),batch=256,device=DEVICE)
        np.save(tp,ot.astype(np.float32)); np.save(vp,ov.astype(np.float32))
        del ot,ov; torch.cuda.empty_cache(); gc.collect()

    ctx=build_context(ds,Path(msca_assets),Path(coliftrec_dir))
    raw_lt=semantic_lift(paths["text_feature"],ctx,"text")
    raw_lv=semantic_lift(paths["visual_feature"],ctx,"visual")
    lt=lift_for_rho(paths["text_feature"],tp,rho_t,ctx,"text",tmp/"text_blend.npy")
    lv=lift_for_rho(paths["visual_feature"],vp,rho_v,ctx,"visual",tmp/"visual_blend.npy")
    at=float(cfg["coliftrec"]["text"]["alpha"]); av=float(cfg["coliftrec"]["visual"]["alpha"])
    score=ctx["base_score"].copy()
    score+=at*(lt-raw_lt); score+=av*(lv-raw_lv)
    ranked=rank_by_score(ctx["items"],score)
    metrics=metrics_at(ranked,ctx["users"],ctx["eval_sets"])
    base_metrics=metrics_at(ctx["base_ranked"],ctx["users"],ctx["eval_sets"])
    delta={k:float(metrics[k]-base_metrics[k]) for k in ALL}
    U=float(np.mean([delta[k]/base_metrics[k] for k in PRIMARY]))
    primary_pos=int(sum(delta[k]>0 for k in PRIMARY))
    sum_primary=float(sum(delta[k] for k in PRIMARY))
    passed=bool(U>0 and primary_pos>=3 and sum_primary>0)
    result={
        "phase":f"{ds.upper()}_M31_CURRENT_RUN_STABLE_ANCHOR",
        "dataset":ds,
        "base_metrics":base_metrics,
        "metrics":metrics,
        "delta_vs_full_coliftrec":delta,
        "U":U,
        "primary_positive_count":primary_pos,
        "sum_primary_delta":sum_primary,
        "anchor":{"beta":beta,"t_edit":t,"guidance":g,"rho_T":rho_t,"rho_V":rho_v},
        "training_protocol":"m31_fixed_all_items",
        "train_scope":"all_catalog_items",
        "training_seed":int(dcfg["training_seed"]),
        "purification_seeds":seeds,
        "diffusion_checkpoint_sha256":sha256(ck),
        "text_sha256":sha256(tp),
        "visual_sha256":sha256(vp),
        "VALIDATION_TARGET_USED_FOR_TRAINING":False,
        "TEST_TARGET_USED":False,
        "TEST_USED_FOR_SELECTION":False,
        f"{ds.upper()}_M31_CURRENT_RUN_REPRO":"PASS" if passed else "FAIL",
    }
    out=evid/f"{ds}_m31_current_run_anchor.json"
    out.write_text(json.dumps(result,indent=2)+"\n")

    freeze={
        "phase":f"{ds.upper()}_DIFFUSION_FROZEN_BEFORE_TEST",
        "dataset":ds,
        "status":"FROZEN_STABLE_M31_ANCHOR" if passed else "CURRENT_RUN_ANCHOR_FAIL",
        "msca_checkpoint_sha256":train_meta["source_msca_checkpoint_sha256"],
        "coliftrec_config":cfg["coliftrec"],
        "diffusion_checkpoint_sha256":sha256(ck),
        "training_protocol":"m31_fixed_all_items",
        "train_item_definition":"all catalog items np.arange(n_items)",
        "training_seed":int(dcfg["training_seed"]),
        "purification_seeds":seeds,
        "beta":beta,"t_edit":t,"guidance":g,"rho_T":rho_t,"rho_V":rho_v,
        "validation_metrics":metrics,
        "delta_vs_full_coliftrec":delta,
        "U":U,
        "primary_positive_count":primary_pos,
        "sum_primary_delta":sum_primary,
        "TEST_USED_FOR_SELECTION":False,
        f"{ds.upper()}_DIFFUSION_TEST":"CLOSED",
    }
    (evid/f"{ds.upper()}_DIFFUSION_FROZEN_BEFORE_TEST.json").write_text(json.dumps(freeze,indent=2)+"\n")
    if not passed:
        result["DIFFUSION_BACKBONE_CHECKPOINT_SENSITIVITY"]="OBSERVED"
        out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"dataset":ds,"PASS":passed,"U":U,"primary_positive_count":primary_pos,
                      "sum_primary_delta":sum_primary,"metrics":metrics,"delta":delta,
                      f"{ds.upper()}_DIFFUSION_TEST":"CLOSED"},sort_keys=True),flush=True)
    return result

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset",required=True)
    ap.add_argument("--msca-assets",required=True)
    ap.add_argument("--coliftrec-dir",required=True)
    ap.add_argument("--diffusion-dir",required=True)
    a=ap.parse_args()
    run(a.dataset,a.msca_assets,a.coliftrec_dir,a.diffusion_dir)
