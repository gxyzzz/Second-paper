from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch

from modules.diffusion import NativeTVX0Denoiser, purify_indices
from modules.ranking import metrics_at, rank_by_score, sha256_file
from pipelines.dataset_config import load_dataset_config
from diffusion_validate import (
    PRIMARY, ALL, build_context, semantic_lift, lift_for_rho,
    per_user_primary, metrics_from_per, sha256
)

BETA=0.5
T_EDIT=3
GUIDANCE=2.0
RHO_T=0.25
RHO_V=1.0
DIFFUSION_SEED=20261501
PURIFICATION_SEEDS=[20261001,20261002,20261003,20261004]
BOOTSTRAP_SEED=20262001
BOOTSTRAP_N=1000
EXPECTED_COLIFT={
    "lambda_text":1.0,"lambda_attribute":0.75,"lambda_visual":0.25,
    "alpha_text":0.25,"alpha_attribute":0.15,"alpha_visual":0.025,
}

def _load_model(diff_dir:Path):
    ck=diff_dir/"checkpoints"/"baby_beta_0p5.pt"
    z=torch.load(ck,map_location="cpu",weights_only=False)
    m=NativeTVX0Denoiser(int(z["D"]),cond_dim=64,hidden=int(z["hidden"]),time_dim=64).cuda()
    m.load_state_dict(z["state_dict"],strict=True); m.eval()
    return m,ck,z

def _bootstrap(base_per,diff_per):
    n=len(base_per); rng=np.random.RandomState(BOOTSTRAP_SEED); vals=np.empty(BOOTSTRAP_N,np.float64)
    for i in range(BOOTSTRAP_N):
        idx=rng.randint(0,n,size=n)
        bm=np.asarray(base_per[idx]).mean(axis=0)
        dm=np.asarray(diff_per[idx]).mean(axis=0)-bm
        vals[i]=float(np.mean(dm/bm))
    return {
        "resamples":BOOTSTRAP_N,"seed":BOOTSTRAP_SEED,
        "mean_delta_U":float(vals.mean()),
        "std_delta_U":float(vals.std(ddof=1)),
        "ci95":[float(np.quantile(vals,.025)),float(np.quantile(vals,.975))],
        "P_delta_U_gt_0":float(np.mean(vals>0)),
    }

def run(seed:int,msca_assets:Path,coliftrec_dir:Path,diff_dir:Path,out_dir:Path):
    out_dir.mkdir(parents=True,exist_ok=True)
    evid=out_dir/"evidence"; pur=out_dir/"purified"; tmp=out_dir/"tmp"
    for p in (evid,pur,tmp): p.mkdir(parents=True,exist_ok=True)

    ma=json.loads((msca_assets/"audit.json").read_text())
    if ma.get("TEST_ACCESSED") is not False or int(ma["seed"])!=int(seed):
        raise RuntimeError(("MSCA provenance",ma.get("seed"),seed,ma.get("TEST_ACCESSED")))
    cs=json.loads((coliftrec_dir/"summary.json").read_text())
    if cs.get("TEST_ACCESSED") is not False or cs["source_checkpoint_sha256"]!=ma["checkpoint_sha256"]:
        raise RuntimeError("CoLiftRec/MSCA provenance mismatch")
    if cs["config"]!=EXPECTED_COLIFT:
        raise RuntimeError(("CoLiftRec config changed",cs["config"],EXPECTED_COLIFT))

    dm=json.loads((diff_dir/"evidence"/"beta_0p5_training.json").read_text())
    required={
        "beta":BETA,"training_seed":DIFFUSION_SEED,"epochs":80,"final_epoch":80,
        "train_item_count":7050,"training_protocol":"m31_fixed_all_items",
        "train_scope":"all_catalog_items","checkpoint_selection":"final_epoch",
        "lambda_ctr":0.1,"p_uncond":0.15,"optimizer":"AdamW","lr":0.001,"weight_decay":0.0001,
        "VALIDATION_TARGET_USED":False,"TEST_TARGET_USED":False,"TEST_ACCESSED":False,
    }
    for k,v in required.items():
        if dm.get(k)!=v: raise RuntimeError(("Diffusion provenance",k,dm.get(k),v))
    if dm["source_msca_checkpoint_sha256"]!=ma["checkpoint_sha256"]:
        raise RuntimeError("Diffusion/MSCA checkpoint mismatch")

    cfg=load_dataset_config("baby"); paths=cfg["resolved_paths"]
    raw_t=np.load(paths["text_feature"],mmap_mode="r",allow_pickle=False)
    raw_v=np.load(paths["visual_feature"],mmap_mode="r",allow_pickle=False)
    condp=diff_dir/"assets"/"condition_beta_0p5.npy"
    cond=np.load(condp,mmap_mode="r",allow_pickle=False)
    model,ck,z=_load_model(diff_dir)
    if sha256(ck)!=dm["checkpoint_sha256"] or sha256(condp)!=dm["condition_sha256"]:
        raise RuntimeError("Diffusion SHA mismatch")
    ids=np.arange(len(raw_t),dtype=np.int64)
    tp=pur/"fixed_text.npy"; vp=pur/"fixed_visual.npy"
    ot,ov=purify_indices(model,raw_t,raw_v,cond,ids,t_edit=T_EDIT,guidance=GUIDANCE,
                         seeds=tuple(PURIFICATION_SEEDS),batch=256,device="cuda")
    np.save(tp,ot.astype(np.float32)); np.save(vp,ov.astype(np.float32))
    del ot,ov,model; torch.cuda.empty_cache()

    ctx=build_context("baby",msca_assets,coliftrec_dir)
    raw_lt=semantic_lift(paths["text_feature"],ctx,"text")
    raw_lv=semantic_lift(paths["visual_feature"],ctx,"visual")
    lt=lift_for_rho(paths["text_feature"],tp,RHO_T,ctx,"text",tmp/"blend_t.npy")
    lv=lift_for_rho(paths["visual_feature"],vp,RHO_V,ctx,"visual",tmp/"blend_v.npy")
    score=ctx["base_score"].copy()
    score += float(cfg["coliftrec"]["text"]["alpha"])*(lt-raw_lt)
    score += float(cfg["coliftrec"]["visual"]["alpha"])*(lv-raw_lv)
    ranked=rank_by_score(ctx["items"],score)
    diff_metrics=metrics_at(ranked,ctx["users"],ctx["eval_sets"])
    colift_metrics=metrics_at(ctx["base_ranked"],ctx["users"],ctx["eval_sets"])
    msca_metrics={k:float(ma["validation_metrics"][k]) for k in ALL}

    d_diff={k:float(diff_metrics[k]-colift_metrics[k]) for k in ALL}
    d_colift={k:float(colift_metrics[k]-msca_metrics[k]) for k in ALL}
    U=float(np.mean([d_diff[k]/colift_metrics[k] for k in PRIMARY]))
    pos=int(sum(d_diff[k]>0 for k in PRIMARY)); sm=float(sum(d_diff[k] for k in PRIMARY))
    base_per=per_user_primary(ctx["base_ranked"],ctx["users"],ctx["eval_sets"])
    diff_per=per_user_primary(ranked,ctx["users"],ctx["eval_sets"])
    # Exactness guard.
    for k,v in metrics_from_per(base_per).items():
        if abs(v-colift_metrics[k])>1e-10: raise RuntimeError(("base per exactness",k,v,colift_metrics[k]))
    for k,v in metrics_from_per(diff_per).items():
        if abs(v-diff_metrics[k])>1e-10: raise RuntimeError(("diff per exactness",k,v,diff_metrics[k]))
    boot=_bootstrap(base_per,diff_per)
    result={
        "phase":"BABY_BACKBONE_ROBUSTNESS_FIXED_DIFFUSION_VALIDATION",
        "dataset":"baby","msca_seed":int(seed),
        "MSCA":{"best_epoch":int(ma["checkpoint_epoch"]),"checkpoint_sha256":ma["checkpoint_sha256"],"metrics":msca_metrics},
        "COLIFTREC":{
            "config":EXPECTED_COLIFT,"metrics":colift_metrics,"delta_vs_msca":d_colift,
            "primary_positive_count":int(sum(d_colift[k]>0 for k in PRIMARY)),
            "sum_primary_delta":float(sum(d_colift[k] for k in PRIMARY)),
        },
        "DIFFUSION":{
            "beta":BETA,"training_seed":DIFFUSION_SEED,"checkpoint_sha256":dm["checkpoint_sha256"],
            "condition_sha256":dm["condition_sha256"],"t_edit":T_EDIT,"guidance":GUIDANCE,
            "rho_T":RHO_T,"rho_V":RHO_V,"purification_seeds":PURIFICATION_SEEDS,
            "purified_text_sha256":sha256(tp),"purified_visual_sha256":sha256(vp),
            "metrics":diff_metrics,"delta_vs_coliftrec":d_diff,"U":U,
            "primary_positive_count":pos,"sum_primary_delta":sm,
            "primary_4of4":bool(pos==4),"PASS":bool(U>0 and pos>=3 and sm>0),
            "bootstrap":boot,
        },
        "native_feature_sha":{
            "text":sha256_file(paths["text_feature"]),
            "visual":sha256_file(paths["visual_feature"]),
            "interaction":sha256_file(paths["interaction"]),
        },
        "VALIDATION_ONLY":True,"TEST_ACCESSED":False,"TEST_USED_FOR_SELECTION":False,
    }
    (evid/"fixed_diffusion_validation.json").write_text(json.dumps(result,indent=2)+"\n")
    np.savez_compressed(evid/"fixed_diffusion_per_user.npz",base=base_per,diffusion=diff_per,users=ctx["users"])
    print(json.dumps({
        "seed":seed,"msca_R20":msca_metrics["R20"],"colift_R20":colift_metrics["R20"],
        "diff_R20":diff_metrics["R20"],"colift_primary":result["COLIFTREC"]["primary_positive_count"],
        "diff_primary":pos,"U":U,"PASS":result["DIFFUSION"]["PASS"],"bootstrap":boot,
    },sort_keys=True),flush=True)
    return result

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--seed",type=int,required=True)
    ap.add_argument("--msca-assets",required=True)
    ap.add_argument("--coliftrec-dir",required=True)
    ap.add_argument("--diffusion-dir",required=True)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    run(a.seed,Path(a.msca_assets),Path(a.coliftrec_dir),Path(a.diffusion_dir),Path(a.out))
