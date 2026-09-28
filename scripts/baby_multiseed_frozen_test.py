from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch

from frozen_diffusion_test import build_test_eval, params, make_blend, semantic_lift, sha256
from diffusion_validate import per_user_primary, metrics_from_per
from modules.attribute import attribute_z, build_item_matrices, build_profiles
from modules.coliftrec import fit_backgrounds, score_coliftrec
from modules.ranking import metrics_at, rank_by_score, semantic_z_for_candidates, topk_from_embeddings
from pipelines.dataset_config import load_dataset_config
from pipelines.msca_assets import build_train_histories_and_validation

PRIMARY=("R10","N10","R20","N20")
ALL=("R10","N10","R20","N20","R50","N50")
SEEDS=(999,1000,1001,1002)
BOOTSTRAP_SEED=20262002
BOOTSTRAP_N=1000
EXPECTED_COLIFT={
    "lambda_text":1.0,"lambda_attribute":0.75,"lambda_visual":0.25,
    "alpha_text":0.25,"alpha_attribute":0.15,"alpha_visual":0.025,
}
EXPECTED_DIFF={
    "beta":0.5,"training_seed":20261501,"t_edit":3,"guidance":2.0,
    "rho_T":0.25,"rho_V":1.0,
    "purification_seeds":[20261001,20261002,20261003,20261004],
}

def seed_paths(seed:int):
    root=Path(f"runs/backbone_robustness/baby_seed{seed}")
    if seed==999:
        return root,Path("runs/assets/msca_baby_seed999"),Path("runs/generic_refactor/baby_formal"),Path("runs/diffusion_rescue/baby_historical_seed_replay/beta_0p5")
    return root,root/"msca/assets",root/"coliftrec",root/"diffusion"

def provenance(seed:int,freeze_path:Path):
    root,assets,colift,diff=seed_paths(seed)
    freeze=json.loads(freeze_path.read_text())
    if freeze["canonical_seed"]!=1000 or freeze["selection_does_not_use_test"] is not True:
        raise RuntimeError("canonical pre-Test freeze invalid")
    if list(freeze["candidate_seeds"])!=list(SEEDS):
        raise RuntimeError("seed set differs from pre-Test freeze")
    va=json.loads((root/"evidence/fixed_diffusion_validation.json").read_text())
    aa=json.loads((assets/"audit.json").read_text())
    cs=json.loads((colift/"summary.json").read_text())
    tr=json.loads((diff/"evidence/beta_0p5_training.json").read_text())
    expected=freeze["all_seed_checkpoints"][str(seed)]
    if int(aa["seed"])!=seed or int(aa["checkpoint_epoch"])!=int(expected["best_epoch"]):
        raise RuntimeError(("MSCA seed/epoch mismatch",seed,aa["seed"],aa["checkpoint_epoch"]))
    if aa["checkpoint_sha256"]!=expected["checkpoint_sha256"] or va["MSCA"]["checkpoint_sha256"]!=expected["checkpoint_sha256"]:
        raise RuntimeError("MSCA SHA mismatch")
    if aa.get("TEST_ACCESSED") is not False:
        raise RuntimeError("MSCA asset set was not Validation-only")
    if cs["source_checkpoint_sha256"]!=expected["checkpoint_sha256"] or cs["config"]!=EXPECTED_COLIFT:
        raise RuntimeError("CoLiftRec provenance/config mismatch")
    if cs.get("TEST_ACCESSED") is not False or cs.get("TEST_USED_FOR_SELECTION") is not False:
        raise RuntimeError("CoLiftRec Validation provenance invalid")
    d=va["DIFFUSION"]
    for k,v in EXPECTED_DIFF.items():
        if d[k]!=v: raise RuntimeError(("fixed Diffusion config mismatch",k,d[k],v))
    if tr["source_msca_checkpoint_sha256"]!=expected["checkpoint_sha256"]:
        raise RuntimeError("Diffusion source MSCA mismatch")
    for k,v in {"beta":0.5,"training_seed":20261501,"epochs":80,"final_epoch":80,
                "training_protocol":"m31_fixed_all_items","checkpoint_selection":"final_epoch",
                "lambda_ctr":0.1,"p_uncond":0.15,"optimizer":"AdamW",
                "lr":0.001,"weight_decay":0.0001,"TEST_ACCESSED":False}.items():
        if tr.get(k)!=v: raise RuntimeError(("Diffusion training provenance mismatch",k,tr.get(k),v))
    ck=diff/"checkpoints/baby_beta_0p5.pt"
    cond=diff/"assets/condition_beta_0p5.npy"
    tp=root/"purified/fixed_text.npy"; vp=root/"purified/fixed_visual.npy"
    for p in (ck,cond,tp,vp):
        if not p.exists(): raise FileNotFoundError(p)
    if sha256(ck)!=tr["checkpoint_sha256"] or sha256(ck)!=d["checkpoint_sha256"]:
        raise RuntimeError("Diffusion checkpoint SHA mismatch")
    if sha256(cond)!=tr["condition_sha256"] or sha256(cond)!=d["condition_sha256"]:
        raise RuntimeError("condition SHA mismatch")
    if sha256(tp)!=d["purified_text_sha256"] or sha256(vp)!=d["purified_visual_sha256"]:
        raise RuntimeError("fixed purified feature SHA mismatch")
    return root,assets,colift,diff,aa,va,tr,tp,vp

def delta_block(a:dict,b:dict):
    absolute={k:float(b[k]-a[k]) for k in ALL}
    relative={k:float(absolute[k]/a[k]) for k in ALL}
    return {
        "absolute":absolute,"relative":relative,
        "primary_positive_count":int(sum(absolute[k]>0 for k in PRIMARY)),
        "overall_positive_count":int(sum(absolute[k]>0 for k in ALL)),
        "sum_primary_delta":float(sum(absolute[k] for k in PRIMARY)),
        "mean_relative_primary_gain":float(np.mean([relative[k] for k in PRIMARY])),
        "primary_4of4":bool(all(absolute[k]>0 for k in PRIMARY)),
        "all_6of6":bool(all(absolute[k]>0 for k in ALL)),
    }

def paired_bootstrap(base:np.ndarray,cand:np.ndarray):
    if base.shape!=cand.shape or base.shape[1]!=4: raise RuntimeError("bootstrap shape mismatch")
    obs=np.asarray(cand).mean(0)-np.asarray(base).mean(0)
    rng=np.random.RandomState(BOOTSTRAP_SEED)
    vals=np.empty((BOOTSTRAP_N,4),dtype=np.float64)
    u=np.empty(BOOTSTRAP_N,dtype=np.float64)
    n=len(base)
    for i in range(BOOTSTRAP_N):
        idx=rng.randint(0,n,size=n)
        bm=np.asarray(base[idx]).mean(0); cm=np.asarray(cand[idx]).mean(0)
        vals[i]=cm-bm
        u[i]=np.mean((cm-bm)/bm)
    out={"resamples":BOOTSTRAP_N,"seed":BOOTSTRAP_SEED,"paired_user_resampling":True,"metrics":{}}
    for j,k in enumerate(PRIMARY):
        out["metrics"][k]={
            "observed_delta":float(obs[j]),
            "bootstrap_mean_delta":float(vals[:,j].mean()),
            "ci95":[float(np.quantile(vals[:,j],.025)),float(np.quantile(vals[:,j],.975))],
            "P_delta_gt_0":float(np.mean(vals[:,j]>0)),
        }
    out["mean_relative_primary_gain"]={
        "bootstrap_mean":float(u.mean()),
        "ci95":[float(np.quantile(u,.025)),float(np.quantile(u,.975))],
        "P_gt_0":float(np.mean(u>0)),
    }
    return out

def run(seed:int,freeze_path:Path,out_dir:Path):
    if seed not in SEEDS: raise RuntimeError("seed not in frozen four-seed set")
    marker=out_dir/"TEST_RUN_COMPLETED"
    if marker.exists(): raise RuntimeError(f"Baby seed{seed} multiseed Test already executed; refusing second run")
    root,assets,colift,diff,aa,va,tr,tp,vp=provenance(seed,freeze_path)
    cfg=load_dataset_config("baby"); paths=cfg["resolved_paths"]; ccfg=cfg["coliftrec"]
    n_users=int(aa["n_users"]); n_items=int(aa["n_items"])

    # First Test target access occurs only after every frozen provenance guard above passed.
    histories,pseudo_hist,expected_pseudo_users,_,_=build_train_histories_and_validation(paths["interaction"],n_users)
    histories2,test_users,test_sets=build_test_eval(paths["interaction"],n_users)
    if histories!=histories2: raise RuntimeError("TRAIN history mismatch")

    pseudo=np.load(assets/"train_pseudo_top100.npz")
    pseudo_users=pseudo["users"].astype(np.int64); pseudo_items=pseudo["items"].astype(np.int32)
    if not np.array_equal(pseudo_users,expected_pseudo_users): raise RuntimeError("pseudo user order mismatch")

    emb=np.load(assets/"embeddings.npz")
    fu=torch.as_tensor(emb["final_user"],device="cuda")
    fi=torch.as_tensor(emb["final_item"],device="cuda")
    test_items,test_scores=topk_from_embeddings(
        fu,fi,test_users,histories,top_l=int(ccfg.get("top_l",100)),batch_users=1024
    )
    del fu,fi; torch.cuda.empty_cache()

    p=params(ccfg); enabled={m:bool(ccfg[m]["enabled"]) for m in ("text","attribute","visual")}
    raw_t=paths["text_feature"]; raw_v=paths["visual_feature"]
    zt_tr,_=semantic_z_for_candidates(raw_t,pseudo_hist,pseudo_users,pseudo_items,batch_users=256)
    zt_te,_=semantic_z_for_candidates(raw_t,histories,test_users,test_items,batch_users=256)
    zv_tr,_=semantic_z_for_candidates(raw_v,pseudo_hist,pseudo_users,pseudo_items,batch_users=128)
    zv_te,_=semantic_z_for_candidates(raw_v,histories,test_users,test_items,batch_users=128)

    acfg=ccfg["attribute"]
    mats,_=build_item_matrices(
        paths["metadata"],n_items,
        min_df=int(acfg.get("tfidf_min_df",2)),
        max_df=float(acfg.get("tfidf_max_df",.8)),
        description_len=int(acfg.get("description_len",128)),
    )
    prof=build_profiles(mats,histories,n_items)
    pprof=build_profiles(mats,pseudo_hist,n_items)
    za_tr,_=attribute_z(mats,pprof,pseudo_users,pseudo_items,batch=256)
    za_te,_=attribute_z(mats,prof,test_users,test_items,batch=256)
    bgs=fit_backgrounds(pseudo_items,zt_tr,za_tr,zv_tr,n_items)
    full_score,_=score_coliftrec(test_scores,test_items,zt_te,za_te,zv_te,bgs,p,enabled=enabled)

    msca_rank=test_items
    full_rank=rank_by_score(test_items,full_score)
    msca_metrics=metrics_at(msca_rank,test_users,test_sets)
    full_metrics=metrics_at(full_rank,test_users,test_sets)

    tmp=out_dir/"tmp"; tmp.mkdir(parents=True,exist_ok=True)
    bt=tmp/"blend_text.npy"; bv=tmp/"blend_visual.npy"
    make_blend(raw_t,tp,.25,bt)
    make_blend(raw_v,vp,1.0,bv)
    raw_lt=semantic_lift(raw_t,pseudo_hist,pseudo_users,pseudo_items,histories,test_users,test_items,n_items,p.lambda_text,128)
    raw_lv=semantic_lift(raw_v,pseudo_hist,pseudo_users,pseudo_items,histories,test_users,test_items,n_items,p.lambda_visual,128)
    dif_lt=semantic_lift(bt,pseudo_hist,pseudo_users,pseudo_items,histories,test_users,test_items,n_items,p.lambda_text,128)
    dif_lv=semantic_lift(bv,pseudo_hist,pseudo_users,pseudo_items,histories,test_users,test_items,n_items,p.lambda_visual,128)
    diff_score=full_score.copy()
    diff_score+=p.alpha_text*(dif_lt-raw_lt)+p.alpha_visual*(dif_lv-raw_lv)
    diff_rank=rank_by_score(test_items,diff_score)
    diff_metrics=metrics_at(diff_rank,test_users,test_sets)

    msca_per=per_user_primary(msca_rank,test_users,test_sets)
    full_per=per_user_primary(full_rank,test_users,test_sets)
    diff_per=per_user_primary(diff_rank,test_users,test_sets)
    for name,arr,met in (
        ("MSCA",msca_per,msca_metrics),
        ("CoLiftRec",full_per,full_metrics),
        ("Diffusion",diff_per,diff_metrics),
    ):
        mm=metrics_from_per(arr)
        for k in PRIMARY:
            if abs(mm[k]-met[k])>1e-10:
                raise RuntimeError(("per-user exactness",name,k,mm[k],met[k]))

    colift_delta=delta_block(msca_metrics,full_metrics)
    diff_delta=delta_block(full_metrics,diff_metrics)
    diff_pass=bool(diff_delta["mean_relative_primary_gain"]>0 and diff_delta["primary_positive_count"]>=3)
    out={
      "phase":"BABY_MULTI_BACKBONE_FROZEN_TEST","dataset":"baby","msca_seed":seed,
      "canonical_backbone":bool(seed==1000),
      "canonical_freeze_file":str(freeze_path),"canonical_freeze_sha256":sha256(freeze_path),
      "MSCA":{
          "best_epoch":int(aa["checkpoint_epoch"]),
          "checkpoint_sha256":aa["checkpoint_sha256"],
          "metrics":msca_metrics,
      },
      "COLIFTREC":{
          "config":EXPECTED_COLIFT,
          "metrics":full_metrics,
          "vs_MSCA":colift_delta,
          "bootstrap_vs_MSCA":paired_bootstrap(msca_per,full_per),
      },
      "DIFFUSION":{
          "config":EXPECTED_DIFF,
          "checkpoint_sha256":tr["checkpoint_sha256"],
          "condition_sha256":tr["condition_sha256"],
          "metrics":diff_metrics,
          "vs_COLIFTREC":diff_delta,
          "bootstrap_vs_COLIFTREC":paired_bootstrap(full_per,diff_per),
          "TEST_PASS":diff_pass,
          "TEST_STATUS":"STRONG_PASS" if diff_pass and diff_delta["primary_4of4"] else ("PASS" if diff_pass else "FAIL"),
      },
      "BABY_DIFFUSION_TEST_RUN_COUNT":1,
      "TEST_USED_FOR_SELECTION":False,
      "NO_POST_TEST_TUNING":True,
      "BABY_METHOD_SEARCH":"CLOSED",
      "BABY_PARAMETER_SEARCH":"CLOSED",
      "BABY_SEED_SEARCH":"CLOSED",
    }
    out_dir.mkdir(parents=True,exist_ok=True)
    (out_dir/"summary.json").write_text(json.dumps(out,indent=2)+"\n")
    np.savez_compressed(
        out_dir/"test_scores.npz",
        users=test_users,items=test_items,msca=test_scores,
        full_coliftrec=full_score,diffusion=diff_score,
    )
    np.savez_compressed(
        out_dir/"test_per_user_primary.npz",
        users=test_users,msca=msca_per,coliftrec=full_per,diffusion=diff_per,
    )
    marker.write_text(
        f"BABY_SEED{seed}_DIFFUSION_TEST_RUN_COUNT=1\n"
        "NO_POST_TEST_TUNING=true\nTEST_USED_FOR_SELECTION=false\n"
    )
    for x in (bt,bv): x.unlink(missing_ok=True)
    try: tmp.rmdir()
    except OSError: pass
    print(json.dumps({
        "seed":seed,"MSCA":msca_metrics,"CoLiftRec":full_metrics,"Diffusion":diff_metrics,
        "diff_primary":diff_delta["primary_positive_count"],
        "U":diff_delta["mean_relative_primary_gain"],
        "status":out["DIFFUSION"]["TEST_STATUS"],
        "bootstrap_P_U_gt_0":out["DIFFUSION"]["bootstrap_vs_COLIFTREC"]["mean_relative_primary_gain"]["P_gt_0"],
    },sort_keys=True),flush=True)
    return out

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--seed",type=int,required=True)
    ap.add_argument(
        "--freeze",
        default="docs/evidence/paper/BABY_CANONICAL_BACKBONE_FROZEN_BEFORE_MULTISEED_TEST.json",
    )
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    run(a.seed,Path(a.freeze),Path(a.out))
