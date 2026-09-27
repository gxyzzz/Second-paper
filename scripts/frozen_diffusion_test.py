from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from modules.attribute import attribute_z, build_item_matrices, build_profiles
from modules.coliftrec import CoLiftConfig, fit_backgrounds, score_coliftrec, shrink_item_background
from modules.ranking import metrics_at, rank_by_score, row_zscore, semantic_z_for_candidates, topk_from_embeddings
from modules.diffusion import l2_rows_np
from pipelines.dataset_config import load_dataset_config
from pipelines.msca_assets import build_train_histories_and_validation

PRIMARY=("R10","N10","R20","N20")
ALL=("R10","N10","R20","N20","R50","N50")
DEVICE="cuda"

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for x in iter(lambda:f.read(1<<20),b""): h.update(x)
    return h.hexdigest()

def btag(x): return ("%.1f"%float(x)).replace(".","p")
def gtag(x): return ("%.1f"%float(x)).replace(".","p")

def build_test_eval(inter_path,n_users):
    df=pd.read_csv(inter_path,sep="\t",usecols=["userID","itemID","timestamp","x_label"])
    df["_row"]=np.arange(len(df),dtype=np.int64)
    train=df[df.x_label==0].copy().sort_values(["userID","timestamp","_row"],kind="stable")
    histories=[[] for _ in range(n_users)]
    for u,g in train.groupby("userID",sort=False):
        histories[int(u)]=g.itemID.astype(np.int64).tolist()
    train_users={u for u,h in enumerate(histories) if h}
    test=df[(df.x_label==2)&df.userID.isin(train_users)].copy()
    users=test.userID.drop_duplicates().astype(np.int64).to_numpy()
    sets={int(u):set(g.itemID.astype(int).tolist()) for u,g in test.groupby("userID",sort=False)}
    return histories,users,sets

def params(ccfg):
    return CoLiftConfig(
        lambda_text=float(ccfg["text"]["lambda"]),
        lambda_attribute=float(ccfg["attribute"]["lambda"]),
        lambda_visual=float(ccfg["visual"]["lambda"]),
        alpha_text=float(ccfg["text"]["alpha"]),
        alpha_attribute=float(ccfg["attribute"]["alpha"]),
        alpha_visual=float(ccfg["visual"]["alpha"]),
    )

def make_blend(raw_path,pur_path,rho,out_path,batch=1024):
    raw=np.load(raw_path,mmap_mode="r",allow_pickle=False)
    pur=np.load(pur_path,mmap_mode="r",allow_pickle=False)
    if raw.shape!=pur.shape: raise RuntimeError(("blend shape mismatch",raw.shape,pur.shape))
    mm=np.lib.format.open_memmap(out_path,mode="w+",dtype=np.float32,shape=raw.shape)
    for st in range(0,len(raw),batch):
        a=l2_rows_np(raw[st:st+batch]); b=l2_rows_np(pur[st:st+batch])
        mm[st:st+batch]=l2_rows_np((1-float(rho))*a+float(rho)*b)
    mm.flush(); del mm

def semantic_lift(path,pseudo_hist,pseudo_users,pseudo_items,histories,users,items,n_items,lam,batch):
    ztr,_=semantic_z_for_candidates(path,pseudo_hist,pseudo_users,pseudo_items,batch_users=batch)
    zev,_=semantic_z_for_candidates(path,histories,users,items,batch_users=batch)
    bg=shrink_item_background(pseudo_items,ztr,n_items)
    return row_zscore(zev-float(lam)*bg["shrunk_mean"][items])

def freeze_gate(dataset,freeze,asset_audit,cfg):
    up=dataset.upper()
    if freeze.get("TEST_USED_FOR_SELECTION") is not False:
        raise RuntimeError("freeze was selected using Test")
    if freeze.get(f"{up}_DIFFUSION_TEST")!="CLOSED":
        raise RuntimeError("Test was already opened or freeze state is invalid")
    if freeze.get("msca_checkpoint_sha256")!=asset_audit["checkpoint_sha256"]:
        raise RuntimeError("MSCA checkpoint SHA mismatch")
    if freeze.get("coliftrec_config")!=cfg["coliftrec"]:
        raise RuntimeError("CoLiftRec config differs from frozen Validation config")
    protocol=freeze.get("training_protocol")
    if protocol=="m31_fixed_all_items":
        ok=(freeze.get("status")=="FROZEN_STABLE_M31_ANCHOR"
            and float(freeze.get("U",0))>0
            and int(freeze.get("primary_positive_count",0))>=3
            and float(freeze.get("sum_primary_delta",0))>0)
    elif protocol=="m32_train_monitor":
        cross=freeze.get("crossfit_summary",{})
        ok=(freeze.get("final_reason")=="CROSSFIT_PASS_FULL_VALIDATION_MAX_U"
            and freeze.get("final_config",{}).get("id")!="NO_DIFFUSION"
            and bool(cross.get(f"{up}_DIFFUSION_UPGRADE_PASS")))
    else:
        raise RuntimeError("unknown frozen training protocol")
    if not ok: raise RuntimeError("Validation Gate did not PASS; Test remains closed")
    return protocol
def resolve_purified(dataset,diff_dir,freeze):
    protocol=freeze["training_protocol"]
    beta=float(freeze["beta"] if protocol=="m31_fixed_all_items" else freeze["selected_beta"])
    t=int(freeze["t_edit"]); g=float(freeze["guidance"])
    if protocol=="m31_fixed_all_items":
        stem=Path(diff_dir)/"purified_anchor"/f"beta_{btag(beta)}_t{t}_g{gtag(g)}"
        ck=Path(diff_dir)/"checkpoints"/f"{dataset}_beta_{btag(beta)}.pt"
        expected_ck=freeze["diffusion_checkpoint_sha256"]
    else:
        stem=Path(diff_dir)/"purified"/f"beta_{btag(beta)}"/f"t{t}_g{gtag(g)}"
        ck=Path(diff_dir)/"checkpoints"/f"{dataset}_beta_{btag(beta)}.pt"
        expected_ck=freeze["beta_checkpoint_sha256"]
    tp=Path(str(stem)+"_text.npy"); vp=Path(str(stem)+"_visual.npy")
    for p in (tp,vp,ck):
        if not p.exists(): raise FileNotFoundError(p)
    if sha256(ck)!=expected_ck: raise RuntimeError("Diffusion checkpoint SHA mismatch")
    fc=freeze.get("final_config",{})
    if fc.get("text_sha256") and sha256(tp)!=fc["text_sha256"]: raise RuntimeError("purified text SHA mismatch")
    if fc.get("visual_sha256") and sha256(vp)!=fc["visual_sha256"]: raise RuntimeError("purified visual SHA mismatch")
    return tp,vp,ck

def run(dataset,assets_dir,diff_dir,freeze_path,out_dir):
    cfg=load_dataset_config(dataset); dataset=cfg["dataset"]; paths=cfg["resolved_paths"]; ccfg=cfg["coliftrec"]
    out_dir=Path(out_dir); marker=out_dir/"TEST_RUN_COMPLETED"
    if marker.exists(): raise RuntimeError("current-run frozen Test already executed; refusing second run")
    freeze=json.loads(Path(freeze_path).read_text())
    asset_audit=json.loads((Path(assets_dir)/"audit.json").read_text())
    protocol=freeze_gate(dataset,freeze,asset_audit,cfg)
    tp,vp,diff_ck=resolve_purified(dataset,diff_dir,freeze)

    # Test target is first read only after all frozen-config guards pass.
    n_users=int(asset_audit["n_users"]); n_items=int(asset_audit["n_items"])
    histories,pseudo_hist,expected_pseudo_users,_,_=build_train_histories_and_validation(paths["interaction"],n_users)
    histories2,test_users,test_sets=build_test_eval(paths["interaction"],n_users)
    if histories!=histories2: raise RuntimeError("TRAIN history mismatch")

    pseudo_asset=np.load(Path(assets_dir)/"train_pseudo_top100.npz")
    pseudo_users=pseudo_asset["users"].astype(np.int64); pseudo_items=pseudo_asset["items"].astype(np.int32)
    if not np.array_equal(pseudo_users,expected_pseudo_users): raise RuntimeError("pseudo user order mismatch")

    emb=np.load(Path(assets_dir)/"embeddings.npz")
    final_user=torch.as_tensor(emb["final_user"],device=DEVICE)
    final_item=torch.as_tensor(emb["final_item"],device=DEVICE)
    test_items,test_scores=topk_from_embeddings(final_user,final_item,test_users,histories,
        top_l=int(ccfg.get("top_l",100)),batch_users=1024)
    del final_user,final_item; torch.cuda.empty_cache()

    p=params(ccfg); enabled={m:bool(ccfg[m]["enabled"]) for m in ("text","attribute","visual")}
    raw_t=paths["text_feature"]; raw_v=paths["visual_feature"]

    zt_tr,_=semantic_z_for_candidates(raw_t,pseudo_hist,pseudo_users,pseudo_items,batch_users=256)
    zt_te,_=semantic_z_for_candidates(raw_t,histories,test_users,test_items,batch_users=256)
    zv_tr,_=semantic_z_for_candidates(raw_v,pseudo_hist,pseudo_users,pseudo_items,batch_users=128)
    zv_te,_=semantic_z_for_candidates(raw_v,histories,test_users,test_items,batch_users=128)

    acfg=ccfg["attribute"]
    mats,_=build_item_matrices(paths["metadata"],n_items,
        min_df=int(acfg.get("tfidf_min_df",2)),max_df=float(acfg.get("tfidf_max_df",.8)),
        description_len=int(acfg.get("description_len",128)))
    prof=build_profiles(mats,histories,n_items); pprof=build_profiles(mats,pseudo_hist,n_items)
    za_tr,_=attribute_z(mats,pprof,pseudo_users,pseudo_items,batch=256)
    za_te,_=attribute_z(mats,prof,test_users,test_items,batch=256)

    bgs=fit_backgrounds(pseudo_items,zt_tr,za_tr,zv_tr,n_items)
    full_score,_=score_coliftrec(test_scores,test_items,zt_te,za_te,zv_te,bgs,p,enabled=enabled)
    msca_metrics=metrics_at(test_items,test_users,test_sets)
    full_rank=rank_by_score(test_items,full_score)
    full_metrics=metrics_at(full_rank,test_users,test_sets)

    tmp=out_dir/"tmp"; tmp.mkdir(parents=True,exist_ok=True)
    bt=tmp/"blend_text.npy"; bv=tmp/"blend_visual.npy"
    make_blend(raw_t,tp,float(freeze["rho_T"]),bt)
    make_blend(raw_v,vp,float(freeze["rho_V"]),bv)
    raw_lt=semantic_lift(raw_t,pseudo_hist,pseudo_users,pseudo_items,histories,test_users,test_items,n_items,p.lambda_text,128)
    raw_lv=semantic_lift(raw_v,pseudo_hist,pseudo_users,pseudo_items,histories,test_users,test_items,n_items,p.lambda_visual,128)
    dif_lt=semantic_lift(bt,pseudo_hist,pseudo_users,pseudo_items,histories,test_users,test_items,n_items,p.lambda_text,128)
    dif_lv=semantic_lift(bv,pseudo_hist,pseudo_users,pseudo_items,histories,test_users,test_items,n_items,p.lambda_visual,128)
    diff_score=full_score.copy()
    diff_score+=p.alpha_text*(dif_lt-raw_lt)
    diff_score+=p.alpha_visual*(dif_lv-raw_lv)
    diff_rank=rank_by_score(test_items,diff_score)
    diff_metrics=metrics_at(diff_rank,test_users,test_sets)
    d_full={k:float(full_metrics[k]-msca_metrics[k]) for k in ALL}
    d_diff={k:float(diff_metrics[k]-full_metrics[k]) for k in ALL}
    result={
        "phase":f"{dataset.upper()}_CURRENT_RUN_FROZEN_TEST",
        "dataset":dataset,
        "training_protocol":protocol,
        "MSCA":msca_metrics,
        "MSCA_FULL_COLIFTREC_TAV":full_metrics,
        "MSCA_FULL_COLIFTREC_DIFFUSION":diff_metrics,
        "delta_coliftrec_vs_msca":d_full,
        "delta_diffusion_vs_full_coliftrec":d_diff,
        "diffusion_primary_positive_count":int(sum(d_diff[k]>0 for k in PRIMARY)),
        "msca_checkpoint_sha256":asset_audit["checkpoint_sha256"],
        "diffusion_checkpoint_sha256":sha256(diff_ck),
        "freeze_file":str(freeze_path),
        "freeze_sha256":sha256(freeze_path),
        "CURRENT_RUN_TEST_RUN_COUNT":1,
        "TEST_USED_FOR_SELECTION":False,
        "NO_POST_TEST_TUNING":True,
    }
    out_dir.mkdir(parents=True,exist_ok=True)
    (out_dir/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    np.savez_compressed(out_dir/"test_scores.npz",users=test_users,items=test_items,
        msca=test_scores,full_coliftrec=full_score,diffusion=diff_score)
    marker.write_text("CURRENT_RUN_TEST_RUN_COUNT=1\nNO_POST_TEST_TUNING=true\n")
    for x in (bt,bv):
        x.unlink(missing_ok=True)
    try: tmp.rmdir()
    except OSError: pass
    print(json.dumps(result,sort_keys=True),flush=True)
    return result

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset",required=True)
    ap.add_argument("--msca-assets",required=True)
    ap.add_argument("--diffusion-dir",required=True)
    ap.add_argument("--freeze",required=True)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    run(a.dataset,Path(a.msca_assets),Path(a.diffusion_dir),Path(a.freeze),Path(a.out))
