from __future__ import annotations
import argparse, gc, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from modules.attribute import attribute_z, build_item_matrices, build_profiles
from modules.coliftrec import CoLiftConfig, fit_backgrounds, score_coliftrec, shrink_item_background
from modules.diffusion import NativeTVX0Denoiser, l2_rows_np, purify_indices
from modules.ranking import metrics_at, rank_by_score, row_zscore, semantic_z_for_candidates, topk_from_embeddings
from pipelines.dataset_config import load_dataset_config
from pipelines.msca_assets import build_train_histories_and_validation

PRIMARY=("R10","N10","R20","N20")
ALL=("R10","N10","R20","N20","R50","N50")
DEVICE="cuda"
PURIFY_BATCH=256

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()

def btag(x): return ("%.1f"%float(x)).replace(".","p")

def params(ccfg):
    return CoLiftConfig(
        lambda_text=float(ccfg["text"]["lambda"]),
        lambda_attribute=float(ccfg["attribute"]["lambda"]),
        lambda_visual=float(ccfg["visual"]["lambda"]),
        alpha_text=float(ccfg["text"]["alpha"]),
        alpha_attribute=float(ccfg["attribute"]["alpha"]),
        alpha_visual=float(ccfg["visual"]["alpha"]),
    )

def load_diffusion_model(diff_dir,dataset,beta):
    checkpoint=Path(diff_dir)/"checkpoints"/f"{dataset}_beta_{btag(beta)}.pt"
    state=torch.load(checkpoint,map_location="cpu",weights_only=False)
    model=NativeTVX0Denoiser(int(state["D"]),cond_dim=64,hidden=int(state["hidden"]),time_dim=64).to(DEVICE)
    model.load_state_dict(state["state_dict"],strict=True)
    model.eval()
    return model,checkpoint

def generate_fixed_purified(dataset,diff_dir,out_dir):
    cfg=load_dataset_config(dataset); dataset=cfg["dataset"]; paths=cfg["resolved_paths"]; dcfg=cfg["diffusion"]
    beta=float(dcfg["beta"]); t_edit=int(dcfg["t_edit"]); guidance=float(dcfg["guidance"])
    seeds=tuple(int(x) for x in dcfg["purification_seeds"])
    model,checkpoint=load_diffusion_model(Path(diff_dir),dataset,beta)
    condition_path=Path(diff_dir)/"assets"/f"condition_beta_{btag(beta)}.npy"
    condition=np.load(condition_path,mmap_mode="r",allow_pickle=False)
    raw_text=np.load(paths["text_feature"],mmap_mode="r",allow_pickle=False)
    raw_visual=np.load(paths["visual_feature"],mmap_mode="r",allow_pickle=False)
    ids=np.arange(len(raw_text),dtype=np.int64)
    out_dir=Path(out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    text,visual=purify_indices(
        model,raw_text,raw_visual,condition,ids,
        t_edit=t_edit,guidance=guidance,seeds=seeds,batch=PURIFY_BATCH,device=DEVICE)
    text_path=out_dir/"fixed_text.npy"; visual_path=out_dir/"fixed_visual.npy"
    np.save(text_path,text.astype(np.float32)); np.save(visual_path,visual.astype(np.float32))
    manifest={
        "phase":"PUBLICATION_FIXED_PURIFICATION","dataset":dataset,
        "beta":beta,"t_edit":t_edit,"guidance":guidance,
        "rho_text":float(dcfg["rho_text"]),"rho_visual":float(dcfg["rho_visual"]),
        "purification_seeds":list(seeds),
        "checkpoint_sha256":sha256(checkpoint),"condition_sha256":sha256(condition_path),
        "text_path":str(text_path),"text_sha256":sha256(text_path),
        "visual_path":str(visual_path),"visual_sha256":sha256(visual_path),
        "TEST_ACCESSED":False,"NO_PARAMETER_SELECTION":True}
    (out_dir/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    del model,text,visual; torch.cuda.empty_cache(); gc.collect()
    return manifest

def make_blend(raw_path,purified_path,rho,out_path,batch=1024):
    raw=np.load(raw_path,mmap_mode="r",allow_pickle=False)
    purified=np.load(purified_path,mmap_mode="r",allow_pickle=False)
    if raw.shape!=purified.shape: raise RuntimeError(("blend shape mismatch",raw.shape,purified.shape))
    mm=np.lib.format.open_memmap(out_path,mode="w+",dtype=np.float32,shape=raw.shape)
    for start in range(0,len(raw),batch):
        a=l2_rows_np(raw[start:start+batch]); b=l2_rows_np(purified[start:start+batch])
        mm[start:start+batch]=l2_rows_np((1.0-float(rho))*a+float(rho)*b)
    mm.flush(); del mm

def semantic_lift(path,pseudo_hist,pseudo_users,pseudo_items,histories,users,items,n_items,lam,batch=128):
    z_train,_=semantic_z_for_candidates(path,pseudo_hist,pseudo_users,pseudo_items,batch_users=batch)
    z_eval,_=semantic_z_for_candidates(path,histories,users,items,batch_users=batch)
    background=shrink_item_background(pseudo_items,z_train,n_items)
    return row_zscore(z_eval-float(lam)*background["shrunk_mean"][items])

def build_test_eval(inter_path,n_users):
    df=pd.read_csv(inter_path,sep="\t",usecols=["userID","itemID","timestamp","x_label"])
    df["_row"]=np.arange(len(df),dtype=np.int64)
    train=df[df.x_label==0].copy().sort_values(["userID","timestamp","_row"],kind="stable")
    histories=[[] for _ in range(n_users)]
    for user,group in train.groupby("userID",sort=False):
        histories[int(user)]=group.itemID.astype(np.int64).tolist()
    train_users={u for u,h in enumerate(histories) if h}
    test=df[(df.x_label==2)&df.userID.isin(train_users)].copy()
    users=test.userID.drop_duplicates().astype(np.int64).to_numpy()
    sets={int(u):set(g.itemID.astype(int).tolist()) for u,g in test.groupby("userID",sort=False)}
    return histories,users,sets

def apply_fixed_diffusion(cfg,pseudo_hist,pseudo_users,pseudo_items,histories,users,items,n_items,base_score,purified_text,purified_visual,tmp_dir):
    paths=cfg["resolved_paths"]; ccfg=cfg["coliftrec"]; dcfg=cfg["diffusion"]; p=params(ccfg)
    tmp_dir=Path(tmp_dir); tmp_dir.mkdir(parents=True,exist_ok=True)
    bt=tmp_dir/"blend_text.npy"; bv=tmp_dir/"blend_visual.npy"
    make_blend(paths["text_feature"],purified_text,float(dcfg["rho_text"]),bt)
    make_blend(paths["visual_feature"],purified_visual,float(dcfg["rho_visual"]),bv)
    raw_lt=semantic_lift(paths["text_feature"],pseudo_hist,pseudo_users,pseudo_items,histories,users,items,n_items,p.lambda_text)
    raw_lv=semantic_lift(paths["visual_feature"],pseudo_hist,pseudo_users,pseudo_items,histories,users,items,n_items,p.lambda_visual)
    diff_lt=semantic_lift(bt,pseudo_hist,pseudo_users,pseudo_items,histories,users,items,n_items,p.lambda_text)
    diff_lv=semantic_lift(bv,pseudo_hist,pseudo_users,pseudo_items,histories,users,items,n_items,p.lambda_visual)
    score=np.asarray(base_score).copy()
    # Preserve the arithmetic order of each domain's frozen publication evaluator.
    # This does not change the method formula; it prevents float32 reassociation
    # from changing stable ordering for near-tied candidates during refactor parity.
    if cfg["dataset"]=="baby":
        score+=p.alpha_text*(diff_lt-raw_lt)+p.alpha_visual*(diff_lv-raw_lv)
    else:
        score+=p.alpha_text*(diff_lt-raw_lt)
        score+=p.alpha_visual*(diff_lv-raw_lv)
    bt.unlink(missing_ok=True); bv.unlink(missing_ok=True)
    try: tmp_dir.rmdir()
    except OSError: pass
    return score

def evaluate_validation(dataset,assets_dir,coliftrec_dir,purified_text,purified_visual,out_dir):
    cfg=load_dataset_config(dataset); dataset=cfg["dataset"]; paths=cfg["resolved_paths"]
    audit=json.loads((Path(assets_dir)/"audit.json").read_text())
    if audit.get("TEST_ACCESSED") is not False:
        raise RuntimeError("MSCA assets are not Validation-only")
    val=np.load(Path(assets_dir)/"validation_top100.npz")
    pseudo=np.load(Path(assets_dir)/"train_pseudo_top100.npz")
    scores=np.load(Path(coliftrec_dir)/"validation_scores.npz")
    users=scores["users"].astype(np.int64)
    items=scores["items"].astype(np.int32)
    base_score=scores["full_coliftrec"].astype(np.float32)
    if not np.array_equal(users,val["users"]) or not np.array_equal(items,val["items"]):
        raise RuntimeError("validation candidate identity mismatch")
    histories,pseudo_hist,pseudo_users,expected_users,eval_sets=build_train_histories_and_validation(
        paths["interaction"],int(audit["n_users"]))
    if not np.array_equal(users,expected_users) or not np.array_equal(pseudo_users,pseudo["users"]):
        raise RuntimeError("validation user identity mismatch")
    diff_score=apply_fixed_diffusion(
        cfg,pseudo_hist,pseudo_users,pseudo["items"].astype(np.int32),
        histories,users,items,int(audit["n_items"]),base_score,
        Path(purified_text),Path(purified_visual),Path(out_dir)/"tmp")
    msca_rank=items
    full_rank=rank_by_score(items,base_score)
    diff_rank=rank_by_score(items,diff_score)
    metrics={
        "MSCA":metrics_at(msca_rank,users,eval_sets),
        "MSCA_FULL_COLIFTREC_TAV":metrics_at(full_rank,users,eval_sets),
        "MSCA_FULL_COLIFTREC_DIFFUSION":metrics_at(diff_rank,users,eval_sets)}
    out_dir=Path(out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    result={
        "phase":"PUBLICATION_FIXED_VALIDATION","dataset":dataset,**metrics,
        "msca_checkpoint_sha256":audit["checkpoint_sha256"],
        "publication_config":cfg["_method_config_path"],
        "TEST_ACCESSED":False,"NO_PARAMETER_SELECTION":True}
    (out_dir/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    np.savez_compressed(
        out_dir/"ranked_items.npz",users=users,
        msca=msca_rank,full_coliftrec=full_rank,diffusion=diff_rank)
    np.savez_compressed(
        out_dir/"scores.npz",users=users,items=items,msca=val["scores"],
        full_coliftrec=base_score,diffusion=diff_score)
    return result

def evaluate_test(dataset,assets_dir,purified_text,purified_visual,out_dir):
    cfg=load_dataset_config(dataset); dataset=cfg["dataset"]; paths=cfg["resolved_paths"]; ccfg=cfg["coliftrec"]
    audit=json.loads((Path(assets_dir)/"audit.json").read_text())
    if int(audit.get("seed",-1))!=int(cfg["backbone"]["seed"]):
        raise RuntimeError("MSCA publication seed mismatch")
    n_users,n_items=int(audit["n_users"]),int(audit["n_items"])
    histories,pseudo_hist,expected_pseudo_users,_,_=build_train_histories_and_validation(paths["interaction"],n_users)
    histories_test,test_users,test_sets=build_test_eval(paths["interaction"],n_users)
    if histories!=histories_test: raise RuntimeError("TRAIN history mismatch")
    pseudo=np.load(Path(assets_dir)/"train_pseudo_top100.npz")
    pseudo_users=pseudo["users"].astype(np.int64); pseudo_items=pseudo["items"].astype(np.int32)
    if not np.array_equal(pseudo_users,expected_pseudo_users): raise RuntimeError("pseudo user order mismatch")

    emb=np.load(Path(assets_dir)/"embeddings.npz")
    final_user=torch.as_tensor(emb["final_user"],device=DEVICE)
    final_item=torch.as_tensor(emb["final_item"],device=DEVICE)
    test_items,test_scores=topk_from_embeddings(
        final_user,final_item,test_users,histories,
        top_l=int(ccfg["top_l"]),batch_users=1024)
    del final_user,final_item; torch.cuda.empty_cache()

    p=params(ccfg); raw_text=paths["text_feature"]; raw_visual=paths["visual_feature"]
    zt_train,_=semantic_z_for_candidates(raw_text,pseudo_hist,pseudo_users,pseudo_items,batch_users=256)
    zt_test,_=semantic_z_for_candidates(raw_text,histories,test_users,test_items,batch_users=256)
    zv_train,_=semantic_z_for_candidates(raw_visual,pseudo_hist,pseudo_users,pseudo_items,batch_users=128)
    zv_test,_=semantic_z_for_candidates(raw_visual,histories,test_users,test_items,batch_users=128)
    acfg=ccfg["attribute"]
    mats,_=build_item_matrices(
        paths["metadata"],n_items,min_df=int(acfg["tfidf_min_df"]),
        max_df=float(acfg["tfidf_max_df"]),description_len=int(acfg["description_len"]))
    full_profiles=build_profiles(mats,histories,n_items)
    pseudo_profiles=build_profiles(mats,pseudo_hist,n_items)
    za_train,_=attribute_z(mats,pseudo_profiles,pseudo_users,pseudo_items,batch=256)
    za_test,_=attribute_z(mats,full_profiles,test_users,test_items,batch=256)
    backgrounds=fit_backgrounds(pseudo_items,zt_train,za_train,zv_train,n_items)
    full_score,_=score_coliftrec(
        test_scores,test_items,zt_test,za_test,zv_test,backgrounds,p,
        enabled={"text":True,"attribute":True,"visual":True})
    diff_score=apply_fixed_diffusion(
        cfg,pseudo_hist,pseudo_users,pseudo_items,histories,test_users,test_items,n_items,
        full_score,Path(purified_text),Path(purified_visual),Path(out_dir)/"tmp")
    msca_rank=test_items
    full_rank=rank_by_score(test_items,full_score)
    diff_rank=rank_by_score(test_items,diff_score)
    metrics={
        "MSCA":metrics_at(msca_rank,test_users,test_sets),
        "MSCA_FULL_COLIFTREC_TAV":metrics_at(full_rank,test_users,test_sets),
        "MSCA_FULL_COLIFTREC_DIFFUSION":metrics_at(diff_rank,test_users,test_sets)}
    out_dir=Path(out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    result={
        "phase":"PUBLICATION_FIXED_TEST_EVALUATION","dataset":dataset,**metrics,
        "msca_checkpoint_sha256":audit["checkpoint_sha256"],
        "publication_config":cfg["_method_config_path"],
        "REFRACTOR_PARITY_EVALUATION":True,
        "TEST_USED_FOR_SELECTION":False,"NO_PARAMETER_SELECTION":True}
    (out_dir/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    np.savez_compressed(
        out_dir/"ranked_items.npz",users=test_users,
        msca=msca_rank,full_coliftrec=full_rank,diffusion=diff_rank)
    np.savez_compressed(
        out_dir/"scores.npz",users=test_users,items=test_items,msca=test_scores,
        full_coliftrec=full_score,diffusion=diff_score)
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset",required=True)
    ap.add_argument("--mode",choices=["purify","validation","test"],required=True)
    ap.add_argument("--msca-assets")
    ap.add_argument("--coliftrec-dir")
    ap.add_argument("--diffusion-dir")
    ap.add_argument("--purified-dir")
    ap.add_argument("--purified-text")
    ap.add_argument("--purified-visual")
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    if a.mode=="purify":
        result=generate_fixed_purified(a.dataset,Path(a.diffusion_dir),Path(a.out))
    else:
        text=Path(a.purified_text) if a.purified_text else Path(a.purified_dir)/"fixed_text.npy"
        visual=Path(a.purified_visual) if a.purified_visual else Path(a.purified_dir)/"fixed_visual.npy"
        if a.mode=="validation":
            result=evaluate_validation(
                a.dataset,Path(a.msca_assets),Path(a.coliftrec_dir),
                text,visual,Path(a.out))
        else:
            result=evaluate_test(
                a.dataset,Path(a.msca_assets),text,visual,Path(a.out))
    print(json.dumps(result,sort_keys=True))

if __name__=="__main__":
    main()
