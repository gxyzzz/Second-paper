from __future__ import annotations
import argparse, gc, hashlib, json, time
from logging import getLogger
from pathlib import Path
import numpy as np
import torch

from modules.diffusion import NativeTVX0Denoiser, cosine_alpha_bar, l2_rows_np, paired_errors, training_loss
from modules.semantic_purifier import condition_beta
from pipelines.dataset_config import load_dataset_config
from pipelines.diffusion_assets import build_current_run_assets

DEVICE="cuda"

def _hp(dcfg):
    arch=dcfg["architecture"]; sched=dcfg["schedule"]; loss=dcfg["loss"]
    clf=dcfg["classifier_free"]; opt=dcfg["optimizer"]
    if str(opt.get("name","AdamW")).lower()!="adamw":
        raise ValueError(f"unsupported diffusion optimizer: {opt.get('name')}")
    wt=float(loss["recon_text_weight"]); wv=float(loss["recon_visual_weight"])
    if wt<0 or wv<0 or abs(wt+wv-1.0)>=1e-8:
        raise ValueError(f"invalid reconstruction weights: {wt}, {wv}")
    return {
        "hidden":int(arch["hidden"]), "bottleneck":int(arch["bottleneck"]),
        "time_dim":int(arch["time_dim"]), "diffusion_steps":int(sched["diffusion_steps"]),
        "cosine_s":float(sched["cosine_s"]), "lambda_ctr":float(loss["lambda_ctr"]),
        "recon_text_weight":wt, "recon_visual_weight":wv,
        "p_uncond":float(clf["p_uncond"]), "lr":float(opt["lr"]),
        "weight_decay":float(opt["weight_decay"]),
        "train_batch":int(dcfg.get("train_batch",512)),
        "monitor_batch":int(dcfg.get("monitor_batch",512)),
    }

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for x in iter(lambda:f.read(1<<20),b""): h.update(x)
    return h.hexdigest()

def beta_tag(beta):
    return ("%.1f"%float(beta)).replace(".","p")

def make_x(raw_t,raw_v,ids):
    return np.concatenate([l2_rows_np(raw_t[ids]),l2_rows_np(raw_v[ids])],axis=1).astype(np.float32,copy=False)

def split_train_monitor(ids,seed):
    ids=np.asarray(ids,dtype=np.int64).copy()
    rng=np.random.RandomState(int(seed)); rng.shuffle(ids)
    n_monitor=max(1,int(round(len(ids)*0.05)))
    monitor=np.sort(ids[:n_monitor]); train=np.sort(ids[n_monitor:])
    assert len(np.intersect1d(train,monitor))==0
    assert len(train)+len(monitor)==len(ids)
    return train,monitor

def monitor_objective(model,raw_t,raw_v,cond,ids,ab,noise_seed,hp):
    model.eval(); total=rec=ctr=0.; n=0
    with torch.no_grad():
        for bi,st in enumerate(range(0,len(ids),hp["monitor_batch"])):
            ii=ids[st:st+hp["monitor_batch"]]
            x0=torch.from_numpy(make_x(raw_t,raw_v,ii)).to(DEVICE)
            cc=torch.from_numpy(np.asarray(cond[ii],np.float32)).to(DEVICE)
            gen=torch.Generator(device=DEVICE); gen.manual_seed(int(noise_seed)+bi)
            tt=torch.randint(1,hp["diffusion_steps"]+1,(len(ii),),generator=gen,device=DEVICE)
            noise=torch.randn(x0.shape,generator=gen,device=DEVICE,dtype=x0.dtype)
            loss,info=training_loss(
                model,x0,cc,tt,noise,ab,lambda_ctr=hp["lambda_ctr"],p_uncond=0.,
                recon_text_weight=hp["recon_text_weight"],
                recon_visual_weight=hp["recon_visual_weight"])
            bs=len(ii); n+=bs
            total+=float(loss)*bs; rec+=float(info["loss_rec"])*bs; ctr+=float(info["loss_ctr"])*bs
    return {"monitor_objective":total/n,"monitor_rec":rec/n,"monitor_ctr":ctr/n}

def train_epoch(model,opt,raw_t,raw_v,cond,ids,ab,shuffle_seed,epoch,hp):
    model.train()
    order=np.asarray(ids,dtype=np.int64).copy()
    rng=np.random.RandomState(int(shuffle_seed)+int(epoch)); rng.shuffle(order)
    vals=[]; t0=time.time()
    for st in range(0,len(order),hp["train_batch"]):
        ii=order[st:st+hp["train_batch"]]
        x0=torch.from_numpy(make_x(raw_t,raw_v,ii)).to(DEVICE)
        cc=torch.from_numpy(np.asarray(cond[ii],np.float32)).to(DEVICE)
        tt=torch.randint(1,hp["diffusion_steps"]+1,(len(ii),),device=DEVICE)
        noise=torch.randn_like(x0)
        loss,info=training_loss(
            model,x0,cc,tt,noise,ab,lambda_ctr=hp["lambda_ctr"],p_uncond=hp["p_uncond"],
            recon_text_weight=hp["recon_text_weight"],
            recon_visual_weight=hp["recon_visual_weight"])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        vals.append((len(ii),float(loss),info))
    denom=sum(x[0] for x in vals)
    def wav(key):
        if key=="loss": return sum(bs*v for bs,v,_ in vals)/denom
        return sum(bs*float(info[key]) for bs,_,info in vals)/denom
    return {"epoch":epoch,"train_loss":wav("loss"),"train_rec":wav("loss_rec"),"train_ctr":wav("loss_ctr"),
            "train_e_text":wav("e_text"),"train_e_visual":wav("e_visual"),
            "unconditional_ratio":wav("unconditional_ratio"),"epoch_runtime_sec":time.time()-t0}

def make_model(seed,dim,cond_dim,hp):
    torch.manual_seed(int(seed)); torch.cuda.manual_seed_all(int(seed))
    model=NativeTVX0Denoiser(
        dim,cond_dim=cond_dim,hidden=hp["hidden"],bottleneck=hp["bottleneck"],
        time_dim=hp["time_dim"]).to(DEVICE)
    return model,torch.optim.AdamW(
        model.parameters(),lr=hp["lr"],weight_decay=hp["weight_decay"])
def mechanism_audit(model,raw_t,raw_v,cond,ids,ab,hp):
    ids=np.asarray(ids,dtype=np.int64)[:min(len(ids),1024)]
    keys=["true_text","true_visual","true_balanced","shuf_text","shuf_visual","shuf_balanced","null_text","null_visual","null_balanced"]
    accum={k:[] for k in keys}; model.eval()
    with torch.no_grad():
        for ti in (5,10,15,20,30,40):
            for bi,st in enumerate(range(0,len(ids),256)):
                ii=ids[st:st+256]
                x0=torch.from_numpy(make_x(raw_t,raw_v,ii)).to(DEVICE)
                cc=torch.from_numpy(np.asarray(cond[ii],np.float32)).to(DEVICE)
                gen=torch.Generator(device=DEVICE); gen.manual_seed(20261200+ti+bi)
                tt=torch.full((len(ii),),ti,device=DEVICE,dtype=torch.long)
                noise=torch.randn(x0.shape,generator=gen,device=DEVICE,dtype=x0.dtype)
                e=paired_errors(
                    model,x0,cc,tt,noise,ab,
                    recon_text_weight=hp["recon_text_weight"],
                    recon_visual_weight=hp["recon_visual_weight"])
                for k,v in e.items(): accum[k].append(v.detach().cpu().numpy())
    m={k:float(np.concatenate(v).mean()) for k,v in accum.items()}
    m["true_better_shuffled_balanced"]=bool(m["true_balanced"]<m["shuf_balanced"])
    m["true_better_null_balanced"]=bool(m["true_balanced"]<m["null_balanced"])
    return m

def run_m31(dataset,cfg,asset_audit,raw_t,raw_v,collab,final,out_dir,mode,betas):
    dcfg=cfg["diffusion"]; hp=_hp(dcfg); seed=int(dcfg["training_seed"]); epochs=int(dcfg["epochs"])
    if dcfg["train_scope"]!="all_catalog_items" or dcfg["checkpoint_selection"]!="final_epoch":
        raise RuntimeError("invalid m31 config")
    ids=np.arange(len(raw_t),dtype=np.int64)
    if len(ids)!=len(collab): raise RuntimeError("catalog item count mismatch")
    train_ids=ids[:min(1024,len(ids))] if mode=="smoke" else ids
    checkpoint_dir=out_dir/"checkpoints"; evidence_dir=out_dir/"evidence"; asset_dir=out_dir/"assets"
    ab=cosine_alpha_bar(hp["diffusion_steps"],hp["cosine_s"]).to(DEVICE); dim=raw_t.shape[1]+raw_v.shape[1]
    for beta in betas:
        cond=condition_beta(collab,final,beta)
        cp=asset_dir/f"condition_beta_{beta_tag(beta)}.npy"; np.save(cp,cond,allow_pickle=False)
        model,opt=make_model(seed,dim,collab.shape[1],hp); logs=[]
        end=2 if mode=="smoke" else epochs
        for ep in range(1,end+1):
            rec=train_epoch(model,opt,raw_t,raw_v,cond,train_ids,ab,seed,ep,hp); logs.append(rec)
            if ep==1 or ep%10==0 or ep==end:
                getLogger().info("Diffusion epoch %s/%s dataset=%s beta=%s %s", ep, end, dataset, beta, json.dumps(rec,sort_keys=True))
        if mode=="smoke":
            ev={"phase":"M31_FIXED_ALL_ITEMS_SMOKE","dataset":dataset,"beta":beta,
                "training_protocol":"m31_fixed_all_items","train_scope":"all_catalog_items",
                "training_seed":seed,"epochs_smoke":end,"finite":bool(np.isfinite(logs[-1]["train_loss"])),
                "VALIDATION_TARGET_USED":False,"TEST_TARGET_USED":False,"ALL_CATALOG_SIDE_INFORMATION":True}
            (evidence_dir/f"m31_beta_{beta_tag(beta)}_smoke.json").write_text(json.dumps(ev,indent=2)+"\n")
        else:
            ck=checkpoint_dir/f"{dataset}_beta_{beta_tag(beta)}.pt"
            torch.save({"state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},
                        "D":dim,"hidden":hp["hidden"],"bottleneck":hp["bottleneck"],
                        "time_dim":hp["time_dim"],"beta":beta,
                        "source_msca_checkpoint_sha256":asset_audit["source_checkpoint_sha256"]},ck)
            mech=mechanism_audit(model,raw_t,raw_v,cond,ids,ab,hp)
            meta={"phase":"DIFFUSION_FORMAL_TRAINING","dataset":dataset,"beta":beta,
                  "training_protocol":"m31_fixed_all_items","train_scope":"all_catalog_items",
                  "checkpoint_selection":"final_epoch",
                  "architecture":{"hidden":hp["hidden"],"bottleneck":hp["bottleneck"],"time_dim":hp["time_dim"]},
                  "prediction":"x0","schedule":"cosine","diffusion_steps":hp["diffusion_steps"],"cosine_s":hp["cosine_s"],
                  "lambda_ctr":hp["lambda_ctr"],"p_uncond":hp["p_uncond"],"optimizer":"AdamW",
                  "lr":hp["lr"],"weight_decay":hp["weight_decay"],
                  "recon_text_weight":hp["recon_text_weight"],"recon_visual_weight":hp["recon_visual_weight"],
                  "training_seed":seed,"epochs":epochs,"final_epoch":epochs,"train_item_count":int(len(ids)),
                  "checkpoint":str(ck),"checkpoint_sha256":sha256(ck),
                  "condition_path":str(cp),"condition_sha256":sha256(cp),
                  "source_msca_checkpoint_sha256":asset_audit["source_checkpoint_sha256"],
                  "mechanism":mech,"epoch_log":logs,
                  "VALIDATION_TARGET_USED":False,"TEST_TARGET_USED":False,
                  "ALL_CATALOG_SIDE_INFORMATION":True,"VALIDATION_RANKING_USED_FOR_TRAINING":False,
                  "TEST_ACCESSED":False}
            (evidence_dir/f"beta_{beta_tag(beta)}_training.json").write_text(json.dumps(meta,indent=2)+"\n")
            getLogger().info("Diffusion training complete dataset=%s beta=%s final_epoch=%s checkpoint_sha256=%s", dataset, beta, epochs, meta["checkpoint_sha256"])
        del model,opt; torch.cuda.empty_cache(); gc.collect()
def run_m32(dataset,cfg,asset_audit,raw_t,raw_v,collab,final,out_dir,mode,betas,eligible_ids):
    dcfg=cfg["diffusion"]; hp=_hp(dcfg)
    split_seed=int(dcfg["split_seed"]); seed_base=int(dcfg["train_seed_base"])
    monitor_seed=int(dcfg["monitor_noise_seed"]); max_epochs=int(dcfg["max_epochs"])
    min_epochs=int(dcfg["min_epochs"]); patience=int(dcfg["patience"])
    if dcfg["train_scope"]!="train_interaction_items" or dcfg["checkpoint_selection"]!="best_monitor":
        raise RuntimeError("invalid m32 config")
    train_ids,monitor_ids=split_train_monitor(eligible_ids,split_seed)
    if mode=="smoke":
        train_ids=train_ids[:min(1024,len(train_ids))]
        monitor_ids=monitor_ids[:min(256,len(monitor_ids))]
    asset_dir=out_dir/"assets"; checkpoint_dir=out_dir/"checkpoints"; evidence_dir=out_dir/"evidence"
    np.save(asset_dir/"diffusion_train_item_ids.npy",train_ids,allow_pickle=False)
    np.save(asset_dir/"diffusion_monitor_item_ids.npy",monitor_ids,allow_pickle=False)
    ab=cosine_alpha_bar(hp["diffusion_steps"],hp["cosine_s"]).to(DEVICE); dim=raw_t.shape[1]+raw_v.shape[1]
    for beta in betas:
        seed=seed_base+int(beta*1000)
        cond=condition_beta(collab,final,beta)
        cp=asset_dir/f"condition_beta_{beta_tag(beta)}.npy"; np.save(cp,cond,allow_pickle=False)
        model,opt=make_model(seed,dim,collab.shape[1],hp)
        best={"value":float("inf"),"epoch":0,"state_dict":None,"since":0}; logs=[]
        end=2 if mode=="smoke" else max_epochs; stop_reason="MAX_EPOCHS_REACHED"; stop_epoch=end
        for ep in range(1,end+1):
            rec=train_epoch(model,opt,raw_t,raw_v,cond,train_ids,ab,seed,ep,hp)
            mon=monitor_objective(model,raw_t,raw_v,cond,monitor_ids,ab,monitor_seed,hp)
            rec.update(mon); logs.append(rec)
            if ep==1 or ep%10==0 or ep==end:
                getLogger().info("Diffusion epoch %s/%s dataset=%s beta=%s %s", ep, end, dataset, beta, json.dumps(rec,sort_keys=True))
            if mon["monitor_objective"]<best["value"]-1e-12:
                best={"value":float(mon["monitor_objective"]),"epoch":ep,
                      "state_dict":{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},"since":0}
            else:
                best["since"]+=1
            if ep==1 or ep%10==0 or ep==end:
                getLogger().info(
                    "Diffusion monitor epoch=%s objective=%.12f best_monitor=%.12f best_epoch=%s patience=%s/%s",
                    ep, float(mon["monitor_objective"]), float(best["value"]),
                    int(best["epoch"]), int(best["since"]), patience,
                )
            if mode=="formal" and ep>=min_epochs and best["since"]>=patience:
                stop_reason="EARLY_STOP_PATIENCE"; stop_epoch=ep; break
        if mode=="smoke":
            ev={"phase":"DIFFUSION_TRAIN_MONITOR_SMOKE","dataset":dataset,"beta":beta,
                "finite":bool(np.isfinite(best["value"])),"training_protocol":"m32_train_monitor",
                "TEST_ACCESSED":False}
            (evidence_dir/f"m32_beta_{beta_tag(beta)}_smoke.json").write_text(json.dumps(ev,indent=2)+"\n")
        else:
            ck=checkpoint_dir/f"{dataset}_beta_{beta_tag(beta)}.pt"
            torch.save({"state_dict":best["state_dict"],"D":dim,"hidden":hp["hidden"],
                        "bottleneck":hp["bottleneck"],"time_dim":hp["time_dim"],"beta":beta,
                        "source_msca_checkpoint_sha256":asset_audit["source_checkpoint_sha256"]},ck)
            model.load_state_dict(best["state_dict"],strict=True)
            mech=mechanism_audit(model,raw_t,raw_v,cond,monitor_ids,ab,hp)
            meta={"phase":"DIFFUSION_FORMAL_TRAINING","dataset":dataset,"beta":beta,
                  "training_protocol":"m32_train_monitor","train_scope":"train_interaction_items",
                  "checkpoint_selection":"best_monitor","split_seed":split_seed,
                  "architecture":{"hidden":hp["hidden"],"bottleneck":hp["bottleneck"],"time_dim":hp["time_dim"]},
                  "diffusion_steps":hp["diffusion_steps"],"cosine_s":hp["cosine_s"],
                  "lambda_ctr":hp["lambda_ctr"],"p_uncond":hp["p_uncond"],"optimizer":"AdamW",
                  "lr":hp["lr"],"weight_decay":hp["weight_decay"],
                  "recon_text_weight":hp["recon_text_weight"],"recon_visual_weight":hp["recon_visual_weight"],
                  "training_seed":seed,"monitor_noise_seed":monitor_seed,
                  "MAX_EPOCHS":max_epochs,"MIN_EPOCHS":min_epochs,"PATIENCE":patience,
                  "train_item_count":int(len(train_ids)),"monitor_item_count":int(len(monitor_ids)),
                  "best_epoch":int(best["epoch"]),"best_monitor_objective":float(best["value"]),
                  "stop_epoch":int(stop_epoch),"stop_reason":stop_reason,
                  "checkpoint":str(ck),"checkpoint_sha256":sha256(ck),
                  "condition_path":str(cp),"condition_sha256":sha256(cp),
                  "source_msca_checkpoint_sha256":asset_audit["source_checkpoint_sha256"],
                  "mechanism":mech,"epoch_log":logs,
                  "VALIDATION_RANKING_USED_FOR_TRAINING":False,"TEST_ACCESSED":False}
            (evidence_dir/f"beta_{beta_tag(beta)}_training.json").write_text(json.dumps(meta,indent=2)+"\n")
        del model,opt,best; torch.cuda.empty_cache(); gc.collect()
def run(dataset,msca_assets,out_dir,mode,betas_override=None,training_seed_override=None):
    cfg=load_dataset_config(dataset); dataset=cfg["dataset"]; paths=cfg["resolved_paths"]; dcfg=cfg["diffusion"]
    if training_seed_override is not None:
        dcfg["training_seed"]=int(training_seed_override)
    out_dir=Path(out_dir); asset_dir=out_dir/"assets"; checkpoint_dir=out_dir/"checkpoints"; evidence_dir=out_dir/"evidence"
    for p in (asset_dir,checkpoint_dir,evidence_dir): p.mkdir(parents=True,exist_ok=True)
    asset_audit=build_current_run_assets(dataset,Path(msca_assets),asset_dir)
    ep=np.load(asset_dir/"condition_endpoints.npz")
    collab=ep["collab_item"].astype(np.float32); final=ep["final_item"].astype(np.float32)
    raw_t=np.load(paths["text_feature"],mmap_mode="r",allow_pickle=False)
    raw_v=np.load(paths["visual_feature"],mmap_mode="r",allow_pickle=False)
    if raw_t.shape[1]!=384 or raw_v.shape[1]!=4096 or raw_t.shape[0]!=collab.shape[0]:
        raise RuntimeError("native asset mismatch")
    betas=[float(x) for x in (betas_override if betas_override is not None else dcfg["beta_candidates"])]
    protocol=dcfg["training_protocol"]
    if protocol=="m31_fixed_all_items":
        run_m31(dataset,cfg,asset_audit,raw_t,raw_v,collab,final,out_dir,mode,betas)
    elif protocol=="m32_train_monitor":
        eligible=np.load(asset_dir/"train_item_ids.npy",allow_pickle=False)
        run_m32(dataset,cfg,asset_audit,raw_t,raw_v,collab,final,out_dir,mode,betas,eligible)
    else:
        raise RuntimeError(f"unknown diffusion training protocol: {protocol}")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset",required=True)
    ap.add_argument("--msca-assets",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--mode",choices=["smoke","formal"],required=True)
    ap.add_argument("--betas",nargs="*",type=float)
    ap.add_argument("--training-seed",type=int)
    a=ap.parse_args()
    run(a.dataset,a.msca_assets,a.out,a.mode,a.betas,a.training_seed)
