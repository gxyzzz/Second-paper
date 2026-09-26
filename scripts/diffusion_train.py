from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch

from modules.diffusion import NativeTVX0Denoiser, cosine_alpha_bar, l2_rows_np, paired_errors, training_loss
from modules.semantic_purifier import condition_beta
from pipelines.dataset_config import load_dataset_config
from pipelines.diffusion_assets import build_current_run_assets

MAX_EPOCHS=80
MIN_EPOCHS=25
PATIENCE=8
LAMBDA_CTR=0.1
P_UNCOND=0.15
LR=1e-3
WEIGHT_DECAY=1e-4
SPLIT_SEED=20261101
TRAIN_SEED=20261121
MONITOR_NOISE_SEED=20261131
TRAIN_BATCH=512
MONITOR_BATCH=512
DEVICE="cuda"

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for x in iter(lambda:f.read(1<<20),b""): h.update(x)
    return h.hexdigest()

def beta_tag(beta): return ("%.1f"%float(beta)).replace(".","p")

def split_train_monitor(ids):
    ids=np.asarray(ids,dtype=np.int64).copy()
    rng=np.random.RandomState(SPLIT_SEED); rng.shuffle(ids)
    n_monitor=max(1,int(round(len(ids)*0.05)))
    monitor_ids=np.sort(ids[:n_monitor]); train_ids=np.sort(ids[n_monitor:])
    assert len(np.intersect1d(train_ids,monitor_ids))==0
    assert len(train_ids)+len(monitor_ids)==len(ids)
    return train_ids,monitor_ids

def make_x(raw_t,raw_v,ids):
    return np.concatenate([l2_rows_np(raw_t[ids]),l2_rows_np(raw_v[ids])],axis=1).astype(np.float32,copy=False)

def monitor_objective(model,raw_t,raw_v,cond,ids,ab):
    model.eval(); total=rec=ctr=0.0; n=0
    with torch.no_grad():
        for bi,st in enumerate(range(0,len(ids),MONITOR_BATCH)):
            ii=ids[st:st+MONITOR_BATCH]
            x0=torch.from_numpy(make_x(raw_t,raw_v,ii)).to(DEVICE)
            cc=torch.from_numpy(np.asarray(cond[ii],np.float32)).to(DEVICE)
            gen=torch.Generator(device=DEVICE); gen.manual_seed(MONITOR_NOISE_SEED+bi)
            tt=torch.randint(1,51,(len(ii),),generator=gen,device=DEVICE)
            noise=torch.randn(x0.shape,generator=gen,device=DEVICE,dtype=x0.dtype)
            loss,info=training_loss(model,x0,cc,tt,noise,ab,lambda_ctr=LAMBDA_CTR,p_uncond=0.0)
            bs=len(ii); n+=bs
            total+=float(loss)*bs; rec+=float(info["loss_rec"])*bs; ctr+=float(info["loss_ctr"])*bs
    return {"monitor_objective":total/n,"monitor_rec":rec/n,"monitor_ctr":ctr/n}

def train_one_epoch(beta,epoch,model,opt,raw_t,raw_v,cond,train_ids,ab):
    model.train(); order=train_ids.copy()
    rng=np.random.RandomState(TRAIN_SEED+int(beta*1000)+epoch); rng.shuffle(order)
    vals=[]; t0=time.time()
    for st in range(0,len(order),TRAIN_BATCH):
        ii=order[st:st+TRAIN_BATCH]
        x0=torch.from_numpy(make_x(raw_t,raw_v,ii)).to(DEVICE)
        cc=torch.from_numpy(np.asarray(cond[ii],np.float32)).to(DEVICE)
        tt=torch.randint(1,51,(len(ii),),device=DEVICE)
        noise=torch.randn_like(x0)
        loss,info=training_loss(model,x0,cc,tt,noise,ab,lambda_ctr=LAMBDA_CTR,p_uncond=P_UNCOND)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        vals.append((len(ii),float(loss),info))
    denom=sum(x[0] for x in vals)
    def wav(key):
        if key=="loss": return sum(bs*v for bs,v,_ in vals)/denom
        return sum(bs*float(info[key]) for bs,_,info in vals)/denom
    return {"epoch":epoch,"train_loss":wav("loss"),"train_rec":wav("loss_rec"),"train_ctr":wav("loss_ctr"),
            "train_e_text":wav("e_text"),"train_e_visual":wav("e_visual"),"unconditional_ratio":wav("unconditional_ratio"),
            "train_ctr_e_true":wav("ctr_e_true"),"train_ctr_e_shuffled":wav("ctr_e_shuffled"),"epoch_runtime_sec":time.time()-t0}

def make_model(beta,dim,cond_dim):
    seed=TRAIN_SEED+int(beta*1000)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model=NativeTVX0Denoiser(dim,cond_dim=cond_dim,hidden=1024,time_dim=64).to(DEVICE)
    return model,torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)

def mechanism_audit(model,raw_t,raw_v,cond,monitor_ids,ab):
    ids=monitor_ids[:min(len(monitor_ids),1024)]
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
                e=paired_errors(model,x0,cc,tt,noise,ab)
                for k,v in e.items(): accum[k].append(v.detach().cpu().numpy())
    means={k:float(np.concatenate(v).mean()) for k,v in accum.items()}
    means["true_better_shuffled_balanced"]=bool(means["true_balanced"]<means["shuf_balanced"])
    means["true_better_null_balanced"]=bool(means["true_balanced"]<means["null_balanced"])
    return means

def run(dataset,msca_assets,out_dir,mode):
    cfg=load_dataset_config(dataset); dataset=cfg["dataset"]; paths=cfg["resolved_paths"]
    out_dir=Path(out_dir); asset_dir=out_dir/"assets"; checkpoint_dir=out_dir/"checkpoints"; evidence_dir=out_dir/"evidence"
    for p in (asset_dir,checkpoint_dir,evidence_dir): p.mkdir(parents=True,exist_ok=True)
    asset_audit=build_current_run_assets(dataset,Path(msca_assets),asset_dir)
    ep=np.load(asset_dir/"condition_endpoints.npz")
    collab=ep["collab_item"].astype(np.float32); final=ep["final_item"].astype(np.float32)
    all_train=np.load(asset_dir/"train_item_ids.npy",allow_pickle=False)
    train_ids,monitor_ids=split_train_monitor(all_train)
    np.save(asset_dir/"diffusion_train_item_ids.npy",train_ids,allow_pickle=False)
    np.save(asset_dir/"diffusion_monitor_item_ids.npy",monitor_ids,allow_pickle=False)
    raw_t=np.load(paths["text_feature"],mmap_mode="r",allow_pickle=False)
    raw_v=np.load(paths["visual_feature"],mmap_mode="r",allow_pickle=False)
    dim=int(raw_t.shape[1]+raw_v.shape[1]); cond_dim=int(collab.shape[1])
    if dim!=4480 or raw_t.shape[1]!=384 or raw_v.shape[1]!=4096: raise RuntimeError("native dimensions changed")
    betas=[float(x) for x in cfg["diffusion"]["beta_candidates"]]
    ab=cosine_alpha_bar(50).to(DEVICE)
    if mode=="smoke":
        train_ids=train_ids[:min(1024,len(train_ids))]; monitor_ids=monitor_ids[:min(256,len(monitor_ids))]
        smoke={"phase":"DIFFUSION_TRAINING_SMOKE","dataset":dataset,"results":{},"TEST_ACCESSED":False}
    for beta in betas:
        cond=condition_beta(collab,final,beta)
        cond_path=asset_dir/f"condition_beta_{beta_tag(beta)}.npy"; np.save(cond_path,cond,allow_pickle=False)
        model,opt=make_model(beta,dim,cond_dim)
        best={"value":float("inf"),"epoch":0,"state_dict":None,"epochs_since_best":0}; logs=[]
        end=2 if mode=="smoke" else MAX_EPOCHS
        stop_reason="MAX_EPOCHS_REACHED"; stop_epoch=end
        for epoch in range(1,end+1):
            rec=train_one_epoch(beta,epoch,model,opt,raw_t,raw_v,cond,train_ids,ab)
            mon=monitor_objective(model,raw_t,raw_v,cond,monitor_ids,ab); rec.update(mon); logs.append(rec)
            if mon["monitor_objective"]<best["value"]-1e-12:
                best={"value":float(mon["monitor_objective"]),"epoch":epoch,
                      "state_dict":{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},"epochs_since_best":0}
            else: best["epochs_since_best"]+=1
            if epoch==1 or epoch==5 or epoch%5==0:
                print("DIFFUSION_TRAIN_EPOCH",dataset,beta,json.dumps({k:v for k,v in rec.items() if k!="epoch_runtime_sec"},sort_keys=True),flush=True)
            if mode=="formal" and epoch>=MIN_EPOCHS and best["epochs_since_best"]>=PATIENCE:
                stop_reason="EARLY_STOP_PATIENCE"; stop_epoch=epoch; break
        if mode=="smoke":
            smoke["results"][str(beta)]={"finite":bool(np.isfinite(best["value"])),"best_monitor":best["value"],"logs":logs}
        else:
            checkpoint_path=checkpoint_dir/f"{dataset}_beta_{beta_tag(beta)}.pt"
            torch.save({"state_dict":best["state_dict"],"D":dim,"hidden":1024,"beta":beta,
                        "source_msca_checkpoint_sha256":asset_audit["source_checkpoint_sha256"]},checkpoint_path)
            model.load_state_dict(best["state_dict"],strict=True)
            mechanism=mechanism_audit(model,raw_t,raw_v,cond,monitor_ids,ab)
            meta={"phase":"DIFFUSION_FORMAL_TRAINING","dataset":dataset,"beta":beta,
                  "architecture":"4480->1024->512->1024->4480","prediction":"x0","schedule":"cosine",
                  "diffusion_steps":50,"time_embedding":64,"lambda_ctr":LAMBDA_CTR,"p_uncond":P_UNCOND,
                  "optimizer":"AdamW","lr":LR,"weight_decay":WEIGHT_DECAY,"MAX_EPOCHS":MAX_EPOCHS,
                  "MIN_EPOCHS":MIN_EPOCHS,"PATIENCE":PATIENCE,"monitor_split_seed":SPLIT_SEED,
                  "monitor_allowed_objectives":["reconstruction_loss","contrastive_denoising_loss"],
                  "recommendation_validation_metric_used_for_early_stopping":False,
                  "train_item_count":int(len(train_ids)),"monitor_item_count":int(len(monitor_ids)),
                  "best_epoch":int(best["epoch"]),"best_monitor_objective":float(best["value"]),
                  "stop_epoch":int(stop_epoch),"stop_reason":stop_reason,"checkpoint":str(checkpoint_path),
                  "checkpoint_sha256":sha256(checkpoint_path),"condition_path":str(cond_path),
                  "condition_sha256":sha256(cond_path),"source_msca_checkpoint_sha256":asset_audit["source_checkpoint_sha256"],
                  "mechanism":mechanism,"epoch_log":logs,"TEST_ACCESSED":False,"VALIDATION_RANKING_USED_FOR_TRAINING":False}
            (evidence_dir/f"beta_{beta_tag(beta)}_training.json").write_text(json.dumps(meta,indent=2)+"\n")
            print("DIFFUSION_TRAIN_DONE",dataset,beta,json.dumps({"best_epoch":best["epoch"],"stop_epoch":stop_epoch,
                  "stop_reason":stop_reason,"best_monitor":best["value"],"sha256":meta["checkpoint_sha256"]},sort_keys=True),flush=True)
        del model,opt,best; torch.cuda.empty_cache(); gc.collect()
    if mode=="smoke":
        smoke["PASS"]=all(x["finite"] for x in smoke["results"].values())
        (evidence_dir/"training_smoke.json").write_text(json.dumps(smoke,indent=2)+"\n")
        print("DIFFUSION_TRAINING_SMOKE",json.dumps({"dataset":dataset,"PASS":smoke["PASS"]},sort_keys=True),flush=True)
        if not smoke["PASS"]: raise SystemExit(2)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset",required=True); ap.add_argument("--msca-assets",required=True)
    ap.add_argument("--out",required=True); ap.add_argument("--mode",choices=["smoke","formal"],required=True)
    a=ap.parse_args(); run(a.dataset,a.msca_assets,a.out,a.mode)
