#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/"src"
sys.path.insert(0,str(SRC))
import torch
from pipelines.coliftrec import run as run_coliftrec
from pipelines.dataset_config import canonical_dataset, load_dataset_config
from pipelines.diffusion_train import run as run_diffusion_train
from pipelines.msca_assets import export_validation_assets
from pipelines.publication_eval import evaluate_test, evaluate_validation, generate_fixed_purified

def find_checkpoint(dataset,seed=999):
    candidates=[]
    for path in (ROOT/"runs/checkpoints").glob("*.pth"):
        try:
            state=torch.load(path,map_location="cpu",weights_only=False); config=state.get("config",{})
            if config.get("dataset")==dataset and int(config.get("seed",-1))==seed:
                candidates.append((path.stat().st_mtime,path))
        except Exception:
            continue
    if not candidates:
        raise RuntimeError(f"No MSCA checkpoint for dataset={dataset}, seed={seed}; run --stage msca or pass --checkpoint.")
    return max(candidates)[1]

def workspace_paths(dataset,workdir=None):
    base=workdir or ROOT/"runs/reproduction"/dataset
    return {
        "base":base,"assets":base/"msca_assets","coliftrec":base/"coliftrec",
        "diffusion":base/"diffusion","purified":base/"purified",
        "validation":base/"validation","test":base/"test"}

def print_plan(dataset,stage,gpu,paths,checkpoint):
    cfg=load_dataset_config(dataset); d=cfg["diffusion"]
    print(json.dumps({
        "dataset":dataset,"stage":stage,"gpu":gpu,
        "publication_seed":cfg["backbone"]["seed"],
        "checkpoint":str(checkpoint) if checkpoint else None,
        "workspace":{k:str(v) for k,v in paths.items()},
        "method_config":cfg["_method_config_path"],
        "diffusion":{k:d[k] for k in (
            "training_protocol","train_scope","beta","training_seed","t_edit",
            "guidance","rho_text","rho_visual","purification_seeds","checkpoint_selection")}},indent=2))

def run_msca(dataset,gpu):
    subprocess.run(
        [sys.executable,"main.py","-m","MSCA","-d",dataset,"--gpu-id",str(gpu)],
        cwd=SRC,check=True)
    return find_checkpoint(dataset,999)

def ensure_assets(dataset,gpu,checkpoint,paths):
    audit=paths["assets"]/"audit.json"
    if audit.exists(): return paths["assets"]
    checkpoint=checkpoint or find_checkpoint(dataset,999)
    export_validation_assets(checkpoint,paths["assets"],gpu_id=gpu)
    return paths["assets"]

def require(path,hint):
    if not path.exists(): raise RuntimeError(f"Required artifact missing: {path}. {hint}")

def execute(dataset,stage,gpu,checkpoint,workdir):
    dataset=canonical_dataset(dataset); cfg=load_dataset_config(dataset)
    if int(cfg["backbone"]["seed"])!=999: raise RuntimeError("Publication backbone seed must remain 999")
    paths=workspace_paths(dataset,workdir); paths["base"].mkdir(parents=True,exist_ok=True)
    if stage=="msca":
        print(f"MSCA_CHECKPOINT={run_msca(dataset,gpu)}"); return
    if stage=="coliftrec":
        assets=ensure_assets(dataset,gpu,checkpoint,paths)
        run_coliftrec(dataset,assets,paths["coliftrec"]); return
    if stage=="diffusion":
        assets=ensure_assets(dataset,gpu,checkpoint,paths)
        run_diffusion_train(dataset,assets,paths["diffusion"],"formal",betas_override=[float(cfg["diffusion"]["beta"])])
        generate_fixed_purified(dataset,paths["diffusion"],paths["purified"]); return
    if stage=="validation":
        assets=ensure_assets(dataset,gpu,checkpoint,paths)
        require(paths["coliftrec"]/"validation_scores.npz","Run --stage coliftrec first.")
        require(paths["purified"]/"fixed_text.npy","Run --stage diffusion first.")
        evaluate_validation(dataset,assets,paths["coliftrec"],paths["purified"]/"fixed_text.npy",paths["purified"]/"fixed_visual.npy",paths["validation"]); return
    if stage=="test":
        assets=ensure_assets(dataset,gpu,checkpoint,paths)
        require(paths["purified"]/"fixed_text.npy","Run --stage diffusion first.")
        evaluate_test(dataset,assets,paths["purified"]/"fixed_text.npy",paths["purified"]/"fixed_visual.npy",paths["test"]); return
    if stage=="all":
        ck=run_msca(dataset,gpu)
        assets=ensure_assets(dataset,gpu,ck,paths)
        run_coliftrec(dataset,assets,paths["coliftrec"])
        run_diffusion_train(dataset,assets,paths["diffusion"],"formal",betas_override=[float(cfg["diffusion"]["beta"])])
        generate_fixed_purified(dataset,paths["diffusion"],paths["purified"])
        evaluate_validation(dataset,assets,paths["coliftrec"],paths["purified"]/"fixed_text.npy",paths["purified"]/"fixed_visual.npy",paths["validation"])
        evaluate_test(dataset,assets,paths["purified"]/"fixed_text.npy",paths["purified"]/"fixed_visual.npy",paths["test"]); return
    raise ValueError(stage)

def main():
    ap=argparse.ArgumentParser(description="Reproduce the frozen publication pipeline.")
    ap.add_argument("--dataset",required=True,choices=["baby","sports","elec","electronics"])
    ap.add_argument("--stage",default="all",choices=["msca","coliftrec","diffusion","validation","test","all"])
    ap.add_argument("--gpu",type=int,default=0)
    ap.add_argument("--checkpoint")
    ap.add_argument("--workdir")
    ap.add_argument("--dry-run",action="store_true")
    a=ap.parse_args()
    dataset=canonical_dataset(a.dataset)
    checkpoint=Path(a.checkpoint).resolve() if a.checkpoint else None
    workdir=Path(a.workdir).resolve() if a.workdir else None
    paths=workspace_paths(dataset,workdir)
    if a.dry_run:
        print_plan(dataset,a.stage,a.gpu,paths,checkpoint); return
    execute(dataset,a.stage,a.gpu,checkpoint,workdir)

if __name__=="__main__":
    main()
