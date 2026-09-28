from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/"src"
sys.path.insert(0,str(SRC))
os.chdir(SRC)

import torch
from utils.quick_start import quick_start
from modules.ranking import sha256_file

def run(seed:int,out:Path,gpu_id:int=0,smoke:bool=False):
    out = out if out.is_absolute() else (ROOT / out)
    out=out.resolve()
    ckpt=out/"msca"/"checkpoints"
    ckpt.mkdir(parents=True,exist_ok=True)
    cfg={
        "gpu_id":int(gpu_id),
        "seed":[int(seed)],
        "checkpoint_dir":str(ckpt),
        "show_progress":False,
    }
    if smoke:
        cfg["epochs"]=2
        cfg["stopping_step"]=2
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        torch.cuda.reset_peak_memory_stats(gpu_id)
    result=quick_start("MSCA","baby",cfg,save_model=True)
    cp=Path(result["checkpoint"]).resolve()
    ck=torch.load(cp,map_location="cpu",weights_only=False)
    peak=int(torch.cuda.max_memory_allocated(gpu_id)) if torch.cuda.is_available() else 0
    evidence={
        "phase":"BABY_MSCA_BACKBONE_ROBUSTNESS_SMOKE" if smoke else "BABY_MSCA_BACKBONE_ROBUSTNESS_TRAIN",
        "dataset":"baby","seed":int(seed),"random_init":True,
        "validation_only":True,"TEST_ACCESSED":False,"TEST_USED_FOR_SELECTION":False,
        "smoke":bool(smoke),"checkpoint":str(cp),"checkpoint_sha256":sha256_file(cp),
        "best_epoch":int(ck["epoch"]),
        "official_msca_source_commit":"48455de8efa943e16d49db665e7f2fcb0c6c5e17",
        "selection_metric":"Recall@20","epochs_protocol":1000 if not smoke else 2,
        "stopping_step_protocol":20 if not smoke else 2,
        "train_batch_size":2048,
        "best_valid_score":float(result["valid_score"]),
        "best_valid_result":{k:float(v) for k,v in result["valid_result"].items()},
        "peak_cuda_allocated_bytes":peak,
        "peak_cuda_allocated_gib":peak/(1024**3),
    }
    ep=out/"evidence"; ep.mkdir(parents=True,exist_ok=True)
    (ep/("msca_smoke.json" if smoke else "msca_training.json")).write_text(json.dumps(evidence,indent=2)+"\n")
    print(json.dumps(evidence,sort_keys=True),flush=True)
    return evidence

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--seed",type=int,required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--gpu-id",type=int,default=0)
    ap.add_argument("--smoke",action="store_true")
    a=ap.parse_args()
    run(a.seed,Path(a.out),a.gpu_id,a.smoke)
