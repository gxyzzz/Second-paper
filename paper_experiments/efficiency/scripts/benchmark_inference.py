from __future__ import annotations

import argparse
import csv
import json
import platform
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))

from modules.ranking import topk_from_embeddings
from pipelines.preference_pipeline import DomainEvaluator
from paper_experiments.common import load_model


def sync(device):
    if device.type == "cuda": torch.cuda.synchronize(device)

@torch.no_grad()
def msca_inference(model, evaluator):
    model.eval(); user_emb, item_emb = model.forward(test=True)
    return topk_from_embeddings(user_emb, item_emb, evaluator.users, evaluator.histories, 100, 1024)


def bench(fn, device, repeats=5):
    fn(); sync(device)
    values=[]; peaks=[]
    for _ in range(repeats):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
        sync(device); t0=time.perf_counter(); fn(); sync(device); dt=time.perf_counter()-t0
        values.append(dt); peaks.append(torch.cuda.max_memory_allocated(device)/(1024**3))
    return {
        "all_repeats_seconds": values,
        "median_seconds": float(statistics.median(values)),
        "mean_seconds": float(statistics.mean(values)),
        "std_seconds": float(statistics.pstdev(values)),
        "peak_cuda_allocated_GB": float(max(peaks)),
    }


def one(dataset, gpu_id):
    b0, _, binfo = load_model(dataset, "b0", gpu_id)
    p1, _, pinfo = load_model(dataset, "p1", gpu_id)
    eb = DomainEvaluator(dataset, b0, split="validation")
    ep = DomainEvaluator(dataset, p1, split="validation")
    device=b0.device; n=int(len(eb.users))
    methods={
        "MSCA": lambda: msca_inference(b0, eb),
        "MSCA+CoLiftRec": lambda: eb.evaluate(b0),
        "MSCA+CoLiftRec+PDBR": lambda: ep.evaluate(p1),
    }
    rows=[]
    for method, fn in methods.items():
        result=bench(fn, device)
        row={"dataset":dataset,"method":method,"users":n,**result,
             "median_ms_per_user":result["median_seconds"]*1000/n,
             "mean_ms_per_user":result["mean_seconds"]*1000/n,
             "diffusion_inference_calls":0}
        rows.append(row)
    return rows, {"b0":binfo["sha256"],"p1":pinfo["sha256"]}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset",choices=["baby","sports","elec","all"],default="all"); ap.add_argument("--gpu-id",type=int,default=0); a=ap.parse_args()
    ds=["baby","sports","elec"] if a.dataset=="all" else [a.dataset]
    rows=[]; cps={}
    for d in ds:
        print(f"[Inference benchmark] {d}",flush=True); r,c=one(d,a.gpu_id); rows+=r; cps[d]=c
    ev=ROOT/"paper_experiments/efficiency/evidence"; ev.mkdir(parents=True,exist_ok=True)
    fields=["dataset","method","users","median_seconds","mean_seconds","std_seconds","median_ms_per_user","mean_ms_per_user","peak_cuda_allocated_GB","diffusion_inference_calls","all_repeats_seconds"]
    with (ev/"INFERENCE_BENCHMARK.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator=chr(10)); w.writeheader();
        for row in rows:
            rr=dict(row); rr["all_repeats_seconds"]=json.dumps(rr["all_repeats_seconds"]); w.writerow(rr)
    cpu=platform.processor().strip()
    if not cpu:
        text=Path("/proc/cpuinfo").read_text(errors="ignore") if Path("/proc/cpuinfo").exists() else ""
        m=re.search(r"^model name\s*:\s*(.+)$",text,re.MULTILINE); cpu=m.group(1).strip() if m else "unknown"
    batch_sizes={}
    for d in ds:
        probe, cfg, _info = load_model(d, "b0", a.gpu_id)
        batch_sizes[d]={
            "train_batch_size": int(cfg["train_batch_size"]),
            "eval_batch_size": int(cfg["eval_batch_size"]),
        }
        del probe
        torch.cuda.empty_cache()
    env={"gpu_name":torch.cuda.get_device_name(a.gpu_id),"cuda_version":torch.version.cuda,"pytorch_version":torch.__version__,"python_version":platform.python_version(),"cpu":cpu,"batch_size":batch_sizes,"datasets":ds,"seed":1000,"commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"timestamp":time.strftime("%Y-%m-%dT%H:%M:%S%z"),"checkpoint_sha":cps}
    (ev/"ENVIRONMENT.json").write_text(json.dumps(env,indent=2)+"\n",encoding="utf-8")
    print(f"wrote {ev/'INFERENCE_BENCHMARK.csv'}")

if __name__=="__main__": main()
