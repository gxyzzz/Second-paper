from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import torch
from pipelines.msca_assets import export_validation_assets
from pipelines.coliftrec import run as run_coliftrec
from pipelines.dataset_config import load_dataset_config

ROOT=Path(__file__).resolve().parents[1]
PRIMARY=("R10","N10","R20","N20")
ALL=("R10","N10","R20","N20","R50","N50")

def sha256(p: Path):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for x in iter(lambda:f.read(1<<20),b""): h.update(x)
 return h.hexdigest()

def find_checkpoint(dataset: str, seed: int=999) -> Path:
 candidates=[]
 for p in (ROOT/"runs/checkpoints").glob("*.pth"):
  try:
   z=torch.load(p,map_location="cpu",weights_only=False)
   c=z.get("config",{})
   if c.get("dataset")==dataset and int(c.get("seed",-1))==seed:
    candidates.append((p.stat().st_mtime,p))
  except Exception:
   continue
 if not candidates: raise RuntimeError(f"no checkpoint for {dataset} seed={seed}")
 return max(candidates)[1]

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--dataset",required=True,choices=["sports","elec"])
 ap.add_argument("--checkpoint")
 ap.add_argument("--gpu-id",type=int,default=0)
 a=ap.parse_args()
 cfg=load_dataset_config(a.dataset)
 ck=Path(a.checkpoint) if a.checkpoint else find_checkpoint(a.dataset)
 assets=ROOT/f"runs/assets/msca_{a.dataset}_seed999"
 colift=ROOT/f"runs/validation/{a.dataset}_coliftrec"
 audit=export_validation_assets(ck,assets,gpu_id=a.gpu_id)
 summary=run_coliftrec(a.dataset,assets,colift)
 current=audit["validation_metrics"]; historical=cfg.get("historical_msca_validation")
 if historical:
  delta={k:float(current[k]-historical[k]) for k in ALL}
  max_primary=max(abs(delta[k]) for k in PRIMARY)
  parity="PASS" if max_primary < 0.01 else "REVIEW"
 else:
  delta=None; parity="NO_HISTORICAL_REFERENCE"
 msca_ev={
  "phase":f"{a.dataset.upper()}_MSCA_FROM_SCRATCH","dataset":a.dataset,"seed":999,
  "best_epoch":int(audit["checkpoint_epoch"]),"checkpoint_sha256":audit["checkpoint_sha256"],
  "current_validation":current,"historical_validation":historical,"absolute_delta":delta,
  f"{a.dataset.upper()}_MSCA_FROM_SCRATCH_PARITY":parity,"TEST_ACCESSED":False
 }
 (ROOT/f"docs/evidence/{a.dataset}_msca_from_scratch.json").write_text(json.dumps(msca_ev,indent=2)+"\n")
 (ROOT/f"docs/evidence/{a.dataset}_coliftrec_validation.json").write_text(json.dumps(summary,indent=2)+"\n")
 print(json.dumps({"dataset":a.dataset,"checkpoint":str(ck),"checkpoint_sha256":sha256(ck),
  "parity":parity,"full_primary_positive_count":summary["full_vs_msca_primary_positive_count"],
  "alpha_zero_identity":summary["alpha_zero_identity"],"TEST_ACCESSED":False},sort_keys=True))
if __name__=="__main__": main()
