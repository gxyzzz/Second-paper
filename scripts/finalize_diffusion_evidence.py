from __future__ import annotations
import argparse, json, shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SEEDS=[20261111,20261112,20261113,20261114,20261115]

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--dataset",default="baby")
 ap.add_argument("--diffusion-dir",required=True)
 a=ap.parse_args()
 ds=a.dataset; up=ds.upper(); d=Path(a.diffusion_dir); e=d/"evidence"
 paths={
  "pool":e/f"{up}_DIFFUSION_FROZEN_CANDIDATE_POOL.json",
  "cross":e/f"{ds}_crossfit_summary.json",
  "freeze":e/f"{up}_DIFFUSION_FROZEN_BEFORE_TEST.json",
  "grid":e/f"{ds}_validation_grid.json",
  "manifest":e/"purified_asset_manifest.json",
 }
 for p in paths.values():
  if not p.exists(): raise FileNotFoundError(p)
 pool=json.loads(paths["pool"].read_text()); cross=json.loads(paths["cross"].read_text()); freeze=json.loads(paths["freeze"].read_text()); grid=json.loads(paths["grid"].read_text()); manifest=json.loads(paths["manifest"].read_text())
 assert pool["frozen_before_crossfit"] is True and pool["MAX_SHORTLIST"]==8 and len(pool["configs"])<=8
 assert pool["TEST_ACCESSED"] is False and cross["TEST_ACCESSED"] is False and grid["TEST_ACCESSED"] is False and manifest["TEST_ACCESSED"] is False
 assert cross["K"]==5 and cross["REPEATS"]==5 and cross["seeds"]==SEEDS and cross["OOF_total"]==25
 gate=bool(cross["OOF_mean_delta_U"]>0 and cross["P_delta_U_gt_0"]>=0.80 and cross["repeat_positive_mean_count"]>=4)
 assert gate==bool(cross[f"{up}_DIFFUSION_UPGRADE_PASS"])
 assert freeze["TEST_USED_FOR_SELECTION"] is False and freeze[f"{up}_DIFFUSION_TEST"]=="CLOSED"
 final=freeze["final_config"]
 if gate:
  assert final["id"]!="NO_DIFFUSION"
  assert final["primary_positive_count"]>=3 and final["U"]>0 and final["sum_primary_delta"]>0
 else:
  assert final["id"]=="NO_DIFFUSION"
 diff=[r for r in grid["rows"] if r["kind"]=="DIFFUSION"]
 best={}
 for beta in (0.5,1.0):
  xs=[r for r in diff if float(r["beta"])==beta]
  best[str(beta)]=max(xs,key=lambda x:x["U"])
 summary={
  "phase":f"{up}_DIFFUSION_FINAL_VALIDATION_SUMMARY","dataset":ds,
  "base_metrics":grid["base_metrics"],"candidate_count":grid["candidate_count"],
  "best_validation_by_beta":best,
  "crossfit":{"OOF_mean_delta_U":cross["OOF_mean_delta_U"],"P_delta_U_gt_0":cross["P_delta_U_gt_0"],"repeat_positive_mean_count":cross["repeat_positive_mean_count"],"selected_config_frequency":cross["selected_config_frequency"],"upgrade_pass":gate},
  "final_config":final,"final_reason":freeze["final_reason"],
  f"{up}_DIFFUSION_TEST":"CLOSED","TEST_USED_FOR_SELECTION":False
 }
 out=ROOT/"docs/evidence"
 shutil.copy2(paths["pool"],out/f"{ds}_diffusion_frozen_candidate_pool.json")
 shutil.copy2(paths["cross"],out/f"{ds}_diffusion_crossfit_summary.json")
 shutil.copy2(paths["freeze"],out/f"{ds}_diffusion_frozen_before_test.json")
 (out/f"{ds}_diffusion_validation_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
 print(json.dumps({"dataset":ds,"upgrade_pass":gate,"final_config":final["id"],f"{up}_DIFFUSION_TEST":"CLOSED"},sort_keys=True))
if __name__=="__main__": main()
