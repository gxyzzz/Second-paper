from __future__ import annotations
import argparse, json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ALL=("R10","N10","R20","N20","R50","N50")

def load(path,required=True):
    p=Path(path)
    if not p.exists():
        if required: raise FileNotFoundError(p)
        return None
    return json.loads(p.read_text())

def metric_row(name,metrics):
    return {"method":name,**{k:float(metrics[k]) for k in ALL}}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default=str(ROOT/"docs/evidence/three_domain_second_paper_summary.json"))
    a=ap.parse_args()

    legacy=load(ROOT/"docs/evidence/diffusion_migration_scoring_parity.json")
    baby=load(ROOT/"docs/evidence/baby_m31_current_run_anchor.json")
    sports=load(ROOT/"docs/evidence/sports_m31_current_run_anchor.json")
    sports_test=load(ROOT/"runs/test/sports_current_frozen/summary.json",required=False)
    elec_base=load(ROOT/"docs/evidence/elec_coliftrec_validation.json")
    elec_freeze=load(ROOT/"runs/diffusion_repro/elec_m32_formal/evidence/ELEC_DIFFUSION_FROZEN_BEFORE_TEST.json",required=False)
    elec_test=load(ROOT/"runs/test/elec_current_frozen/summary.json",required=False)

    out={
      "phase":"THREE_DOMAIN_SECOND_PAPER_REPRODUCTION_SUMMARY",
      "legacy_migration_scoring":legacy,
      "baby":{
        "status":"CHECKPOINT_SENSITIVITY_OBSERVED" if baby.get("BABY_M31_CURRENT_RUN_REPRO")=="FAIL" else "PASS",
        "validation":[
          metric_row("MSCA+Full CoLiftRec",baby["base_metrics"]),
          metric_row("MSCA+Full CoLiftRec+Diffusion",baby["metrics"]),
        ],
        "delta_diffusion_vs_full":baby["delta_vs_full_coliftrec"],
        "U":baby["U"],"primary_positive_count":baby["primary_positive_count"],
        "test_status":"CLOSED",
      },
      "sports":{
        "status":"PASS" if sports.get("SPORTS_M31_CURRENT_RUN_REPRO")=="PASS" else "FAIL",
        "validation":[
          metric_row("MSCA+Full CoLiftRec",sports["base_metrics"]),
          metric_row("MSCA+Full CoLiftRec+Diffusion",sports["metrics"]),
        ],
        "delta_diffusion_vs_full":sports["delta_vs_full_coliftrec"],
        "U":sports["U"],"primary_positive_count":sports["primary_positive_count"],
        "test":sports_test,
      },
      "electronics":{
        "base_validation":elec_base["metrics"],
        "freeze":elec_freeze,
        "test":elec_test,
      },
      "TEST_DRIVEN_TUNING":False,
    }
    Path(a.out).write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps({
      "baby":out["baby"]["status"],
      "sports":out["sports"]["status"],
      "sports_test_ready":sports_test is not None,
      "elec_validation_ready":elec_freeze is not None,
      "elec_test_ready":elec_test is not None,
    },sort_keys=True))

if __name__=="__main__":
    main()
