from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))

from modules.preference_diffusion import PreferenceDiffusion
from paper_experiments.common import checkpoint_info, load_model


def nparams(module, trainable_only=True):
    params = module.parameters()
    if trainable_only:
        params = [p for p in params if p.requires_grad]
    return int(sum(p.numel() for p in params))


def main():
    out = {"seed": 1000, "datasets": {}, "definitions": {
        "coliftrec_trainable_parameters": 0,
        "pdbr_inference_parameters": 0,
        "note": "PDBR diffusion is training-only and is not deployed for ranking inference.",
    }}
    diff = PreferenceDiffusion()
    diff_params = nparams(diff)
    for dataset in ("baby", "sports", "elec"):
        model, _, info = load_model(dataset, "b0", gpu_id=1 if dataset != "elec" else 0)
        backbone = nparams(model)
        out["datasets"][dataset] = {
            "backbone_trainable_parameters": backbone,
            "coliftrec_trainable_parameters": 0,
            "pdbr_diffusion_trainable_parameters": diff_params,
            "MSCA_total_training_parameters": backbone,
            "MSCA_plus_CoLiftRec_total_training_parameters": backbone,
            "MSCA_plus_CoLiftRec_plus_PDBR_total_training_parameters": backbone + diff_params,
            "MSCA_deployed_inference_parameters": backbone,
            "MSCA_plus_CoLiftRec_deployed_inference_parameters": backbone,
            "MSCA_plus_CoLiftRec_plus_PDBR_deployed_inference_parameters": backbone,
            "PDBR_training_time_parameters": diff_params,
            "PDBR_inference_time_parameters": 0,
            "checkpoint_sha": info["sha256"],
        }
        del model
    path = ROOT / "paper_experiments/efficiency/evidence/PARAMETER_COUNT.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))

if __name__ == "__main__": main()
