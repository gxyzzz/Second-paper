from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from diffusion_validate import build_context, semantic_lift, lift_for_rho

SEEDS=[999,1000,1001,1002]
def paths(s):
    if s==999:
        return (Path("runs/assets/msca_baby_seed999"),
                Path("runs/generic_refactor/baby_formal"),
                Path("runs/backbone_robustness/baby_seed999"))
    r=Path(f"runs/backbone_robustness/baby_seed{s}")
    return r/"msca/assets",r/"coliftrec",r

def row_norm(x):
    x=np.asarray(x,np.float64)
    return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)

def stats(x):
    x=np.asarray(x,np.float64)
    return {"mean":float(x.mean()),"std":float(x.std()),"min":float(x.min()),"max":float(x.max())}

def main():
    out={"phase":"BABY_BACKBONE_SENSITIVITY_DIAGNOSTIC","analysis_only":True,
         "used_for_selection":False,"TEST_ACCESSED":False,"seeds":{}}
    for s in SEEDS:
        ma,co,root=paths(s)
        z=np.load(ma/"embeddings.npz")
        c=row_norm(z["collab_item"]); f=row_norm(z["final_item"])
        cond=np.load(root/"diffusion/assets/condition_beta_0p5.npy" if s!=999
                     else Path("runs/diffusion_rescue/baby_historical_seed_replay/beta_0p5/assets/condition_beta_0p5.npy"))
        q=row_norm(cond)
        geom={
            "cos_collab_final":stats(np.sum(c*f,axis=1)),
            "l2_collab_final":stats(np.linalg.norm(c-f,axis=1)),
            "l2_collab_condition_beta0p5":stats(np.linalg.norm(c-q,axis=1)),
            "cos_collab_condition_beta0p5":stats(np.sum(c*q,axis=1)),
        }
        vs=np.load(co/"validation_scores.npz")
        sc=np.sort(np.asarray(vs["full_coliftrec"],np.float64),axis=1)[:,::-1]
        def gaps(lo,hi):
            # 1-based ranks lo..hi, adjacent score gaps within that window.
            a=sc[:,lo-1:hi]
            g=a[:,:-1]-a[:,1:]
            return {"mean_adjacent_gap":float(g.mean()),"std_adjacent_gap":float(g.std())}
        margins={"rank_8_12":gaps(8,12),"rank_18_22":gaps(18,22),
                 "cutoff_10_11_mean":float((sc[:,9]-sc[:,10]).mean()),
                 "cutoff_20_21_mean":float((sc[:,19]-sc[:,20]).mean())}
        ctx=build_context("baby",ma,co)
        rawt=semantic_lift(ctx["paths"]["text_feature"],ctx,"text")
        rawv=semantic_lift(ctx["paths"]["visual_feature"],ctx,"visual")
        pt=root/"purified/fixed_text.npy"; pv=root/"purified/fixed_visual.npy"
        lt=lift_for_rho(ctx["paths"]["text_feature"],pt,.25,ctx,"text",root/"tmp/diag_t.npy")
        lv=lift_for_rho(ctx["paths"]["visual_feature"],pv,1.0,ctx,"visual",root/"tmp/diag_v.npy")
        dt=np.asarray(lt-rawt,np.float64); dv=np.asarray(lv-rawv,np.float64)
        lift={"text_abs_mean":float(np.abs(dt).mean()),"text_rms":float(np.sqrt(np.mean(dt*dt))),
              "visual_abs_mean":float(np.abs(dv).mean()),"visual_rms":float(np.sqrt(np.mean(dv*dv)))}
        ev=json.load(open(root/"evidence/fixed_diffusion_validation.json"))
        out["seeds"][str(s)]={"PASS":ev["DIFFUSION"]["PASS"],"U":ev["DIFFUSION"]["U"],
                              "geometry":geom,"candidate_score_margins":margins,
                              "diffusion_induced_semantic_lift":lift}
        print("DIAG_DONE",s,ev["DIFFUSION"]["PASS"],ev["DIFFUSION"]["U"],flush=True)
    p=Path("docs/evidence/paper/baby_backbone_sensitivity_diagnostic.json")
    p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps(out,sort_keys=True))
if __name__=="__main__": main()
