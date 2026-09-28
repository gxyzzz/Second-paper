from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from modules.ranking import metrics_at,rank_by_score
from diffusion_validate import (
    ALL,PRIMARY,btag,gtag,sha256,build_context,semantic_lift,lift_for_rho,
    per_user_primary,metrics_from_per,baby_grid_summary,make_shortlist,crossfit
)

BETAS=[0.0,0.25,0.5,0.75,1.0]
ROOT=Path(__file__).resolve().parents[1]

def paths(root,beta,shard_tag=None):
    tag=btag(beta); jobs=root/"jobs"
    suffix="" if shard_tag is None else f"_{shard_tag}"
    return jobs/f"score_beta_{tag}{suffix}.json",jobs/f"score_beta_{tag}{suffix}_per.npy"

def score_beta(beta,root,msca_assets,coliftrec_dir,smoke=False,t_values=None,shard_tag=None):
    ctx=build_context("baby",msca_assets,coliftrec_dir); cfg=ctx["cfg"]; p=ctx["paths"]
    raw_lt=semantic_lift(p["text_feature"],ctx,"text"); raw_lv=semantic_lift(p["visual_feature"],ctx,"visual")
    base_metrics=metrics_at(ctx["base_ranked"],ctx["users"],ctx["eval_sets"])
    at=float(cfg["coliftrec"]["text"]["alpha"]); av=float(cfg["coliftrec"]["visual"]["alpha"])
    ts=[int(x) for x in cfg["diffusion"]["t_edit_candidates"]]
    if t_values is not None:
        ts=[t for t in ts if t in set(int(x) for x in t_values)]
        if not ts: raise RuntimeError(("empty t shard",t_values))
    gs=[float(x) for x in cfg["diffusion"]["guidance_candidates"]]
    rs=[float(x) for x in cfg["diffusion"]["rho_text_candidates"]]
    if smoke: ts=ts[:1]; gs=gs[:1]; rs=rs[:1]
    n=len(ts)*len(gs)*len(rs)*len(rs)
    jp,pp=paths(root,beta,shard_tag)
    if smoke:
        jp=root/"jobs"/f"smoke_score_beta_{btag(beta)}.json"
        pp=root/"jobs"/f"smoke_score_beta_{btag(beta)}_per.npy"
    per=np.lib.format.open_memmap(pp,mode="w+",dtype=np.float64,shape=(n,len(ctx["users"]),len(PRIMARY)))
    tmp=root/"tmp_parallel"/f"beta_{btag(beta)}"; tmp.mkdir(parents=True,exist_ok=True)
    rows=[]; idx=0
    for t in ts:
        for g in gs:
            stem=root/"purified"/f"beta_{btag(beta)}"/f"t{t}_g{gtag(g)}"
            tp=Path(str(stem)+"_text.npy"); vp=Path(str(stem)+"_visual.npy")
            if not tp.exists() or not vp.exists(): raise FileNotFoundError((tp,vp))
            lt={r:lift_for_rho(p["text_feature"],tp,r,ctx,"text",tmp/f"t{t}_g{gtag(g)}_T_{r}.npy") for r in rs}
            lv={r:lift_for_rho(p["visual_feature"],vp,r,ctx,"visual",tmp/f"t{t}_g{gtag(g)}_V_{r}.npy") for r in rs}
            for rt in rs:
                for rv in rs:
                    score=ctx["base_score"].copy()+at*(lt[rt]-raw_lt)+av*(lv[rv]-raw_lv)
                    ranked=rank_by_score(ctx["items"],score)
                    met=metrics_at(ranked,ctx["users"],ctx["eval_sets"])
                    pu=per_user_primary(ranked,ctx["users"],ctx["eval_sets"])
                    pam=metrics_from_per(pu)
                    for k in PRIMARY:
                        if abs(pam[k]-met[k])>1e-10: raise RuntimeError(("per-user exactness",beta,k))
                    per[idx]=pu
                    d={k:float(met[k]-base_metrics[k]) for k in ALL}
                    rows.append({"local_index":idx,"beta":beta,"t_edit":t,"guidance":g,"rho_T":rt,"rho_V":rv,
                                 "metrics":met,"delta_vs_no_diffusion":d,
                                 "U":float(np.mean([d[k]/base_metrics[k] for k in PRIMARY])),
                                 "primary_positive_count":int(sum(d[k]>0 for k in PRIMARY)),
                                 "sum_primary_delta":float(sum(d[k] for k in PRIMARY))})
                    idx+=1
            print("SCORE_PROGRESS",beta,t,g,idx,flush=True)
    per.flush(); del per
    out={"phase":"BABY_A2_PARALLEL_BETA_SCORE","beta":beta,"shard_tag":shard_tag,"t_values":ts,"candidate_count":len(rows),
         "base_metrics":base_metrics,"rows":rows,"per_user_path":str(pp),"per_user_sha256":sha256(pp),
         "VALIDATION_ONLY":True,"TEST_ACCESSED":False}
    jp.write_text(json.dumps(out,indent=2)+"\n")
    print("SCORE_BETA_COMPLETE",beta,shard_tag,len(rows),max(x["U"] for x in rows),flush=True)

def combine(root,msca_assets,coliftrec_dir):
    zs=[]
    for b in BETAS:
        for t in [2,3,5,7,10]:
            shard=f"t{t}"
            jp,pp=paths(root,b,shard); z=json.loads(jp.read_text())
            if z["candidate_count"]!=125 or z["TEST_ACCESSED"] is not False or z.get("t_values")!=[t]:
                raise RuntimeError(("score shard invalid",b,t))
            zs.append((b,t,z,np.load(pp,mmap_mode="r",allow_pickle=False)))
    ctx=build_context("baby",msca_assets,coliftrec_dir)
    base_metrics=metrics_at(ctx["base_ranked"],ctx["users"],ctx["eval_sets"])
    base_per=per_user_primary(ctx["base_ranked"],ctx["users"],ctx["eval_sets"])
    master=root/"validation_per_user_primary.npy"
    mm=np.lib.format.open_memmap(master,mode="w+",dtype=np.float64,shape=(3126,len(ctx["users"]),len(PRIMARY)))
    mm[0]=base_per
    rows=[{"id":"NO_DIFFUSION","kind":"NO_DIFFUSION","per_user_index":0,"rho_T":0.0,"rho_V":0.0,
           "metrics":base_metrics,"U":0.0,"primary_positive_count":0,"sum_primary_delta":0.0}]
    gi=1
    for b,t,z,a in zs:
        for r in z["rows"]:
            x=dict(r); li=x.pop("local_index"); x["id"]=f"b_c{gi:04d}"; x["kind"]="DIFFUSION"; x["per_user_index"]=gi
            rows.append(x); mm[gi]=a[li]; gi+=1
    if gi!=3126: raise RuntimeError(("merge count",gi))
    mm.flush(); del mm
    cfg=ctx["cfg"]; grid={"phase":"BABY_DIFFUSION_VALIDATION_GRID","dataset":"baby","candidate_count":3126,
         "diffusion_candidate_count":3125,"rho_T":[float(x) for x in cfg["diffusion"]["rho_text_candidates"]],
         "rho_V":[float(x) for x in cfg["diffusion"]["rho_visual_candidates"]],"base_metrics":base_metrics,
         "rows":rows,"per_user_path":str(master),"per_user_sha256":sha256(master),
         "VALIDATION_ONLY":True,"TEST_ACCESSED":False}
    evid=root/"evidence"; (evid/"baby_validation_grid.json").write_text(json.dumps(grid,indent=2)+"\n")
    summary=baby_grid_summary(root,grid)
    if summary["eligible_count"]==0:
        fail={"phase":"BABY_DIFFUSION_CONDITION_SEARCH_FAIL","reason":"FULL_BETA_GRID_ELIGIBLE_ZERO",
              "BABY_DIFFUSION_TEST":"CLOSED","TEST_USED_FOR_SELECTION":False}
        (evid/"BABY_DIFFUSION_CONDITION_SEARCH_FAIL.json").write_text(json.dumps(fail,indent=2)+"\n")
        print("BABY_DIFFUSION_CONDITION_SEARCH_FAIL",json.dumps(fail),flush=True); return
    shortlist=make_shortlist(grid)
    pool={"phase":"BABY_A3_FROZEN_CANDIDATE_POOL","dataset":"baby","MAX_SHORTLIST":10,
          "frozen_before_crossfit":True,"configs":shortlist,"TEST_ACCESSED":False}
    (evid/"BABY_DIFFUSION_FROZEN_CANDIDATE_POOL.json").write_text(json.dumps(pool,indent=2)+"\n")
    cross=crossfit("baby",pool,grid)
    (evid/"baby_a3_crossfit_summary.json").write_text(json.dumps(cross,indent=2)+"\n")
    if not cross["BABY_DIFFUSION_UPGRADE_PASS"]:
        fail={"phase":"BABY_DIFFUSION_CONDITION_SEARCH_FAIL","reason":"CROSSFIT_FAIL",
              "crossfit_summary":cross,"BABY_DIFFUSION_TEST":"CLOSED","TEST_USED_FOR_SELECTION":False}
        (evid/"BABY_DIFFUSION_CONDITION_SEARCH_FAIL.json").write_text(json.dumps(fail,indent=2)+"\n")
        print("BABY_A3_CROSSFIT_FAIL",json.dumps({"OOF_mean_delta_U":cross["OOF_mean_delta_U"],
              "P_delta_U_gt_0":cross["P_delta_U_gt_0"],"repeat_positive_mean_count":cross["repeat_positive_mean_count"]}),flush=True); return
    eligible=[r for r in rows[1:] if r["U"]>0 and r["primary_positive_count"]>=3 and r["sum_primary_delta"]>0]
    final=max(eligible,key=lambda r:r["U"])
    frozen={"phase":"BABY_A3_VALIDATION_FROZEN_CONFIG","dataset":"baby","final_config":final,
            "selection_rule":"CROSSFIT_PASS_THEN_FULL_VALIDATION_MAX_U_ELIGIBLE",
            "crossfit_summary":cross,"training_seed":20261101,
            "purification_seeds":[20261001,20261002,20261003,20261004],
            "BABY_DIFFUSION_TEST":"CLOSED","TEST_USED_FOR_SELECTION":False,
            "NEXT_REQUIRED":"A4_DIFFUSION_SEED_ROBUSTNESS_20261102_20261103"}
    (evid/"BABY_A3_VALIDATION_FROZEN_CONFIG.json").write_text(json.dumps(frozen,indent=2)+"\n")
    print("BABY_A3_PASS",json.dumps({"beta":final["beta"],"t":final["t_edit"],"g":final["guidance"],
          "rho_T":final["rho_T"],"rho_V":final["rho_V"],"U":final["U"],
          "P_delta_U_gt_0":cross["P_delta_U_gt_0"]}),flush=True)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--mode",choices=["score","combine","smoke"],required=True)
    ap.add_argument("--beta",type=float); ap.add_argument("--t-values",nargs="*",type=int); ap.add_argument("--shard-tag")
    ap.add_argument("--root",default="runs/diffusion_rescue/baby_condition_rescue")
    ap.add_argument("--msca-assets",default="runs/assets/msca_baby_seed999")
    ap.add_argument("--coliftrec-dir",default="runs/generic_refactor/baby_formal"); a=ap.parse_args()
    root=Path(a.root); (root/"jobs").mkdir(parents=True,exist_ok=True)
    if a.mode=="score": score_beta(a.beta,root,Path(a.msca_assets),Path(a.coliftrec_dir),False,a.t_values,a.shard_tag)
    elif a.mode=="smoke": score_beta(a.beta,root,Path(a.msca_assets),Path(a.coliftrec_dir),True,a.t_values,a.shard_tag)
    else: combine(root,Path(a.msca_assets),Path(a.coliftrec_dir))
if __name__=="__main__": main()
