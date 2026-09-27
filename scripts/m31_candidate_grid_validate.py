from __future__ import annotations
import argparse, gc, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from modules.diffusion import NativeTVX0Denoiser, purify_indices
from modules.ranking import metrics_at, rank_by_score
from pipelines.dataset_config import load_dataset_config
from diffusion_validate import build_context, semantic_lift, lift_for_rho

ROOT=Path(__file__).resolve().parents[1]
PRIMARY=["R10","N10","R20","N20"]; ALL=PRIMARY+["R50","N50"]
T_EDITS=[5,10,15,20,30]; GUIDANCE=[0.5,1.0,1.5,2.0]; RHOS=[0.25,0.5,0.75,1.0]
DEVICE="cuda"

def sha256(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for x in iter(lambda:f.read(1<<20),b""): h.update(x)
 return h.hexdigest()

def load_model(diff_dir):
 p=Path(diff_dir)/"checkpoints/baby_beta_1p0.pt"; z=torch.load(p,map_location="cpu",weights_only=False)
 m=NativeTVX0Denoiser(int(z["D"]),cond_dim=64,hidden=int(z["hidden"]),time_dim=64).to(DEVICE)
 m.load_state_dict(z["state_dict"],strict=True); m.eval(); return m,p,z

def purify_one(model,raw_t,raw_v,cond,ids,t,g,seeds,batch):
 return purify_indices(model,raw_t,raw_v,cond,ids,t_edit=t,guidance=g,seeds=tuple(seeds),batch=batch,device=DEVICE)

def smoke(diff_dir,batch=32):
 cfg=load_dataset_config("baby"); paths=cfg["resolved_paths"]; dcfg=cfg["diffusion"]
 meta=json.loads((Path(diff_dir)/"evidence/beta_1p0_training.json").read_text())
 model,ck,z=load_model(diff_dir)
 assert meta.get("training_protocol")=="m31_fixed_all_items" and meta.get("training_seed")==20261101
 assert sha256(ck)==meta["checkpoint_sha256"]
 raw_t=np.load(paths["text_feature"],mmap_mode="r",allow_pickle=False); raw_v=np.load(paths["visual_feature"],mmap_mode="r",allow_pickle=False)
 cond=np.load(Path(diff_dir)/"assets/condition_beta_1p0.npy",mmap_mode="r",allow_pickle=False); ids=np.arange(min(batch,len(raw_t)),dtype=np.int64)
 ot,ov=purify_one(model,raw_t,raw_v,cond,ids,5,1.5,[int(x) for x in dcfg["purification_seeds"]],batch)
 assert ot.shape==(len(ids),384) and ov.shape==(len(ids),4096) and np.isfinite(ot).all() and np.isfinite(ov).all()
 print(json.dumps({"M31B_GRID_SMOKE":"PASS","gpu":torch.cuda.get_device_name(0),"items":len(ids),"checkpoint_sha256":sha256(ck)},sort_keys=True))

def formal(msca_assets,coliftrec_dir,diff_dir,out_dir,batch=128):
 cfg=load_dataset_config("baby"); paths=cfg["resolved_paths"]; dcfg=cfg["diffusion"]; diff=Path(diff_dir)
 train_meta=json.loads((diff/"evidence/beta_1p0_training.json").read_text())
 req={"training_protocol":"m31_fixed_all_items","train_scope":"all_catalog_items","checkpoint_selection":"final_epoch","training_seed":20261101,"epochs":80,"final_epoch":80,"VALIDATION_TARGET_USED":False,"TEST_TARGET_USED":False,"ALL_CATALOG_SIDE_INFORMATION":True,"TEST_ACCESSED":False}
 for k,v in req.items():
  if train_meta.get(k)!=v: raise RuntimeError(("protocol mismatch",k,train_meta.get(k),v))
 model,ck,z=load_model(diff_dir)
 if sha256(ck)!=train_meta["checkpoint_sha256"]: raise RuntimeError("checkpoint hash mismatch")
 seeds=[int(x) for x in dcfg["purification_seeds"]]
 if seeds!=[20261001,20261002,20261003,20261004]: raise RuntimeError(("seed mismatch",seeds))
 raw_t=np.load(paths["text_feature"],mmap_mode="r",allow_pickle=False); raw_v=np.load(paths["visual_feature"],mmap_mode="r",allow_pickle=False)
 cond=np.load(diff/"assets/condition_beta_1p0.npy",mmap_mode="r",allow_pickle=False); ids=np.arange(len(raw_t),dtype=np.int64)
 out=Path(out_dir); pur=out/"purified"; tmp=out/"tmp"; evid=out/"evidence"
 pur.mkdir(parents=True,exist_ok=True); tmp.mkdir(parents=True,exist_ok=True); evid.mkdir(parents=True,exist_ok=True)
 ctx=build_context("baby",Path(msca_assets),Path(coliftrec_dir)); raw_lt=semantic_lift(paths["text_feature"],ctx,"text"); raw_lv=semantic_lift(paths["visual_feature"],ctx,"visual")
 base_metrics=metrics_at(ctx["base_ranked"],ctx["users"],ctx["eval_sets"]); at=float(cfg["coliftrec"]["text"]["alpha"]); av=float(cfg["coliftrec"]["visual"]["alpha"])
 rows=[]; manifest=[]
 for t in T_EDITS:
  for g in GUIDANCE:
   tag=f"t{t}_g{str(g).replace('.','p')}"; tp=pur/f"{tag}_text.npy"; vp=pur/f"{tag}_visual.npy"
   if not (tp.exists() and vp.exists()):
    t0=time.time(); ot,ov=purify_one(model,raw_t,raw_v,cond,ids,t,g,seeds,batch); np.save(tp,ot.astype(np.float32)); np.save(vp,ov.astype(np.float32))
    manifest.append({"t_edit":t,"guidance":g,"text_sha256":sha256(tp),"visual_sha256":sha256(vp),"runtime_sec":time.time()-t0}); del ot,ov; torch.cuda.empty_cache(); gc.collect()
   else: manifest.append({"t_edit":t,"guidance":g,"text_sha256":sha256(tp),"visual_sha256":sha256(vp),"runtime_sec":0.0,"reused":True})
   lt={r:lift_for_rho(paths["text_feature"],tp,r,ctx,"text",tmp/f"{tag}_t_{r}.npy") for r in RHOS}
   lv={r:lift_for_rho(paths["visual_feature"],vp,r,ctx,"visual",tmp/f"{tag}_v_{r}.npy") for r in RHOS}
   for rt in RHOS:
    for rv in RHOS:
     score=ctx["base_score"].copy(); score+=at*(lt[rt]-raw_lt); score+=av*(lv[rv]-raw_lv)
     ranked=rank_by_score(ctx["items"],score); met=metrics_at(ranked,ctx["users"],ctx["eval_sets"]); delta={k:float(met[k]-base_metrics[k]) for k in ALL}
     rows.append({"t_edit":t,"guidance":g,"rho_T":rt,"rho_V":rv,"metrics":met,"delta_vs_full_coliftrec":delta,"U":float(np.mean([delta[k]/base_metrics[k] for k in PRIMARY])),"primary_positive_count":int(sum(delta[k]>0 for k in PRIMARY)),"sum_primary_delta":float(sum(delta[k] for k in PRIMARY))})
   del lt,lv; gc.collect(); best=max(rows,key=lambda x:x["U"]); print("M31B_DIAG_PROGRESS",tag,"best_U",best["U"],"best",best["t_edit"],best["guidance"],best["rho_T"],best["rho_V"],flush=True)
 best=max(rows,key=lambda x:x["U"]); eligible=[r for r in rows if r["U"]>0 and r["primary_positive_count"]>=3 and r["sum_primary_delta"]>0]
 result={"phase":"BABY_M31B_VALIDATION_ONLY_DIAGNOSTIC","dataset":"baby","candidate_count":len(rows),"checkpoint_sha256":sha256(ck),"training_protocol":"m31_fixed_all_items","training_seed":20261101,"purification_seeds":seeds,"t_edit_candidates":T_EDITS,"guidance_candidates":GUIDANCE,"rho_candidates":RHOS,"base_metrics":base_metrics,"best_by_U":best,"eligible_count":len(eligible),"best_eligible":max(eligible,key=lambda x:x["U"]) if eligible else None,"VALIDATION_ONLY":True,"TEST_ACCESSED":False,"TEST_USED_FOR_SELECTION":False,"scientific_role":"development analysis after exact M31 anchor failed on current-run MSCA"}
 (evid/"baby_m31b_validation_diagnostic.json").write_text(json.dumps(result,indent=2)+"\n"); (evid/"baby_m31b_grid.json").write_text(json.dumps({"base_metrics":base_metrics,"rows":rows,"VALIDATION_ONLY":True,"TEST_ACCESSED":False},indent=2)+"\n"); (evid/"purified_manifest.json").write_text(json.dumps({"assets":manifest,"TEST_ACCESSED":False},indent=2)+"\n")
 print("BABY_M31B_VALIDATION_DIAGNOSTIC_COMPLETE",json.dumps({"best_U":best["U"],"best":{k:best[k] for k in ["t_edit","guidance","rho_T","rho_V","primary_positive_count","sum_primary_delta"]},"eligible_count":len(eligible)},sort_keys=True),flush=True)

if __name__=="__main__":
 ap=argparse.ArgumentParser(); ap.add_argument("--smoke",action="store_true"); ap.add_argument("--msca-assets",default="runs/assets/msca_baby_seed999"); ap.add_argument("--coliftrec-dir",default="runs/validation/baby_coliftrec"); ap.add_argument("--diffusion-dir",default="runs/diffusion_repro/baby_m31_formal"); ap.add_argument("--out",default="runs/diffusion_repro/baby_m31b_diagnostic"); ap.add_argument("--batch",type=int,default=128); a=ap.parse_args()
 smoke(a.diffusion_dir,min(a.batch,32)) if a.smoke else formal(a.msca_assets,a.coliftrec_dir,a.diffusion_dir,a.out,a.batch)
