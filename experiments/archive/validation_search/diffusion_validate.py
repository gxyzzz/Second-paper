from __future__ import annotations
import argparse, gc, hashlib, json, time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from modules.coliftrec import shrink_item_background
from modules.diffusion import NativeTVX0Denoiser, l2_rows_np, purify_indices
from modules.ranking import metric_arrays, metrics_at, rank_by_score, row_zscore, semantic_z_for_candidates
from pipelines.dataset_config import load_dataset_config
from pipelines.msca_assets import build_train_histories_and_validation

PRIMARY=["R10","N10","R20","N20"]; SECONDARY=["R50","N50"]; ALL=PRIMARY+SECONDARY
CROSSFIT_SEEDS=[20261111,20261112,20261113,20261114,20261115]
MAX_SHORTLIST=8; BABY_MAX_SHORTLIST=10; DEVICE="cuda"; PURIFY_BATCH=256

def sha256(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for x in iter(lambda:f.read(1<<20),b""): h.update(x)
 return h.hexdigest()
def btag(x): return ("%.1f"%float(x)).replace(".","p")
def gtag(x): return ("%.1f"%float(x)).replace(".","p")
def cfg_tuple(r):
 if r.get("kind")=="NO_DIFFUSION": return ["NO_DIFFUSION",0.,0.,0.,0.]
 return [float(r["beta"]),int(r["t_edit"]),float(r["guidance"]),float(r["rho_T"]),float(r["rho_V"])]

def load_model(diff_dir,dataset,beta):
 p=Path(diff_dir)/"checkpoints"/f"{dataset}_beta_{btag(beta)}.pt"; z=torch.load(p,map_location="cpu",weights_only=False)
 m=NativeTVX0Denoiser(int(z["D"]),cond_dim=64,hidden=int(z["hidden"]),time_dim=64).to(DEVICE)
 m.load_state_dict(z["state_dict"],strict=True); m.eval(); return m,p

def build_context(dataset,msca_assets,coliftrec_dir):
 cfg=load_dataset_config(dataset); paths=cfg["resolved_paths"]; audit=json.loads((Path(msca_assets)/"audit.json").read_text())
 val=np.load(Path(msca_assets)/"validation_top100.npz"); pseudo_asset=np.load(Path(msca_assets)/"train_pseudo_top100.npz")
 scores=np.load(Path(coliftrec_dir)/"validation_scores.npz")
 users=scores["users"].astype(np.int64); items=scores["items"].astype(np.int32); base_score=scores["full_coliftrec"].astype(np.float32)
 if not np.array_equal(users,val["users"]) or not np.array_equal(items,val["items"]): raise RuntimeError("candidate identity mismatch")
 histories,pseudo,pseudo_users,expected_users,eval_sets=build_train_histories_and_validation(paths["interaction"],int(audit["n_users"]))
 if not np.array_equal(users,expected_users) or not np.array_equal(pseudo_users,pseudo_asset["users"]): raise RuntimeError("user identity mismatch")
 return {"cfg":cfg,"paths":paths,"audit":audit,"users":users,"items":items,"base_score":base_score,
  "base_ranked":rank_by_score(items,base_score),"pseudo_items":pseudo_asset["items"].astype(np.int32),
  "histories":histories,"pseudo":pseudo,"pseudo_users":pseudo_users,"eval_sets":eval_sets,"n_items":int(audit["n_items"])}

def semantic_lift(path,ctx,mod):
 ztr,_=semantic_z_for_candidates(path,ctx["pseudo"],ctx["pseudo_users"],ctx["pseudo_items"],batch_users=128)
 zv,_=semantic_z_for_candidates(path,ctx["histories"],ctx["users"],ctx["items"],batch_users=128)
 bg=shrink_item_background(ctx["pseudo_items"],ztr,ctx["n_items"]); lam=float(ctx["cfg"]["coliftrec"][mod]["lambda"])
 return row_zscore(zv-lam*bg["shrunk_mean"][ctx["items"]])

def make_blend(raw_path,pur_path,rho,out_path,batch=1024):
 raw=np.load(raw_path,mmap_mode="r",allow_pickle=False); pur=np.load(pur_path,mmap_mode="r",allow_pickle=False)
 if raw.shape!=pur.shape: raise RuntimeError(("blend shape",raw.shape,pur.shape))
 mm=np.lib.format.open_memmap(out_path,mode="w+",dtype=np.float32,shape=raw.shape)
 for st in range(0,len(raw),batch):
  a=l2_rows_np(raw[st:st+batch]); b=l2_rows_np(pur[st:st+batch]); mm[st:st+batch]=l2_rows_np((1-rho)*a+rho*b)
 mm.flush(); del mm
def lift_for_rho(raw_path,pur_path,rho,ctx,mod,tmp):
 make_blend(raw_path,pur_path,rho,tmp)
 try: return semantic_lift(tmp,ctx,mod)
 finally: Path(tmp).unlink(missing_ok=True)

def per_user_primary(ranked,users,eval_sets):
 _,_,hit,pos_len=metric_arrays(ranked,users,eval_sets,max_k=20); ranks=np.arange(1,21,dtype=np.float64)[None,:]
 recall=np.cumsum(hit,axis=1)/pos_len[:,None]; dcg=np.cumsum(hit/np.log2(ranks+1),axis=1)
 idcg=np.cumsum(np.ones((len(users),20))/np.log2(ranks+1),axis=1)
 for r,plen in enumerate(pos_len):
  cut=min(int(plen),20)
  if cut<20: idcg[r,cut:]=idcg[r,cut-1]
 ndcg=dcg/idcg
 return np.stack([recall[:,9],ndcg[:,9],recall[:,19],ndcg[:,19]],axis=1)
def metrics_from_per(a): return {k:float(v) for k,v in zip(PRIMARY,np.asarray(a).mean(axis=0))}

def generate_assets(dataset,diff_dir):
 cfg=load_dataset_config(dataset); paths=cfg["resolved_paths"]; diff_dir=Path(diff_dir); pur=diff_dir/"purified"; pur.mkdir(parents=True,exist_ok=True)
 purification_seeds=[int(x) for x in cfg["diffusion"]["purification_seeds"]]
 raw_t=np.load(paths["text_feature"],mmap_mode="r",allow_pickle=False); raw_v=np.load(paths["visual_feature"],mmap_mode="r",allow_pickle=False)
 ids=np.arange(len(raw_t),dtype=np.int64); manifest=[]
 for beta in [float(x) for x in cfg["diffusion"]["beta_candidates"]]:
  model,ck=load_model(diff_dir,dataset,beta); cond=np.load(diff_dir/"assets"/f"condition_beta_{btag(beta)}.npy",mmap_mode="r",allow_pickle=False)
  for t in [int(x) for x in cfg["diffusion"]["t_edit_candidates"]]:
   for g in [float(x) for x in cfg["diffusion"]["guidance_candidates"]]:
    d=pur/f"beta_{btag(beta)}"; d.mkdir(parents=True,exist_ok=True); stem=d/f"t{t}_g{gtag(g)}"
    tp=Path(str(stem)+"_text.npy"); vp=Path(str(stem)+"_visual.npy"); mp=Path(str(stem)+".json")
    if tp.exists() and vp.exists() and mp.exists(): manifest.append(json.loads(mp.read_text())); continue
    t0=time.time(); ot,ov=purify_indices(model,raw_t,raw_v,cond,ids,t_edit=t,guidance=g,seeds=tuple(purification_seeds),batch=PURIFY_BATCH,device=DEVICE)
    np.save(tp,ot.astype(np.float32)); np.save(vp,ov.astype(np.float32))
    ev={"beta":beta,"t_edit":t,"guidance":g,"K":4,"noise_seeds":purification_seeds,"checkpoint_sha256":sha256(ck),
        "text_path":str(tp),"text_sha256":sha256(tp),"visual_path":str(vp),"visual_sha256":sha256(vp),"runtime_sec":time.time()-t0}
    mp.write_text(json.dumps(ev,indent=2)+"\n"); manifest.append(ev); del ot,ov; torch.cuda.empty_cache(); gc.collect()
  del model; torch.cuda.empty_cache(); gc.collect()
 (diff_dir/"evidence"/"purified_asset_manifest.json").write_text(json.dumps({"assets":manifest,"TEST_ACCESSED":False},indent=2)+"\n")
 return manifest

def run_grid(dataset,diff_dir,ctx,manifest):
 cfg=ctx["cfg"]; diff_dir=Path(diff_dir); evid=diff_dir/"evidence"; tmp=diff_dir/"tmp"; tmp.mkdir(parents=True,exist_ok=True)
 raw_t=ctx["paths"]["text_feature"]; raw_v=ctx["paths"]["visual_feature"]
 raw_lt=semantic_lift(raw_t,ctx,"text"); raw_lv=semantic_lift(raw_v,ctx,"visual")
 base_metrics=metrics_at(ctx["base_ranked"],ctx["users"],ctx["eval_sets"]); base_per=per_user_primary(ctx["base_ranked"],ctx["users"],ctx["eval_sets"])
 rho_t=[float(x) for x in cfg["diffusion"]["rho_text_candidates"]]; rho_v=[float(x) for x in cfg["diffusion"]["rho_visual_candidates"]]
 rows=[{"id":"NO_DIFFUSION","kind":"NO_DIFFUSION","per_user_index":0,"rho_T":0.,"rho_V":0.,"metrics":base_metrics,"U":0.,"primary_positive_count":0,"sum_primary_delta":0.}]
 total_candidates=1+len(manifest)*len(rho_t)*len(rho_v)
 per_path=diff_dir/"validation_per_user_primary.npy"
 per=np.lib.format.open_memmap(per_path,mode="w+",dtype=np.float64,
                               shape=(total_candidates,len(base_per),len(PRIMARY)))
 per[0]=base_per
 at=float(cfg["coliftrec"]["text"]["alpha"]); av=float(cfg["coliftrec"]["visual"]["alpha"]); idx=1
 for ai,a in enumerate(manifest):
  tp=Path(a["text_path"]); vp=Path(a["visual_path"]); lt={}; lv={}
  for r in rho_t: lt[r]=lift_for_rho(raw_t,tp,r,ctx,"text",tmp/f"a{ai}_t_{r}.npy")
  for r in rho_v: lv[r]=lift_for_rho(raw_v,vp,r,ctx,"visual",tmp/f"a{ai}_v_{r}.npy")
  for rt in rho_t:
   for rv in rho_v:
    score=ctx["base_score"].copy(); score+=at*(lt[rt]-raw_lt); score+=av*(lv[rv]-raw_lv)
    ranked=rank_by_score(ctx["items"],score); met=metrics_at(ranked,ctx["users"],ctx["eval_sets"]); pu=per_user_primary(ranked,ctx["users"],ctx["eval_sets"])
    pam=metrics_from_per(pu)
    for k in PRIMARY:
     if abs(pam[k]-met[k])>1e-10: raise RuntimeError(("per user exactness",k))
    delta={k:float(met[k]-base_metrics[k]) for k in ALL}
    rows.append({"id":f"b_c{idx:03d}","kind":"DIFFUSION","per_user_index":idx,"beta":a["beta"],"t_edit":a["t_edit"],"guidance":a["guidance"],
      "rho_T":rt,"rho_V":rv,"checkpoint_sha256":a["checkpoint_sha256"],"text_sha256":a["text_sha256"],"visual_sha256":a["visual_sha256"],
      "metrics":met,"delta_vs_no_diffusion":delta,"U":float(np.mean([delta[k]/base_metrics[k] for k in PRIMARY])),
      "primary_positive_count":int(sum(delta[k]>0 for k in PRIMARY)),"sum_primary_delta":float(sum(delta[k] for k in PRIMARY))})
    per[idx]=pu; idx+=1
  del lt,lv; gc.collect()
 per.flush(); del per
 grid={"phase":f"{dataset.upper()}_DIFFUSION_VALIDATION_GRID","dataset":dataset,"candidate_count":len(rows),"diffusion_candidate_count":len(rows)-1,
       "rho_T":rho_t,"rho_V":rho_v,"base_metrics":base_metrics,"rows":rows,"per_user_path":str(per_path),"per_user_sha256":sha256(per_path),
       "VALIDATION_ONLY":True,"TEST_ACCESSED":False}
 (evid/f"{dataset}_validation_grid.json").write_text(json.dumps(grid,indent=2)+"\n"); return grid

def make_shortlist(grid):
 diff=grid["rows"][1:]; anchor=grid["rows"][0]
 eligible=[r for r in diff if r["primary_positive_count"]>=3 and r["U"]>0 and r["sum_primary_delta"]>0]
 proposed=[(anchor,"NO_DIFFUSION")]
 for beta in sorted({float(r["beta"]) for r in diff}):
  xs=[r for r in eligible if float(r["beta"])==beta]
  if xs: proposed.append((max(xs,key=lambda x:x["U"]),f"BEST_ELIGIBLE_BETA_{beta}"))
 proposed += [(r,f"OVERALL_{i}") for i,r in enumerate(sorted(eligible,key=lambda x:x["U"],reverse=True),1)]
 limit=BABY_MAX_SHORTLIST if grid.get("dataset")=="baby" else MAX_SHORTLIST
 out=[]; seen=set()
 for r,tag in proposed:
  if r["id"] in seen: continue
  x=dict(r); x["source_tag"]=tag; out.append(x); seen.add(r["id"])
  if len(out)>=limit: break
 return out

def eval_indices(arr,base,idx):
 cm=np.asarray(arr[idx]).mean(axis=0); bm=np.asarray(base[idx]).mean(axis=0); d=cm-bm
 return {"U":float(np.mean(d/bm)),"primary_positive_count":int((d>0).sum()),"sum_primary_delta":float(d.sum()),
         "delta":{k:float(v) for k,v in zip(PRIMARY,d)}}

def crossfit(dataset,pool,grid):
 per=np.load(grid["per_user_path"],mmap_mode="r",allow_pickle=False); base=per[0]; n=len(base); folds=[]; freq=Counter(); repeats=[]
 for ri,seed in enumerate(CROSSFIT_SEEDS,1):
  rng=np.random.RandomState(seed); parts=np.array_split(rng.permutation(n),5); vals=[]
  for fi in range(5):
   hold=np.asarray(parts[fi],dtype=np.int64); sel=np.concatenate([parts[j] for j in range(5) if j!=fi]).astype(np.int64); choices=[]
   for c in pool["configs"]:
    arr=base if c["id"]=="NO_DIFFUSION" else per[c["per_user_index"]]; ev=eval_indices(arr,base,sel)
    eligible=c["id"]!="NO_DIFFUSION" and ev["primary_positive_count"]>=3 and ev["U"]>0 and ev["sum_primary_delta"]>0
    choices.append((c,ev,eligible))
   good=[x for x in choices if x[2]]; chosen=max(good,key=lambda x:x[1]["U"]) if good else next(x for x in choices if x[0]["id"]=="NO_DIFFUSION")
   c,sev,_=chosen; freq[c["id"]]+=1; arr=base if c["id"]=="NO_DIFFUSION" else per[c["per_user_index"]]; hev=eval_indices(arr,base,hold); vals.append(hev["U"])
   folds.append({"repeat":ri,"seed":seed,"fold":fi+1,"selected_config_id":c["id"],"selected_config":cfg_tuple(c),"selection_U":sev["U"],"holdout_delta_U":hev["U"],"holdout_delta_metrics":hev["delta"]})
  x=np.asarray(vals); repeats.append({"repeat":ri,"seed":seed,"mean_delta_U":float(x.mean()),"positive_fold_count":int((x>0).sum()),"mean_positive":bool(x.mean()>0)})
 du=np.asarray([x["holdout_delta_U"] for x in folds]); rep_pos=sum(x["mean_positive"] for x in repeats); passed=bool(du.mean()>0 and np.mean(du>0)>=.80 and rep_pos>=4)
 return {"K":5,"REPEATS":5,"seeds":CROSSFIT_SEEDS,"OOF_mean_delta_U":float(du.mean()),"P_delta_U_gt_0":float(np.mean(du>0)),
         "OOF_positive_count":int((du>0).sum()),"OOF_total":25,"repeat_positive_mean_count":int(rep_pos),"repeat_summaries":repeats,
         "selected_config_frequency":dict(freq),f"{dataset.upper()}_DIFFUSION_UPGRADE_PASS":passed,"folds":folds,"TEST_ACCESSED":False}

def finalize(dataset,diff_dir,grid):
 diff_dir=Path(diff_dir); evid=diff_dir/"evidence"; shortlist=make_shortlist(grid)
 limit=BABY_MAX_SHORTLIST if dataset=="baby" else MAX_SHORTLIST
 pool={"phase":"FROZEN_CANDIDATE_POOL","dataset":dataset,"MAX_SHORTLIST":limit,"frozen_before_crossfit":True,"configs":shortlist,"TEST_ACCESSED":False}
 pool_path=evid/f"{dataset.upper()}_DIFFUSION_FROZEN_CANDIDATE_POOL.json"; pool_path.write_text(json.dumps(pool,indent=2)+"\n")
 cross=crossfit(dataset,pool,grid); (evid/f"{dataset}_crossfit_summary.json").write_text(json.dumps(cross,indent=2)+"\n")
 eligible=[r for r in shortlist if r["id"]!="NO_DIFFUSION" and r["primary_positive_count"]>=3 and r["U"]>0 and r["sum_primary_delta"]>0]
 if cross[f"{dataset.upper()}_DIFFUSION_UPGRADE_PASS"] and eligible: final=max(eligible,key=lambda r:r["U"]); reason="CROSSFIT_PASS_FULL_VALIDATION_MAX_U"
 else: final=shortlist[0]; reason="CROSSFIT_FAIL_NO_DIFFUSION"
 cfg=load_dataset_config(dataset); dcfg=cfg["diffusion"]
 betas=[float(x) for x in dcfg["beta_candidates"]]
 bm=[json.loads((evid/f"beta_{btag(b)}_training.json").read_text()) for b in betas]
 asset_audit=json.loads((diff_dir/"assets"/"audit.json").read_text())
 selected_meta=None if final["id"]=="NO_DIFFUSION" else next(x for x in bm if float(x["beta"])==float(final["beta"]))
 freeze={"phase":f"{dataset.upper()}_DIFFUSION_FROZEN_BEFORE_TEST","dataset":dataset,"final_config":final,"final_reason":reason,
         "msca_checkpoint_sha256":bm[0]["source_msca_checkpoint_sha256"],"coliftrec_config":cfg["coliftrec"],
         "beta_checkpoint_sha256":None if final["id"]=="NO_DIFFUSION" else final["checkpoint_sha256"],
         "training_protocol":dcfg["training_protocol"],"train_item_definition":dcfg["train_scope"],
         "training_seed":None if selected_meta is None else selected_meta.get("training_seed"),
         "purification_seeds":[int(x) for x in dcfg["purification_seeds"]],
         "canonical_text_sha256":asset_audit["text_sha256"],"canonical_visual_sha256":asset_audit["visual_sha256"],
         "selected_condition_sha256":None if selected_meta is None else selected_meta["condition_sha256"],
         "candidate_pool_sha256":sha256(pool_path),"selected_beta":None if final["id"]=="NO_DIFFUSION" else final["beta"],
         "t_edit":None if final["id"]=="NO_DIFFUSION" else final["t_edit"],"guidance":None if final["id"]=="NO_DIFFUSION" else final["guidance"],
         "rho_T":0. if final["id"]=="NO_DIFFUSION" else final["rho_T"],"rho_V":0. if final["id"]=="NO_DIFFUSION" else final["rho_V"],
         "validation_metrics":final["metrics"],"crossfit_summary":cross,"TEST_USED_FOR_SELECTION":False,f"{dataset.upper()}_DIFFUSION_TEST":"CLOSED"}
 (evid/f"{dataset.upper()}_DIFFUSION_FROZEN_BEFORE_TEST.json").write_text(json.dumps(freeze,indent=2)+"\n"); return freeze

def baby_grid_summary(diff,grid):
 rows=grid["rows"][1:]
 def eligible(r): return r["U"]>0 and r["primary_positive_count"]>=3 and r["sum_primary_delta"]>0
 def depth_safe(r):
  d=r["delta_vs_no_diffusion"]
  return d["R50"]>=-5e-4 and d["N50"]>=-5e-4
 by_beta={}
 for beta in sorted({float(r["beta"]) for r in rows}):
  xs=[r for r in rows if float(r["beta"])==beta]
  by_beta[str(beta)]={
   "eligible_config_count":int(sum(eligible(r) for r in xs)),
   "mean_U":float(np.mean([r["U"] for r in xs])),
   "best_U":float(max(r["U"] for r in xs)),
   "primary_4of4_count":int(sum(r["primary_positive_count"]==4 for r in xs)),
   "depth_safe_count":int(sum(depth_safe(r) for r in xs)),
   "best_config":max(xs,key=lambda r:r["U"]),
  }
 ds=[r for r in rows if depth_safe(r)]
 out={"phase":"BABY_A2_FULL_BETA_VALIDATION_GRID","dataset":"baby",
      "candidate_count":len(rows),"eligible_count":int(sum(eligible(r) for r in rows)),
      "beta_summary":by_beta,
      "top20_U":sorted(rows,key=lambda r:r["U"],reverse=True)[:20],
      "top20_sum_primary":sorted(rows,key=lambda r:r["sum_primary_delta"],reverse=True)[:20],
      "top20_depth_safe":sorted(ds,key=lambda r:r["U"],reverse=True)[:20],
      "VALIDATION_ONLY":True,"TEST_ACCESSED":False}
 (Path(diff)/"evidence"/"baby_a2_full_grid_summary.json").write_text(json.dumps(out,indent=2)+"\n")
 return out

def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--dataset",default="baby"); ap.add_argument("--msca-assets",required=True); ap.add_argument("--coliftrec-dir",required=True); ap.add_argument("--diffusion-dir",required=True); a=ap.parse_args()
 diff=Path(a.diffusion_dir); evid=diff/"evidence"; evid.mkdir(parents=True,exist_ok=True)
 cfg=load_dataset_config(a.dataset)
 for beta in [float(x) for x in cfg["diffusion"]["beta_candidates"]]:
  p=evid/f"beta_{btag(beta)}_training.json"; c=diff/"checkpoints"/f"{a.dataset}_beta_{btag(beta)}.pt"
  if not p.exists() or not c.exists(): raise RuntimeError(("formal training missing",beta))
  meta=json.loads(p.read_text())
  if meta["TEST_ACCESSED"] is not False or meta["VALIDATION_RANKING_USED_FOR_TRAINING"] is not False: raise RuntimeError("training discipline fail")
  if sha256(c)!=meta["checkpoint_sha256"]: raise RuntimeError("checkpoint sha mismatch")
 manifest=generate_assets(a.dataset,diff); ctx=build_context(a.dataset,a.msca_assets,a.coliftrec_dir); grid=run_grid(a.dataset,diff,ctx,manifest)
 if a.dataset=="baby":
  summary=baby_grid_summary(diff,grid)
  if summary["eligible_count"]==0:
   fail={"phase":"BABY_DIFFUSION_CONDITION_SEARCH_FAIL","reason":"FULL_BETA_GRID_ELIGIBLE_ZERO",
         "BABY_DIFFUSION_TEST":"CLOSED","TEST_USED_FOR_SELECTION":False}
   (evid/"BABY_DIFFUSION_CONDITION_SEARCH_FAIL.json").write_text(json.dumps(fail,indent=2)+"\n")
   print("BABY_DIFFUSION_CONDITION_SEARCH_FAIL",json.dumps(fail,sort_keys=True),flush=True); return
 freeze=finalize(a.dataset,diff,grid)
 if a.dataset=="baby" and not freeze["crossfit_summary"]["BABY_DIFFUSION_UPGRADE_PASS"]:
  fail={"phase":"BABY_DIFFUSION_CONDITION_SEARCH_FAIL","reason":"CROSSFIT_FAIL",
        "crossfit_summary":freeze["crossfit_summary"],"BABY_DIFFUSION_TEST":"CLOSED","TEST_USED_FOR_SELECTION":False}
  (evid/"BABY_DIFFUSION_CONDITION_SEARCH_FAIL.json").write_text(json.dumps(fail,indent=2)+"\n")
 print(f"{a.dataset.upper()}_DIFFUSION_PRETEST_FREEZE_COMPLETE",json.dumps({"final":cfg_tuple(freeze["final_config"]),"upgrade":freeze["crossfit_summary"][f"{a.dataset.upper()}_DIFFUSION_UPGRADE_PASS"],f"{a.dataset.upper()}_DIFFUSION_TEST":"CLOSED"},sort_keys=True),flush=True)
if __name__=="__main__": main()
