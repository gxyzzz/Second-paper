from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

from modules.coliftrec import shrink_item_background
from modules.ranking import l2_normalize_rows, semantic_z_for_candidates, row_zscore, rank_by_score, metrics_at
from pipelines.dataset_config import load_dataset_config
from pipelines.msca_assets import build_train_histories_and_validation

ROOT=Path("/home/gxy/code/Second-paper")
D=Path("/home/gxy/code/DiCalRec")
M31=D/"experiments/m31a_collaborative_conditioned_multimodal_semantic_purification"
M28=D/"experiments/m28a_ranking_aware_temporal_preference_diffusion"
sys.path.insert(0,str(D/"experiments/msca_coliftrec_full_transfer/scripts"))
import run_m10a as OLD

def blend(raw,pur,rho,out,batch=512):
    a=np.load(raw,mmap_mode="r",allow_pickle=False)
    d=np.load(pur,mmap_mode="r",allow_pickle=False)
    assert a.shape==d.shape
    m=np.lib.format.open_memmap(out,mode="w+",dtype=np.float32,shape=a.shape)
    for s in range(0,len(a),batch):
        x=l2_normalize_rows(a[s:s+batch]); y=l2_normalize_rows(d[s:s+batch])
        m[s:s+batch]=l2_normalize_rows((1-float(rho))*x+float(rho)*y)
    m.flush(); del m

def cur_lift(path,h,pseudo,pu,pi,users,items,nitems,lam):
    zp,_=semantic_z_for_candidates(path,pseudo,pu,pi,batch_users=256)
    zv,_=semantic_z_for_candidates(path,h,users,items,batch_users=256)
    bg=shrink_item_background(pi,zp,nitems)
    return row_zscore(zv-float(lam)*bg["shrunk_mean"][items])

def old_lift(path,h,pseudo,pu,pi,users,items,nitems,lam):
    zp,_=OLD.semantic_z_for_candidates(path,pseudo,pu,pi,batch_users=256)
    zv,_=OLD.semantic_z_for_candidates(path,h,users,items,batch_users=256)
    bg=OLD.shrink_item_background(pi,zp,nitems)
    return OLD.row_zscore(zv-float(lam)*bg["shrunk_mean"][items])

def expected(ds,rt,rv):
    z=json.load(open(M31/f"evidence/{ds}_validation_grid.json"))
    rows=[x for x in z["configs"] if x.get("lambda_ctr")==0.1 and x.get("t_edit")==5 and x.get("guidance")==1.5 and x.get("mode")=="T_PLUS_V" and float(x.get("rho_t"))==rt and float(x.get("rho_v"))==rv]
    assert len(rows)==1
    return rows[0]["metrics"]

def run(ds,rt,rv):
    cfg=load_dataset_config(ds); paths=cfg["resolved_paths"]
    z=np.load(M28/f"outputs/assets/{ds}_validation_frozen_assets.npz",allow_pickle=True)
    users=np.asarray(z["users"],np.int64); items=np.asarray(z["ranked_items"],np.int64); base=np.asarray(z["score_full"],np.float32)
    es={int(u):set(map(int,t)) for u,t in zip(z["users"],z["eval_targets"])}
    nu=max(int(users.max()),max(es))+1
    h,pseudo,pu,vu,_=build_train_histories_and_validation(paths["interaction"],nu)
    assert np.array_equal(users,vu)
    pi=np.load(M31/f"outputs/formal/{ds}_pseudo_items.npy",allow_pickle=False).astype(np.int64)
    assert len(pi)==len(pu)
    nitems=int(np.load(paths["text_feature"],mmap_mode="r").shape[0])
    p=cfg["coliftrec"]; lt=float(p["text"]["lambda"]); lv=float(p["visual"]["lambda"]); at=float(p["text"]["alpha"]); av=float(p["visual"]["alpha"])
    rawt=paths["text_feature"]; rawv=paths["visual_feature"]
    purt=M31/f"outputs/purified/{ds}/l01/t5_g1p5_true_text.npy"
    purv=M31/f"outputs/purified/{ds}/l01/t5_g1p5_true_visual.npy"
    tmp=ROOT/"runs/legacy_replay"; tmp.mkdir(parents=True,exist_ok=True)
    bt=tmp/f"{ds}_t.npy"; bv=tmp/f"{ds}_v.npy"
    blend(rawt,purt,rt,bt); blend(rawv,purv,rv,bv)
    try:
        crt=cur_lift(rawt,h,pseudo,pu,pi,users,items,nitems,lt)
        crv=cur_lift(rawv,h,pseudo,pu,pi,users,items,nitems,lv)
        cnt=cur_lift(bt,h,pseudo,pu,pi,users,items,nitems,lt)
        cnv=cur_lift(bv,h,pseudo,pu,pi,users,items,nitems,lv)
        ort=old_lift(rawt,h,pseudo,pu,pi,users,items,nitems,lt)
        orv=old_lift(rawv,h,pseudo,pu,pi,users,items,nitems,lv)
        ont=old_lift(bt,h,pseudo,pu,pi,users,items,nitems,lt)
        onv=old_lift(bv,h,pseudo,pu,pi,users,items,nitems,lv)
        cs=(base+at*(cnt-crt)+av*(cnv-crv)).astype(np.float32)
        os=(base+at*(ont-ort)+av*(onv-orv)).astype(np.float32)
        cr=rank_by_score(items,cs); orank=OLD.rank_by_score(items,os)
        met=metrics_at(cr,users,es); exp=expected(ds,rt,rv)
        md={k:abs(float(met[k])-float(exp[k])) for k in exp}
        bref=json.load(open(M31/f"evidence/{ds}_native_validation_exactness.json"))["identity_metrics"]
        bm=metrics_at(items,users,es); bd={k:abs(float(bm[k])-float(bref[k])) for k in bref}
        ev={
            "phase":"LEGACY_M31A_SCORING_REPLAY","dataset":ds,
            "anchor":{"beta":1.0,"t_edit":5,"guidance":1.5,"rho_T":rt,"rho_V":rv},
            "ranking_ids_exact":bool(np.array_equal(cr,orank)),
            "score_max_abs_diff_vs_legacy":float(np.max(np.abs(cs.astype(np.float64)-os.astype(np.float64)))),
            "raw_text_lift_max_abs_diff":float(np.max(np.abs(crt.astype(np.float64)-ort.astype(np.float64)))),
            "raw_visual_lift_max_abs_diff":float(np.max(np.abs(crv.astype(np.float64)-orv.astype(np.float64)))),
            "edited_text_lift_max_abs_diff":float(np.max(np.abs(cnt.astype(np.float64)-ont.astype(np.float64)))),
            "edited_visual_lift_max_abs_diff":float(np.max(np.abs(cnv.astype(np.float64)-onv.astype(np.float64)))),
            "base_metrics":bm,"base_metric_abs_diff":bd,"metrics":met,
            "historical_expected_metrics":exp,"metric_abs_diff":md,"TEST_ACCESSED":False
        }
        key="LEGACY_M31A_SCORING_REPLAY_"+ds.upper()
        ev[key]="PASS" if ev["ranking_ids_exact"] and max(md.values())<=1e-10 and max(bd.values())<=1e-10 else "FAIL"
        out=ROOT/f"docs/evidence/legacy_m31a_scoring_replay_{ds}.json"; out.write_text(json.dumps(ev,indent=2)+"\n")
        print(json.dumps(ev,sort_keys=True),flush=True)
        if ev[key]!="PASS": raise SystemExit(2)
    finally:
        bt.unlink(missing_ok=True); bv.unlink(missing_ok=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset",choices=["baby","sports"],required=True); a=ap.parse_args()
    run(a.dataset,0.25 if a.dataset=="baby" else 0.75,1.0 if a.dataset=="baby" else 0.75)
