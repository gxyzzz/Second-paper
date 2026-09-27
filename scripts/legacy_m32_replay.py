from __future__ import annotations
import importlib.util, json
from pathlib import Path
import numpy as np

from modules.coliftrec import shrink_item_background
from modules.ranking import l2_normalize_rows, semantic_z_for_candidates, row_zscore, rank_by_score, metrics_at
from pipelines.dataset_config import load_dataset_config
from pipelines.msca_assets import build_train_histories_and_validation

ROOT=Path("/home/gxy/code/Second-paper")
D=Path("/home/gxy/code/DiCalRec")
M32=D/"experiments/m32a_electronics_prospective_diffusion_transfer"

def load_old():
    p=M32/"scripts/run_validation.py"
    spec=importlib.util.spec_from_file_location("legacy_m32_validation",p)
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

def blend(raw,pur,out,batch=512):
    a=np.load(raw,mmap_mode="r",allow_pickle=False)
    d=np.load(pur,mmap_mode="r",allow_pickle=False)
    assert a.shape==d.shape
    mm=np.lib.format.open_memmap(out,mode="w+",dtype=np.float32,shape=a.shape)
    for s in range(0,len(a),batch):
        x=l2_normalize_rows(a[s:s+batch]); y=l2_normalize_rows(d[s:s+batch])
        mm[s:s+batch]=l2_normalize_rows(y)
    mm.flush(); del mm

def cur_lift(path,h,pseudo,pu,pi,users,items,nitems,lam):
    zp,_=semantic_z_for_candidates(path,pseudo,pu,pi,batch_users=256)
    zv,_=semantic_z_for_candidates(path,h,users,items,batch_users=256)
    bg=shrink_item_background(pi,zp,nitems)
    return row_zscore(zv-float(lam)*bg["shrunk_mean"][items])

def main():
    OLD=load_old(); lctx=OLD.build_validation_context()
    cfg=load_dataset_config("elec"); paths=cfg["resolved_paths"]
    asset=M32/"outputs/assets"
    users=np.asarray(lctx["users"],np.int64)
    items=np.asarray(lctx["items"],np.int64)
    base=np.asarray(lctx["base_score"],np.float32)
    pi=np.asarray(lctx["pseudo_items"],np.int64)
    h,pseudo,pu,vu,es=build_train_histories_and_validation(paths["interaction"],int(users.max())+1)
    assert np.array_equal(users,vu); assert len(pi)==len(pu)
    nitems=int(np.load(paths["text_feature"],mmap_mode="r").shape[0])
    p=cfg["coliftrec"]; lt=float(p["text"]["lambda"]); lv=float(p["visual"]["lambda"]); at=float(p["text"]["alpha"]); av=float(p["visual"]["alpha"])
    purt=M32/"outputs/purified/beta_0p5/t3_g2p0_true_text.npy"
    purv=M32/"outputs/purified/beta_0p5/t3_g2p0_true_visual.npy"
    tmp=ROOT/"runs/legacy_replay"; tmp.mkdir(parents=True,exist_ok=True)
    bt=tmp/"elec_t.npy"; bv=tmp/"elec_v.npy"
    blend(paths["text_feature"],purt,bt); blend(paths["visual_feature"],purv,bv)
    try:
        crt=cur_lift(paths["text_feature"],h,pseudo,pu,pi,users,items,nitems,lt)
        crv=cur_lift(paths["visual_feature"],h,pseudo,pu,pi,users,items,nitems,lv)
        cnt=cur_lift(bt,h,pseudo,pu,pi,users,items,nitems,lt)
        cnv=cur_lift(bv,h,pseudo,pu,pi,users,items,nitems,lv)
        ort=np.asarray(lctx["raw_lt"],np.float32); orv=np.asarray(lctx["raw_lv"],np.float32)
        ont=OLD.semantic_lift(bt,lctx,"text"); onv=OLD.semantic_lift(bv,lctx,"visual")
        cs=(base+at*(cnt-crt)+av*(cnv-crv)).astype(np.float32)
        os=(base+at*(ont-ort)+av*(onv-orv)).astype(np.float32)
        cr=rank_by_score(items,cs); orank=OLD.E.rank_by_score(items,os)
        met=metrics_at(cr,users,es); omet=OLD.E.metrics_at(orank,users,lctx["eval_sets"])
        grid=json.load(open(M32/"evidence/electronics_validation_grid.json"))
        rows=[x for x in grid["rows"] if x.get("beta")==0.5 and x.get("t_edit")==3 and x.get("guidance")==2.0 and x.get("rho_T")==1.0 and x.get("rho_V")==1.0]
        assert len(rows)==1
        exp=rows[0]["metrics"]; md={k:abs(float(met[k])-float(exp[k])) for k in exp}
        ev={
          "phase":"LEGACY_M32A_SCORING_REPLAY","dataset":"elec",
          "anchor":{"beta":0.5,"t_edit":3,"guidance":2.0,"rho_T":1.0,"rho_V":1.0},
          "ranking_ids_exact":bool(np.array_equal(cr,orank)),
          "score_max_abs_diff_vs_legacy":float(np.max(np.abs(cs.astype(np.float64)-os.astype(np.float64)))),
          "raw_text_lift_max_abs_diff_vs_saved":float(np.max(np.abs(crt.astype(np.float64)-ort.astype(np.float64)))),
          "raw_visual_lift_max_abs_diff_vs_saved":float(np.max(np.abs(crv.astype(np.float64)-orv.astype(np.float64)))),
          "edited_text_lift_max_abs_diff":float(np.max(np.abs(cnt.astype(np.float64)-ont.astype(np.float64)))),
          "edited_visual_lift_max_abs_diff":float(np.max(np.abs(cnv.astype(np.float64)-onv.astype(np.float64)))),
          "metrics":met,"legacy_runtime_metrics":omet,"historical_expected_metrics":exp,
          "metric_abs_diff":md,"TEST_ACCESSED":False
        }
        ev["LEGACY_M32A_SCORING_REPLAY_ELEC"]="PASS" if ev["ranking_ids_exact"] and max(md.values())<=1e-10 else "FAIL"
        (ROOT/"docs/evidence/legacy_m32a_scoring_replay_elec.json").write_text(json.dumps(ev,indent=2)+"\n")
        print(json.dumps(ev,sort_keys=True),flush=True)
        if ev["LEGACY_M32A_SCORING_REPLAY_ELEC"]!="PASS": raise SystemExit(2)
    finally:
        bt.unlink(missing_ok=True); bv.unlink(missing_ok=True)
if __name__=="__main__": main()
