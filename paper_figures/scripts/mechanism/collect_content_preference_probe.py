"""CPU-only TRAIN leave-one-out content diagnostic; never train/select a model.

Withhold the final chronological TRAIN item, build the profile from the earlier
TRAIN items, and rank the withheld item by cosine similarity over TRAIN-observed
catalog items. Unobserved items are candidates, not labeled negative preferences.
This probes the motivation; it is not a Validation/Test benchmark or CoLift ablation.
"""
from pathlib import Path
import sys, csv, json, hashlib, subprocess
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import scipy.sparse as sp
ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "paper_figures/raw_data"
OUT = RAW / "mechanism/fig13_content_preference_probe.csv"
SEED = 20261013
N_USERS = 2000

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""): h.update(b)
    return h.hexdigest()

def normalize(a):
    return a/np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-12)

def main():
    inter=ROOT/"data/baby/baby.inter"
    # Only x_label==0 rows enter any profile, target, candidate, or statistic.
    df=pd.read_csv(inter,sep="\t",usecols=["userID","itemID","timestamp","x_label"])
    df["_row"]=np.arange(len(df))
    df=df[df.x_label==0].sort_values(["userID","timestamp","_row"],kind="stable")
    catalog=np.sort(df.itemID.unique())
    histories={int(u):g.itemID.astype(int).tolist() for u,g in df.groupby("userID",sort=True)}
    eligible=[u for u,h in histories.items() if len(h)>=2 and h[-1] not in h[:-1]]
    rng=np.random.default_rng(SEED)
    users=np.sort(rng.choice(eligible,size=min(N_USERS,len(eligible)),replace=False))
    targets=np.array([histories[int(u)][-1] for u in users])
    n_items=int(max(catalog))+1
    rr=[];cc=[];vv=[]
    for i,u in enumerate(users):
        hist=histories[int(u)][:-1]
        rr.extend([i]*len(hist));cc.extend(hist);vv.extend([1/len(hist)]*len(hist))
    H=sp.csr_matrix((np.asarray(vv,np.float32),(rr,cc)),shape=(len(users),n_items))
    outputs=[]
    raw_ranks={}
    feature_shas={}
    for modality,filename in [("Text","text_feat.npy"),("Visual","image_feat.npy")]:
        path=ROOT/"data/baby"/filename
        feat=normalize(np.asarray(np.load(path),np.float32))
        if len(feat)!=n_items: raise RuntimeError("catalog/feature shape mismatch")
        profiles=normalize(np.asarray(H@feat))
        ranks=[]
        for start in range(0,len(users),128):
            end=min(start+128,len(users))
            scores=profiles[start:end]@feat.T
            keep=np.zeros(n_items,dtype=bool);keep[catalog]=True
            scores[:,~keep]=-np.inf
            for bi,u in enumerate(users[start:end]):
                scores[bi,histories[int(u)][:-1]]=-np.inf
            target=targets[start:end]
            value=scores[np.arange(end-start),target]
            # Stable item-ID tie break; target score taken from the same matmul.
            rank=1+np.sum((scores>value[:,None])|((scores==value[:,None])&(np.arange(n_items)[None,:]<target[:,None])),axis=1)
            ranks.extend(rank.tolist())
        rank=np.asarray(ranks)
        raw_ranks[modality]=rank
        feature_shas[filename]=sha(path)
        for k in [1,5,10,20,50,100,200,500]:
            outputs.append(dict(dataset="Baby",modality=modality,top_k=k,retrieval_fraction=float(np.mean(rank<=k)),
                                hit_count=int(np.sum(rank<=k)),user_count=len(users),catalog_size=len(catalog)))
        print(modality, "TRAIN LOO retrieval @100:",np.mean(rank<=100),flush=True)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    with OUT.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(outputs[0]),lineterminator="\n");w.writeheader();w.writerows(outputs)
    np.savez_compressed(OUT.with_suffix(".npz"),users=users,targets=targets,text_rank=raw_ranks["Text"],visual_rank=raw_ranks["Visual"])
    meta={"protocol":"TRAIN_ONLY_CHRONOLOGICAL_LAST_ITEM_CONTENT_PROBE","dataset":"Baby","diagnostic_seed":SEED,
          "trained_model_seed":None,"eligible_users":len(eligible),"sampled_users":len(users),"catalog_size":len(catalog),
          "profile":"mean of normalized raw features of earlier TRAIN items, then L2-normalize",
          "candidates":"TRAIN-observed catalog, excluding earlier TRAIN history",
          "target":"last chronological TRAIN item, absent from earlier TRAIN history",
          "tie_break":"ascending itemID","TEST_EVALUATED":False,"TEST_LABELS_USED":False,"VALIDATION_LABELS_USED":False,
          "model_updates":0,"parameter_selection":False,"cpu_only":True,
          "interaction_sha256":sha(inter),"feature_sha256":feature_shas,
          "interpretation":"Motivation probe, not held-out generalization or proof of negative preference."}
    OUT.with_suffix(".json").write_text(json.dumps(meta,indent=2)+"\n")
    prov=RAW/"PROVENANCE.csv"
    with prov.open(newline="") as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    rows=[r for r in rows if r["figure_id"]!="ContentProbe"]
    rows.append(dict(figure_id="ContentProbe",data_file=str(OUT.relative_to(ROOT)),dataset="Baby",seed="N/A",
        split="TRAIN leave-one-out diagnostic",method_or_variant="raw Text/Visual cosine retrieval",
        source_branch="main",source_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        source_path=str(inter.relative_to(ROOT)),source_key="x_label==0 only; chronological last TRAIN item",
        extraction_script=str(Path(__file__).relative_to(ROOT)),generated_at=datetime.now(timezone.utc).isoformat(),
        notes=f"CPU probe; RNG={SEED}; no model or evaluation seed; input SHA256s in companion JSON; no Test/Validation labels used."))
    with prov.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");w.writeheader();w.writerows(rows)
    print(json.dumps(meta,indent=2),flush=True)
if __name__=="__main__": main()
