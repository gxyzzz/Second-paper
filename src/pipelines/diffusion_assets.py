from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
def sha256(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for chunk in iter(lambda:f.read(1<<20),b""): h.update(chunk)
 return h.hexdigest()
def build_current_run_assets(msca_assets,out_dir):
 out_dir.mkdir(parents=True,exist_ok=True)
 audit=json.loads((msca_assets/"audit.json").read_text())
 assert audit.get("TEST_ACCESSED") is False
 ep=msca_assets/"embeddings.npz"; emb=np.load(ep)
 collab=np.asarray(emb["collab_item"],dtype=np.float32); final=np.asarray(emb["final_item"],dtype=np.float32)
 assert collab.shape==final.shape==(7050,64)
 inter=ROOT/"data/baby/baby.inter"
 df=pd.read_csv(inter,sep="\t",usecols=["itemID","x_label"])
 ids=np.sort(df.loc[df.x_label==0,"itemID"].unique().astype(np.int64)); assert len(ids)==7047
 np.save(out_dir/"train_item_ids.npy",ids,allow_pickle=False)
 np.savez_compressed(out_dir/"condition_endpoints.npz",collab_item=collab,final_item=final)
 z={"phase":"PHASE3_DIFFUSION_ASSET_BUILDER","dataset":"baby","source_checkpoint_sha256":audit["checkpoint_sha256"],"source_checkpoint_epoch":int(audit["checkpoint_epoch"]),"embeddings_sha256":sha256(ep),"interaction_sha256":sha256(inter),"condition_formula":"Norm[c_collab + beta*(c_final-c_collab)]","condition_endpoint_shape":[7050,64],"train_item_count":7047,"item_selection":"unique itemID appearing in x_label==0 TRAIN only","text_shape":[7050,384],"visual_shape":[7050,4096],"joint_state_dim":4480,"TEST_ACCESSED":False,"TRAIN_ONLY_ITEM_AUDIT":"PASS"}
 (out_dir/"audit.json").write_text(json.dumps(z,indent=2)+"\n"); print(json.dumps(z,sort_keys=True)); return z
if __name__=="__main__":
 ap=argparse.ArgumentParser(); ap.add_argument("--msca-assets",required=True); ap.add_argument("--out",required=True); a=ap.parse_args(); build_current_run_assets(Path(a.msca_assets),Path(a.out))
