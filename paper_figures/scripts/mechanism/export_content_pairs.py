"""Export existing paired TRAIN ranks to a versionable CSV, without new inference."""
from pathlib import Path
import csv,json,hashlib,subprocess
from datetime import datetime,timezone
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]
RAW=ROOT/"paper_figures/raw_data"
def main():
    source=RAW/"mechanism/fig13_content_preference_probe.npz"
    with np.load(source,allow_pickle=False) as a:
        d=pd.DataFrame({"userID":a["users"],"heldout_itemID":a["targets"],"text_rank":a["text_rank"],"visual_rank":a["visual_rank"]})
    expected=pd.read_csv(RAW/"mechanism/fig13_content_preference_probe.csv")
    for name,col in [("Text","text_rank"),("Visual","visual_rank")]:
        for _,r in expected[expected.modality==name].iterrows():
            assert (d[col]<=int(r.top_k)).sum()==int(r.hit_count)
    target=RAW/"mechanism/fig14_content_paired_ranks.csv"
    d.to_csv(target,index=False)
    meta={"protocol":"EXISTING_CONTENT_PROBE_PAIRED_RANK_EXPORT","source_npz_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
          "users":len(d),"model_seed":None,"diagnostic_seed":20261013,"TEST_LABELS_USED":False,
          "same_protocol_as":"fig13_content_preference_probe.json"}
    target.with_suffix(".json").write_text(json.dumps(meta,indent=2)+"\n")
    prov=RAW/"PROVENANCE.csv"
    with prov.open(newline="") as f:r=csv.DictReader(f);fields=r.fieldnames;rows=list(r)
    rows=[r for r in rows if r["figure_id"]!="ContentPairs"]
    rows.append(dict(figure_id="ContentPairs",data_file=str(target.relative_to(ROOT)),dataset="Baby",seed="N/A",
        split="TRAIN leave-one-out diagnostic",method_or_variant="paired Text/Visual raw-content ranks",source_branch="main",
        source_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        source_path=str(source.relative_to(ROOT)),source_key="users,targets,text_rank,visual_rank",
        extraction_script=str(Path(__file__).relative_to(ROOT)),generated_at=datetime.now(timezone.utc).isoformat(),
        notes="Export of existing probe; all cutoff counts checked; no new inference or Test labels."))
    with prov.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");w.writeheader();w.writerows(rows)
    print("Paired content ranks exported:",len(d))
if __name__=="__main__":main()
