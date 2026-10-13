"""Frozen Baby checkpoint diagnostics: TRAIN samples and full Validation only.

CPU fallback remaps the two legacy Tensor.cuda() calls during MSCA construction
and loads graphs on CPU. It does not modify src/, checkpoint weights or equations.
CPU RNG streams differ from CUDA: these are new sampled diagnostics, not replay
parity claims for the original training run. Formal Test is never evaluated.
"""
from pathlib import Path
import argparse, csv, hashlib, json, sys, subprocess
from datetime import datetime, timezone
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"src"))
from models.msca import MSCA
from utils.configurator import Config
from utils.dataset import RecDataset
from utils.dataloader import TrainDataLoader
from utils.utils import init_seed
from modules.preference_diffusion import (
    PreferenceDiffusion, LinearDDPM, forward_bundle, latent_stats,
    mine_current_ids, valid_mask, predict_directions, calibrate, state_hash, BAND_NAMES)
from modules.ranking import topk_from_embeddings, rank_by_score, metrics_at
from modules.attribute import attribute_z
from modules.coliftrec import fit_backgrounds, score_coliftrec
from pipelines.preference_pipeline import DomainEvaluator
RAW=ROOT/"paper_figures/raw_data"
CKPT=ROOT/"diffusion_experiments/round32_round26_parameter_sweep/outputs/seed1000/M25/best.pt"
BASE=ROOT/"diffusion_experiments/round26_preference_directed_diffusion/outputs/seed1000/B0/best.pt"
EXPECTED="b72786ce4873d659868da950f5d618330cd4fc1486a638eb2cfd8267c18c7820"
SAMPLE_SEED=20261013

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""):h.update(b)
    return h.hexdigest()

def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n")
        w.writeheader();w.writerows(rows)

def make_cpu_model():
    cfg=Config("MSCA","baby",{"use_gpu":False,"seed":[1000],"data_path":str(ROOT/"data")+"/"})
    for key in cfg["hyper_parameters"]:
        value=cfg[key]
        if isinstance(value,(list,tuple)) and len(value)==1:cfg[key]=value[0]
    cfg["seed"]=1000;cfg["device"]=torch.device("cpu");cfg["use_gpu"]=False
    dataset=RecDataset(cfg);train=dataset.split()[0];train.inter_num=len(train.df)
    loader=TrainDataLoader(cfg,train,batch_size=256,shuffle=False)
    for name in ["image","text"]:
        if not (ROOT/f"data/baby/{name}_adj_10_True.pt").exists():
            raise RuntimeError("Cached graph required; diagnostic must not generate dataset assets")
    original_load=torch.load
    def cpu_load(*args,**kwargs):
        kwargs["map_location"]="cpu";kwargs.setdefault("weights_only",False)
        return original_load(*args,**kwargs)
    init_seed(1000)
    with patch.object(torch.Tensor,"cuda",lambda self,*a,**k:self.to("cpu")),patch("torch.load",cpu_load):
        model=MSCA(cfg,loader)
    model.eval()
    return model,loader

@torch.no_grad()
def eval_scores(model,ev):
    u,i=model.forward(test=True)
    items,scores=topk_from_embeddings(u,i,ev.users,ev.histories,100,256)
    pseudo_items,_=topk_from_embeddings(u,i,ev.pseudo_users,ev.pseudo,100,256)
    ztp=ev._semantic(ev.text,ev.text_pseudo,pseudo_items,64)
    zte=ev._semantic(ev.text,ev.text_full,items,64)
    zvp=ev._semantic(ev.visual,ev.visual_pseudo,pseudo_items,32)
    zve=ev._semantic(ev.visual,ev.visual_full,items,32)
    zap,_=attribute_z(ev.mats,ev.pseudo_profiles,ev.pseudo_users,pseudo_items,128,ev.attribute_weights)
    zae,_=attribute_z(ev.mats,ev.full_profiles,ev.users,items,128,ev.attribute_weights)
    backgrounds=fit_backgrounds(pseudo_items,ztp,zap,zvp,model.n_items)
    full,_=score_coliftrec(scores,items,zte,zae,zve,backgrounds,ev.params,ev.enabled)
    return items,scores,full,(zte,zae,zve,backgrounds)

def band_of_rank(r):
    if r<=10:return "1–10"
    if r<=20:return "11–20"
    if r<=30:return "21–30"
    if r<=60:return "31–60"
    if r<=100:return "61–100"
    return ">100"

@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--events",type=int,default=4096)
    args=parser.parse_args()
    torch.set_num_threads(4)
    if sha(CKPT)!=EXPECTED:raise RuntimeError("Frozen M25 checkpoint hash mismatch")
    payload=torch.load(CKPT,map_location="cpu",weights_only=False)
    base=torch.load(BASE,map_location="cpu",weights_only=False)
    if payload["seed"]!=1000 or payload["variant"]!="M25" or payload["epoch"]!=23:
        raise RuntimeError("M25 identity mismatch")
    source_path="diffusion_experiments/round26_preference_directed_diffusion/evidence/ROUND26_B0_SEED1000.json"
    base_source=json.loads(subprocess.check_output(["git","show","2826d8c5584c0664b8b65c201e746e5c81b94f6c:"+source_path],cwd=ROOT))
    if sha(BASE)!=base_source["checkpoint_sha256"]:raise RuntimeError("B0 checkpoint hash mismatch")
    model,loader=make_cpu_model()
    model.load_state_dict(payload["model_state_dict"],strict=True)
    frozen_hash=state_hash(model.state_dict())
    net=PreferenceDiffusion();net.load_state_dict(payload["diffusion_state_dict"],strict=True);net.eval()
    frozen_net_hash=state_hash(net.state_dict())
    bundle=forward_bundle(model);latent=latent_stats(model)
    hist=[[] for _ in range(model.n_users)]
    for user,group in loader.dataset.df.groupby("userID",sort=False):
        hist[int(user)]=group.itemID.astype(int).tolist()
    mask=valid_mask(loader,model.n_items,"cpu")
    rng=np.random.default_rng(SAMPLE_SEED)
    # Preserve random order: sorting source rows would cluster repeated users
    # and contaminate the roll(1) SHUFFLED-user control with identical users.
    sampled=rng.choice(len(loader.dataset.df),size=min(args.events,len(loader.dataset.df)),replace=False)
    events=loader.dataset.df.iloc[sampled]
    schedule=LinearDDPM("cpu");rows=[]
    for batch_idx,start in enumerate(range(0,len(events),256)):
        batch=events.iloc[start:start+256]
        interaction=torch.tensor(np.stack([batch.userID.to_numpy(),batch.itemID.to_numpy(),np.zeros(len(batch),dtype=int)]),dtype=torch.long)
        ids,offsets,_=mine_current_ids(bundle,interaction,hist,mask,1000,23,batch_idx,active=3)
        directions,dd,_=predict_directions(net,schedule,bundle,interaction,latent,ids,
                                          SAMPLE_SEED+10000+batch_idx,local_t=(1,2,3,4,5))
        for j,band in enumerate(BAND_NAMES):
            _,cd=calibrate(bundle["fu"][interaction[0]],bundle["fi"][interaction[1]],bundle["fi"][ids[j]],directions[j],.25,5.)
            for b in range(len(batch)):
                row={"event_index":int(sampled[start+b]),"userID":int(interaction[0,b]),
                     "shuffled_userID":int(interaction[0].roll(1,0)[b]),"positive_itemID":int(interaction[1,b]),
                     "band":band,"candidate_itemID":int(ids[j][b]),"rank":int(offsets[j][b])+1}
                for key in ["A_true","A_shuf","DeltaA","t"]:row[key]=float(dd[j][key][b])
                for key in ["real_negative_margin","refined_negative_margin","requested_score_shift","actual_score_shift",
                            "angle_deg","applied_refinement","cap_hit","direction_miss"]:
                    row[key]=float(cd[key][b])
                rows.append(row)
        print(f"TRAIN diagnostic {start+len(batch)}/{len(events)}",flush=True)
    if frozen_hash!=state_hash(model.state_dict()) or frozen_net_hash!=state_hash(net.state_dict()):
        raise RuntimeError("Weights mutated during diagnostic")
    d=pd.DataFrame(rows)
    if d.isna().any().any() or not np.isfinite(d.select_dtypes("number")).all().all():raise RuntimeError("nonfinite samples")
    if (d.angle_deg>5.01).any():raise RuntimeError("Angular bound violation")
    pos=d.real_negative_margin>0
    if (d.loc[pos,"refined_negative_margin"]<.75*d.loc[pos,"real_negative_margin"]-1e-5).any():
        raise RuntimeError("Margin bound violation")
    sample_file=RAW/"mechanism/fig15_16_sample_diagnostics.csv"
    write_csv(sample_file,rows)
    print("Building full Validation evaluator (no Test evaluation)",flush=True)
    ev=DomainEvaluator("baby",model,split="validation")
    items,msca,full,parts=eval_scores(model,ev)
    full_rank=rank_by_score(items,full)
    full_metrics=metrics_at(full_rank,ev.users,ev.sets)
    parity={"stored_M25_R20":float(payload["full_colift_R20"]),"cpu_M25_R20":float(full_metrics["R20"])}
    parity["absolute_difference"]=abs(parity["stored_M25_R20"]-parity["cpu_M25_R20"])
    if parity["absolute_difference"]>5e-5:raise RuntimeError(f"CPU Validation parity failed: {parity}")
    model.load_state_dict(base["model_state_dict"],strict=True)
    base_hash=state_hash(model.state_dict())
    bitems,bmsca,bfull,bparts=eval_scores(model,ev)
    base_rank=rank_by_score(bitems,bfull)
    base_metrics=metrics_at(base_rank,ev.users,ev.sets)
    stored_b=float(base_source["best_evaluation"]["colift"]["R20"])
    if abs(base_metrics["R20"]-stored_b)>5e-5:raise RuntimeError("CPU B0 Validation parity failed")
    parity.update(stored_B0_R20=stored_b,cpu_B0_R20=float(base_metrics["R20"]),
                  B0_absolute_difference=abs(base_metrics["R20"]-stored_b))
    transitions=[]
    for idx,user in enumerate(ev.users):
        bm={int(item):r+1 for r,item in enumerate(base_rank[idx])}
        fm={int(item):r+1 for r,item in enumerate(full_rank[idx])}
        for item in sorted(ev.sets[int(user)]):
            br=bm.get(item,101);fr=fm.get(item,101)
            transitions.append({"userID":int(user),"itemID":int(item),"baseline_rank_top100":br,"full_rank_top100":fr,
                                "baseline_band":band_of_rank(br),"full_band":band_of_rank(fr)})
    transition_file=RAW/"performance/fig17_validation_rank_transitions.csv"
    write_csv(transition_file,transitions)
    za,zb,zc,backgrounds=bparts;ablation=[]
    for label,enabled in [("MSCA",None),("T",dict(text=True,attribute=False,visual=False)),
                         ("A",dict(text=False,attribute=True,visual=False)),("V",dict(text=False,attribute=False,visual=True)),
                         ("TA",dict(text=True,attribute=True,visual=False)),("TV",dict(text=True,attribute=False,visual=True)),
                         ("AV",dict(text=False,attribute=True,visual=True)),("TAV",dict(text=True,attribute=True,visual=True))]:
        if enabled is None:ranked=bitems
        else:
            score,_=score_coliftrec(bmsca,bitems,za,zb,zc,backgrounds,ev.params,enabled)
            ranked=rank_by_score(bitems,score)
        ablation.append({"variant":label,**metrics_at(ranked,ev.users,ev.sets)})
    ablation_file=RAW/"performance/fig18_colift_modality_ablation.csv"
    write_csv(ablation_file,ablation)
    if base_hash!=state_hash(model.state_dict()):raise RuntimeError("B0 weights changed")
    meta={"protocol":"FROZEN_BABY_CPU_DIAGNOSTICS","seed":1000,"sampling_seed":SAMPLE_SEED,
          "event_sample_count":len(events),"unique_sample_users":int(events.userID.nunique()),"rows":len(rows),
          "batch_order":"uniform sampled events in random order, never user-sorted",
          "shuffled_same_user_fraction":float((d.userID==d.shuffled_userID).mean()),
          "checkpoint_sha256":EXPECTED,"baseline_checkpoint_sha256":sha(BASE),"checkpoint_epoch":23,
          "active_training_bands_at_checkpoint":["Easy","Medium"],"replayed_bands":list(BAND_NAMES),
          "CPU_ADAPTER":"Tensor.cuda() remapped to cpu only during constructor; graph loads map_location=cpu",
          "CPU_RNG_CUDA_PARITY_CLAIM":False,"VALIDATION":True,"TEST_EVALUATED":False,"MODEL_UPDATES":0,
          "VALIDATION_USERS":len(ev.users),"validation_parity":parity,"validation_metrics":{"B0":base_metrics,"M25":full_metrics},
          "rank_censoring":"101 denotes >100, not an exact catalog rank","colift_ablation":"same frozen B0 checkpoint/candidates/backgrounds/weights, enabled modalities only",
          "torch_version":torch.__version__,"collector_sha256":sha(Path(__file__)),
          "input_sha256":{str(path.relative_to(ROOT)):sha(path) for path in [
              ROOT/"data/baby/baby.inter",ROOT/"data/baby/text_feat.npy",ROOT/"data/baby/image_feat.npy",
              ROOT/"data/baby/metadata_text_cache.jsonl",ROOT/"data/baby/text_adj_10_True.pt",
              ROOT/"data/baby/image_adj_10_True.pt"]}}
    (RAW/"mechanism/ADVANCED_DIAGNOSTICS_PROTOCOL.json").write_text(json.dumps(meta,indent=2)+"\n")
    prov=RAW/"PROVENANCE.csv"
    with prov.open(newline="") as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;pr=list(reader)
    ids=["SampleDiagnostics","ValidationRankTransitions","CoLiftModality"]
    pr=[r for r in pr if r["figure_id"] not in ids]
    for id_,file,split,key in zip(ids,[sample_file,transition_file,ablation_file],
                                ["TRAIN diagnostic","Validation diagnostic","Validation diagnostic"],
                                ["frozen M25; 4096 sampled TRAIN events; all bands","frozen B0/M25 Top100 related-item ranks","same B0; T/A/V switches"]):
        is_base=id_=="CoLiftModality"
        commit="2826d8c5584c0664b8b65c201e746e5c81b94f6c" if is_base else "8520b5a9db80a6af8610eebb6a098c2ba455cde4"
        sp=source_path if is_base else "diffusion_experiments/round32_round26_parameter_sweep/evidence/ROUND32_M25_SEED1000.json"
        branch="exp/round26-preference-directed-diffusion-20261010" if is_base else "exp/round32-round26-parameter-sweep-20261011"
        pr.append(dict(figure_id=id_,data_file=str(file.relative_to(ROOT)),dataset="Baby",seed="1000",split=split,
             method_or_variant="frozen seed1000 M25/B0 CPU diagnostic",source_branch=branch,
             source_commit=commit,source_path=sp,source_key=key,
             extraction_script=str(Path(__file__).relative_to(ROOT)),generated_at=datetime.now(timezone.utc).isoformat(),
             notes="New CPU inference diagnostics; no weight updates; no Test evaluation; SHA/protocol/Validation parity in ADVANCED_DIAGNOSTICS_PROTOCOL.json."))
    with prov.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");w.writeheader();w.writerows(pr)
    print(json.dumps(meta,indent=2),flush=True)
if __name__=="__main__":main()
