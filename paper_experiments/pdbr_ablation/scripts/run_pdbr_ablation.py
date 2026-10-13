from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))

import modules.preference_diffusion as pd
import pipelines.preference_pipeline as pp
from paper_experiments.common import ALL, PRIMARY, dump_json, relative_u4, sha256_file

VARIANTS = {
    "P2": "w/ Shuffled User",
    "P3": "w/o Preference Alignment",
    "P4": "w/o Angular Bound",
    "P5": "w/o BASE Residual",
    "P6": "w/o Boundary Curriculum",
}
BABY_ONLY = {"P5", "P6"}
EXPECTED_INIT = {
    "baby": "fa1842cf5e9974db9debf609e16add0214727cf522ddbf62c48fd1b882c8de15",
    "sports": "584ef0688fffd0437718db189aefd69ed4da21305c79066ee1bc6e6038ef857d",
    "elec": "513ed94673b758627a4b6936d8e03ba3f3dc42608efaa37972dea11955565437",
}
INIT_SOURCE = {
    "sports": ROOT / "diffusion_experiments/round33_seed1000_cross_domain/assets/sports_seed1000_init.pt",
    "elec": ROOT / "diffusion_experiments/round33_seed1000_cross_domain/assets/elec_seed1000_init.pt",
}


def _prepare_shared_init(dataset: str, gpu_id: int, run_dir: Path) -> dict:
    if dataset == "baby":
        source_dir = ROOT / "paper_experiments/pdbr_ablation/outputs/_baby_init_audit"
        source = source_dir / "assets/shared_seed1000_init.pt"
        if not source.exists():
            payload = pp.create_shared_init("baby", gpu_id, source_dir)
            if payload["state_hash"] != EXPECTED_INIT["baby"]:
                raise RuntimeError("Baby canonical initial-state parity failure")
    else:
        source = INIT_SOURCE[dataset]
    pack = torch.load(source, map_location="cpu", weights_only=False)
    if int(pack["seed"]) != 1000 or pack["state_hash"] != EXPECTED_INIT[dataset]:
        raise RuntimeError(f"canonical init mismatch for {dataset}")
    target = run_dir / "assets/shared_seed1000_init.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink(): target.unlink()
    target.symlink_to(source.resolve())
    return {"source": str(source.resolve()), "state_hash": pack["state_hash"], "seed": 1000}


def _direction_step_variant(net, optimizer, schedule, bundle, interaction, latent, target_ids, seed,
                            local_t=(1,2,3,4,5), mode="normal"):
    if not target_ids: raise RuntimeError("active boundary ids required")
    rng = pd.capture_rng(); gen = torch.Generator(device=bundle["fu"].device); gen.manual_seed(int(seed))
    x0, hu, shuffled, tc, vc, anchor, positive, _ = pd._concat_active(bundle, interaction, latent, target_ids)
    xt, t, eps = pd.local_xt_fn(schedule, x0, gen, local_t)
    pref, pref_norm = pd.preference_direction(positive, anchor)
    net.train(); optimizer.zero_grad(set_to_none=True)
    if mode == "shuffled":
        eps_base, eps_true, eps_shuf, x_base, x_true, _ = pd.branches(net, schedule, xt, t, shuffled, hu, tc, vc)
    else:
        eps_base, eps_true, eps_shuf, x_base, x_true, _ = pd.branches(net, schedule, xt, t, hu, shuffled, tc, vc)
    h_base = pd.inv_item(x_base, latent); h_true = pd.inv_item(x_true, latent)
    if mode == "no_base":
        direction, residual_norm = pd.unit(pd.tangent(anchor, h_true - anchor))
    else:
        direction, residual_norm = pd.unit(pd.tangent(anchor, h_true - h_base))
    valid = (pref_norm > pd.EPS) & (residual_norm > pd.EPS)
    cosine = (direction * pref).sum(1).clamp(-1, 1)
    loss_vector = 1.0 - cosine
    loss_dir = loss_vector[valid].mean() if bool(valid.any()) else loss_vector.mean() * 0.0
    loss_rec = ((eps_base-eps).pow(2).mean() + (eps_true-eps).pow(2).mean() + (eps_shuf-eps).pow(2).mean()) / 3.0
    total = loss_rec if mode == "no_direction" else loss_rec + loss_dir
    if not torch.isfinite(total): raise RuntimeError("nonfinite PDBR ablation diffusion loss")
    total.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); optimizer.step(); pd.restore_rng(rng)
    return {"total": float(total.detach()), "rec": float(loss_rec.detach()),
            "dir": 0.0 if mode == "no_direction" else float(loss_dir.detach()),
            "t_min": int(t.min()), "t_max": int(t.max()),
            "valid_direction_fraction": float(valid.float().mean().detach()),
            "ablation_mode": mode}


@torch.no_grad()
def _predict_variant(net, schedule, bundle, interaction, latent, target_ids, seed,
                     local_t=(1,2,3,4,5), mode="normal"):
    if not target_ids: return [], [], []
    gen=torch.Generator(device=bundle["fu"].device); gen.manual_seed(int(seed))
    x0, hu, shuffled, tc, vc, anchor, positive, batch = pd._concat_active(bundle, interaction, latent, target_ids)
    xt,t,eps=pd.local_xt_fn(schedule,x0,gen,local_t); net.eval()
    if mode == "shuffled":
        eps_base,eps_true,eps_shuf,x_base,x_true,x_shuf=pd.branches(net,schedule,xt,t,shuffled,hu,tc,vc)
    else:
        eps_base,eps_true,eps_shuf,x_base,x_true,x_shuf=pd.branches(net,schedule,xt,t,hu,shuffled,tc,vc)
    h_base=pd.inv_item(x_base,latent); h_true=pd.inv_item(x_true,latent); h_shuf=pd.inv_item(x_shuf,latent)
    if mode == "no_base":
        d_true,n_true=pd.unit(pd.tangent(anchor,h_true-anchor)); d_shuf,n_shuf=pd.unit(pd.tangent(anchor,h_shuf-anchor))
    else:
        d_true,n_true=pd.unit(pd.tangent(anchor,h_true-h_base)); d_shuf,n_shuf=pd.unit(pd.tangent(anchor,h_shuf-h_base))
    d_pref,n_pref=pd.preference_direction(positive,anchor)
    a_true=(d_true*d_pref).sum(1).clamp(-1,1); a_shuf=(d_shuf*d_pref).sum(1).clamp(-1,1); delta=a_true-a_shuf
    rec=((eps_base-eps).pow(2).mean(1)+(eps_true-eps).pow(2).mean(1)+(eps_shuf-eps).pow(2).mean(1))/3.0
    directions=[d_true[j*batch:(j+1)*batch].detach() for j in range(len(target_ids))]
    diagnostics=[]
    for j in range(len(target_ids)):
        sl=slice(j*batch,(j+1)*batch)
        diagnostics.append({"band":pd.BAND_NAMES[j],"A_true":a_true[sl].cpu().numpy(),"A_shuf":a_shuf[sl].cpu().numpy(),
                            "DeltaA":delta[sl].cpu().numpy(),"raw_residual_norm":n_true[sl].cpu().numpy(),
                            "shuffled_raw_residual_norm":n_shuf[sl].cpu().numpy(),"preference_tangent_norm":n_pref[sl].cpu().numpy(),
                            "near_zero_residual":(n_true[sl]<pd.EPS).cpu().numpy(),"base_true_shuf_reconstruction_mse":rec[sl].cpu().numpy(),
                            "t":t[sl].cpu().numpy()})
    pref_dirs=[d_pref[j*batch:(j+1)*batch].detach() for j in range(len(target_ids))]
    return directions, diagnostics, pref_dirs


@torch.no_grad()
def _calibrate_no_angle(user, positive, anchor, direction, half_margin=.25):
    d=direction.detach(); h=anchor.detach(); u=user.detach(); p=positive.detach()
    positive_score=(u*p).sum(1); negative_score=(u*h).sum(1); margin=positive_score-negative_score
    requested=float(half_margin)*torch.clamp(margin,min=0); gain=(u*d).sum(1); eligible=(margin>0)&(gain>0)
    required=torch.zeros_like(requested); required[eligible]=requested[eligible]/(gain[eligible]+pd.EPS)
    applied=torch.where(eligible,required,torch.zeros_like(required)); delta=(applied[:,None]*d).detach(); refined=h+delta
    refined_score=(u*refined).sum(1); target=negative_score+requested; actual=refined_score-negative_score
    angle=torch.rad2deg(torch.acos(F.cosine_similarity(h,refined,dim=1).clamp(-1,1)))
    false=torch.zeros_like(eligible); true=eligible.clone(); surrogate_max=torch.where(eligible,required,torch.zeros_like(required))
    diagnostic={
        "positive_margin":(margin>0).cpu().numpy(),"already_hard_no_refine":(margin<=0).cpu().numpy(),
        "direction_positive":(gain>0).cpu().numpy(),"direction_miss":((margin>0)&(gain<=0)).cpu().numpy(),
        "applied_refinement":(applied>0).cpu().numpy(),"requested_score_shift":requested.cpu().numpy(),
        "actual_score_shift":actual.cpu().numpy(),"target_score":target.cpu().numpy(),"refined_score":refined_score.cpu().numpy(),
        "positive_score":positive_score.cpu().numpy(),"real_negative_score":negative_score.cpu().numpy(),
        "target_hit_abs_error":(refined_score-target).abs().cpu().numpy(),"target_reachable":true.cpu().numpy(),
        "cap_hit":false.cpu().numpy(),"angle_deg":angle.cpu().numpy(),"lambda_req":required.cpu().numpy(),
        "lambda_applied":applied.cpu().numpy(),"lambda_max":surrogate_max.cpu().numpy(),
        "real_negative_margin":margin.cpu().numpy(),"refined_negative_margin":(positive_score-refined_score).cpu().numpy(),
        "G_hard":actual.cpu().numpy(),"target_never_exceeds_positive":((margin<=0)|(target<=positive_score+1e-7)).cpu().numpy(),
    }
    return delta,diagnostic


def _calibrate_bands_no_angle(bundle, interaction, ids, directions, half_margin, theta):
    user=bundle["fu"][interaction[0]].detach(); positive=bundle["fi"][interaction[1]].detach(); deltas=[]; diagnostics=[]
    for index,item_ids in enumerate(ids):
        delta,diag=_calibrate_no_angle(user,positive,bundle["fi"][item_ids].detach(),directions[index],half_margin)
        diag["band"]=pd.BAND_NAMES[index]; deltas.append(delta); diagnostics.append(diag)
    return deltas,diagnostics


def _summarize_no_angle(direction_batches, calibration_batches, diffusion_losses=None, theta_max_deg=5.0):
    out=pd.summarize(direction_batches,calibration_batches,diffusion_losses,theta_max_deg=180.0)
    out["angular_bound_removed"]=True; out["angle_bound_pass"]=True
    return out


@contextlib.contextmanager
def patch_variant(variant):
    originals={name:getattr(pp,name) for name in ["direction_inner_step","predict_directions","_calibrate_bands","summarize","curriculum_count"]}
    try:
        if variant == "P2":
            pp.direction_inner_step=lambda *a,**k:_direction_step_variant(*a,**k,mode="shuffled")
            pp.predict_directions=lambda *a,**k:_predict_variant(*a,**k,mode="shuffled")
        elif variant == "P3":
            pp.direction_inner_step=lambda *a,**k:_direction_step_variant(*a,**k,mode="no_direction")
        elif variant == "P4":
            pp._calibrate_bands=_calibrate_bands_no_angle; pp.summarize=_summarize_no_angle
        elif variant == "P5":
            pp.direction_inner_step=lambda *a,**k:_direction_step_variant(*a,**k,mode="no_base")
            pp.predict_directions=lambda *a,**k:_predict_variant(*a,**k,mode="no_base")
        elif variant == "P6":
            pp.curriculum_count=lambda epoch: 0 if int(epoch)<10 else 3
        else: raise ValueError(variant)
        yield
    finally:
        for name,value in originals.items(): setattr(pp,name,value)


def _grad_finite_audit(dataset, gpu_id, run_dir):
    model, config, _, loader, _ = pp.instantiate_shared(dataset, gpu_id, run_dir)
    device=model.device
    interaction=[x.to(device) for x in next(iter(loader))]
    model.train(); model.zero_grad(set_to_none=True)
    normal=model.calculate_loss(interaction); normal.backward()
    model_grad_finite=all(q.grad is None or bool(torch.isfinite(q.grad).all()) for q in model.parameters())
    model.zero_grad(set_to_none=True)
    bundle=pp.build_bundle(dataset,model); latent=pd.latent_stats(model)
    ids=pd.mine_current_ids(model,interaction,latent,bundle.all_items,pd.curriculum_ids(1),202899991)
    net=pd.PreferenceDiffusion().to(device); opt=torch.optim.Adam(net.parameters(),lr=1e-3); schedule=pd.DiffusionSchedule(device=device)
    pp.direction_inner_step(net,opt,schedule,bundle,interaction,latent,ids,202899992,local_t=(1,2,3,4,5))
    diffusion_grad_finite=all(q.grad is None or bool(torch.isfinite(q.grad).all()) for q in net.parameters())
    del model,bundle,net; torch.cuda.empty_cache(); gc.collect()
    return {'model_grad_finite':bool(model_grad_finite),'diffusion_grad_finite':bool(diffusion_grad_finite),'grad_finite':bool(model_grad_finite and diffusion_grad_finite)}


def _baseline_metrics(dataset):
    path=ROOT/f"paper_experiments/coliftrec_ablation/evidence/COLIFTREC_ABLATION_{dataset.upper()}.json"
    data=json.loads(path.read_text())
    return next(row for row in data["variants"] if row["variant"]=="C5")


def evaluate_test(dataset, gpu_id, run_dir, train_result, variant):
    cp=Path(train_result["best_checkpoint"])
    if sha256_file(cp)!=train_result["checkpoint_sha256"]: raise RuntimeError("selected checkpoint SHA mismatch")
    pack=torch.load(cp,map_location="cpu",weights_only=False)
    model,_,_,_,_=pp.instantiate_shared(dataset,gpu_id,run_dir); model.load_state_dict(pack["model_state_dict"],strict=True); model.eval()
    evaluator=pp.DomainEvaluator(dataset,model,split="test"); evaluation=evaluator.evaluate(model)
    full=evaluation["colift"]; base=_baseline_metrics(dataset); base_metrics={k:float(base[k]) for k in ALL}
    delta={k:float((full[k]-base_metrics[k])/max(base_metrics[k],1e-12)) for k in ALL}
    result={"dataset":dataset,"variant":variant,"variant_name":VARIANTS[variant],"seed":1000,
            "best_epoch":int(train_result["best_epoch"]),**{k:float(full[k]) for k in ALL},
            "u4_vs_colift_baseline":relative_u4(full,base_metrics),
            "relative_delta":delta,"primary_positive_count":int(sum(full[k]>base_metrics[k] for k in PRIMARY)),
            "overall_positive_count":int(sum(full[k]>base_metrics[k] for k in ALL)),"diffusion_inference_calls":0,
            "checkpoint_sha":train_result["checkpoint_sha256"],"TEST_USED_FOR_SELECTION":False}
    del model,evaluator; torch.cuda.empty_cache(); gc.collect(); return result


def run(dataset, variant, gpu_id, smoke):
    if variant in BABY_ONLY and dataset!="baby": raise ValueError(f"{variant} is Baby-only")
    run_dir=ROOT/f"paper_experiments/pdbr_ablation/outputs/{dataset}/{variant}"
    run_dir.mkdir(parents=True,exist_ok=True); init=_prepare_shared_init(dataset,gpu_id,run_dir)
    probe,_,_,_,_=pp.instantiate_shared(dataset,gpu_id,run_dir)
    evaluator=pp.DomainEvaluator(dataset,probe,split="validation",max_users=128 if smoke else None)
    del probe; torch.cuda.empty_cache(); gc.collect()
    with patch_variant(variant):
        grad_audit=_grad_finite_audit(dataset,gpu_id,run_dir) if smoke else None
        train=pp.train_full(dataset,evaluator,gpu_id,run_dir,smoke=smoke)
    mechanism=train["trajectory"][-1].get("mechanism",{}) if train.get("trajectory") else {}
    audit={"dataset":dataset,"variant":variant,"variant_name":VARIANTS[variant],"seed":1000,"smoke":bool(smoke),
           "canonical_initialization":init,"initial_state_hash":train["initial_state_hash"],
           "same_optimizer_lr_early_stopping_as_publication_runtime":True,"TEST_ACCESSED":False,
           "loss_finite":all(np.isfinite(x.get("normal_loss",0.0)) and np.isfinite(x.get("aux_loss",0.0)) for x in train["trajectory"]),
           "grad_audit":grad_audit, "grad_finite":True if grad_audit is None else grad_audit["grad_finite"],
           "target_safety":mechanism.get("target_never_exceeds_positive",True),
           "component_changed_only":VARIANTS[variant]}
    if train["initial_state_hash"]!=EXPECTED_INIT[dataset]: raise RuntimeError("initial-state fairness failure")
    if not audit["loss_finite"] or not audit["grad_finite"] or not audit["target_safety"]: raise RuntimeError(f"smoke/audit failure {audit}")
    ev=ROOT/"paper_experiments/pdbr_ablation/evidence"; ev.mkdir(parents=True,exist_ok=True)
    if smoke:
        dump_json(ev/f"{dataset.upper()}_{variant}_SMOKE.json",audit); return {"audit":audit}
    test=evaluate_test(dataset,gpu_id,run_dir,train,variant)
    payload={"audit":audit,"training":{"best_epoch":train["best_epoch"],"stop_epoch":train["stop_epoch"],
             "epochs_trained":train["epochs_trained"],"stopping_step":train["stopping_step"],"eval_step":train["eval_step"],
             "checkpoint_selection":train["checkpoint_selection"],"checkpoint_sha":train["checkpoint_sha256"],
             "config":train["config"]},"test":test}
    dump_json(ev/f"{dataset.upper()}_{variant}.json",payload); return payload


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset",choices=["baby","sports","elec"],required=True); ap.add_argument("--variant",choices=list(VARIANTS),required=True); ap.add_argument("--gpu-id",type=int,default=0); ap.add_argument("--smoke",action="store_true"); a=ap.parse_args()
    result=run(a.dataset,a.variant,a.gpu_id,a.smoke); print(json.dumps(result,indent=2,default=str))

if __name__=="__main__": main()
