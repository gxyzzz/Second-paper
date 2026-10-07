from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from diffusion_experiments.models.round7_anchored_preference import (
    AnchoredPreferenceDDPM,
    anchor_start,
    cosine_alpha_bar,
    ddim_reverse_inference,
    inverse_standardize,
)
from diffusion_experiments.modules.round7_common import (
    ALL_METRICS,
    antithetic_noise,
    cfg_round7,
    git_sha,
    label_sets,
    load_interactions,
    metrics_at,
    relative_u,
    rerank_slots,
    sha,
    transition_counts,
    train_frame,
)


def load_model(generator: Path, device: torch.device):
    ck = torch.load(generator, map_location="cpu", weights_only=False)
    dc = ck["config"]
    model = AnchoredPreferenceDDPM(
        x_dim=64,
        cond_dim=129,
        hidden_dim=int(dc["hidden_dim"]),
        time_dim=int(dc["time_dim"]),
        dropout=float(dc["dropout"]),
        hidden_layers=int(dc["hidden_layers"]),
    ).to(device)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    return model, ck


def generate_all(model, dep, backbone_seed, diffusion_seed, cfg, cond_override=None):
    device = torch.device("cuda:0")
    users = dep["users"].astype(np.int64)
    cond_np = dep["cond"].astype(np.float32) if cond_override is None else np.asarray(cond_override, np.float32)
    anchor_np = dep["anchor_std"].astype(np.float32)
    mean = torch.as_tensor(dep["item_mean"].astype(np.float32), device=device)
    std = torch.as_tensor(dep["item_std_safe"].astype(np.float32), device=device)
    alpha = cosine_alpha_bar(int(cfg["diffusion"]["steps"]), float(cfg["diffusion"]["cosine_s"]), device=device)
    path = [int(x) for x in cfg["diffusion"]["ddim_path"]]
    t_edit = int(cfg["diffusion"]["t_edit"])
    out = np.empty((len(users), 64), np.float32)
    batch = 512
    for st in range(0, len(users), batch):
        en = min(st + batch, len(users))
        ub = users[st:en]
        cond = torch.as_tensor(cond_np[st:en], device=device)
        anchor = torch.as_tensor(anchor_np[st:en], device=device)
        noise = antithetic_noise(backbone_seed, diffusion_seed, ub, 64)
        acc = torch.zeros((len(ub), 64), device=device)
        for k in range(4):
            eps = torch.as_tensor(noise[:, k], device=device)
            x20 = anchor_start(anchor, eps, alpha, t_edit)
            g_std = ddim_reverse_inference(model, x20, cond, alpha, path)
            acc += inverse_standardize(g_std, mean, std)
        out[st:en] = (acc / 4.0).cpu().numpy().astype(np.float32)
    return out


def corr(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    if a.std() <= 0 or b.std() <= 0: return 0.0
    return float(np.corrcoef(a, b)[0,1])


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--backbone-seed',type=int,required=True)
    ap.add_argument('--diffusion-seed',type=int,required=True)
    ap.add_argument('--assets',required=True)
    ap.add_argument('--generator',required=True)
    ap.add_argument('--out',required=True)
    a=ap.parse_args()
    out=Path(a.out)
    if out.exists() and any(out.iterdir()): raise RuntimeError('refuse overwrite nonempty output')
    out.mkdir(parents=True,exist_ok=True)
    t0=time.time(); cfg=cfg_round7(); device=torch.device('cuda:0')
    assets=Path(a.assets); generator=Path(a.generator)
    audit=json.loads((assets/'audit.json').read_text()); train_result=json.loads((generator.parent/'result.json').read_text())
    if int(audit['backbone_seed'])!=a.backbone_seed or int(train_result['backbone_seed'])!=a.backbone_seed or int(train_result['diffusion_seed'])!=a.diffusion_seed: raise RuntimeError('identity mismatch')
    model,ck=load_model(generator,device)
    if sha(generator)!=train_result['generator_sha256']: raise RuntimeError('generator hash mismatch')
    dep=np.load(assets/'deployment.npz'); base=np.load(assets/'baseline_all_users.npz')
    users=base['users'].astype(np.int64); m0=base['m0_items'].astype(np.int32); m1=base['m1_items'].astype(np.int32); s0=base['m1_s0'].astype(np.float32); A=base['A_mask'].astype(bool); q=base['q'].astype(np.float32)
    if not np.array_equal(users,dep['users'].astype(np.int64)): raise RuntimeError('user order mismatch')
    g=generate_all(model,dep,a.backbone_seed,a.diffusion_seed,cfg)
    anchor_raw=dep['anchor_raw'].astype(np.float32); item_raw=dep['collab_item'].astype(np.float32)
    finite=np.isfinite(g).all(1); delta=g-anchor_raw; delta[~finite]=0.0
    d=np.einsum('bd,bld->bl',delta,item_raw[m1],optimize=True).astype(np.float32)
    r=np.clip(d/q[:,None],-float(cfg['inference']['clip_r']),float(cfg['inference']['clip_r'])).astype(np.float32); r[~finite]=0.0
    df=load_interactions(); valid_users,valid_sets=label_sets(df,1)
    if not np.array_equal(valid_users,users): raise RuntimeError('Validation user mismatch')
    m0m=metrics_at(m0,users,valid_sets); m1m=metrics_at(m1,users,valid_sets)
    if abs(m0m['R20']-audit['validation_m0_metrics']['R20'])>1e-12 or abs(m1m['R20']-audit['validation_m1_metrics']['R20'])>1e-12: raise RuntimeError('baseline dry-run mismatch')
    eta_results={}; rankings={}
    identity=rerank_slots(m1,s0,A,r,0.0)
    if not np.array_equal(identity,m1): raise RuntimeError('eta0 identity failure')
    protected=int(cfg['fusion']['protected_rank_le'])
    for eta in [float(x) for x in cfg['fusion']['eta_candidates']]:
        rank=rerank_slots(m1,s0,A,r,eta); rankings[eta]=rank
        # invariant: protected prefix and outside-A slots unchanged
        if not np.array_equal(rank[:,:protected],m1[:,:protected]): raise RuntimeError('protected prefix changed')
        outside=~A
        if not np.array_equal(rank[outside],m1[outside]): raise RuntimeError('outside-A slot changed')
        met=metrics_at(rank,users,valid_sets)
        eta_results[str(eta)]={'metrics':met,'U_vs_M1':relative_u(met,m1m),'changed_users':int(np.sum(np.any(rank!=m1,axis=1))),'changed_slots':int(np.sum(rank!=m1)),'K10':transition_counts(m1,rank,users,valid_sets,10),'K20':transition_counts(m1,rank,users,valid_sets,20)}
    # fixed shuffle diagnostic; not used for eta selection
    rng=np.random.default_rng(202610177+a.backbone_seed+a.diffusion_seed)
    perm=rng.permutation(len(users)); g_shuffle=generate_all(model,dep,a.backbone_seed,a.diffusion_seed,cfg,cond_override=dep['cond'][perm])
    delta_shuffle=g_shuffle-anchor_raw
    delta_norm=np.linalg.norm(delta,axis=1); hnorm=np.linalg.norm(anchor_raw,axis=1); gnorm=np.linalg.norm(g,axis=1)
    degree=np.bincount(train_frame(df).itemID.to_numpy(np.int64),minlength=item_raw.shape[0]).astype(np.float32)
    A_degree=np.array([degree[m1[u,A[u]]].mean() if A[u].any() else 0.0 for u in range(len(users))],np.float32)
    history_len=dep['history_len'].astype(np.int32)
    diag={'finite_users':int(finite.sum()),'fallback_users':int((~finite).sum()),'delta_norm_mean':float(delta_norm.mean()),'delta_norm_median':float(np.median(delta_norm)),'g_norm_mean':float(gnorm.mean()),'anchor_norm_mean':float(hnorm.mean()),'delta_vs_history_len_corr':corr(delta_norm,np.log1p(history_len)),'delta_vs_A_degree_corr':corr(delta_norm,A_degree),'true_vs_shuffle_g_l2_mean':float(np.linalg.norm(g-g_shuffle,axis=1).mean())}
    order=np.argsort(history_len,kind='stable'); strata=[]
    for k,ixs in enumerate(np.array_split(order,4)):
        strata.append({'bin':k+1,'lo':int(history_len[ixs].min()),'hi':int(history_len[ixs].max()),'n':int(len(ixs)),'delta_norm_mean':float(delta_norm[ixs].mean())})
    diag['history_length_strata']=strata
    np.savez_compressed(out/'cache.npz',users=users,g_raw=g,delta=delta,r=r,finite=finite)
    save={'users':users,'m0':m0,'m1':m1}
    for eta,rank in rankings.items(): save[f"m2_eta_{str(eta).replace('.','p')}"]=rank
    np.savez_compressed(out/'validation_rankings.npz',**save)
    result={'status':'COMPLETE_VALIDATION_CACHE','protocol_version':cfg['protocol_version'],'backbone_seed':a.backbone_seed,'diffusion_seed':a.diffusion_seed,'generator_sha256':sha(generator),'assets_audit_sha256':sha(assets/'audit.json'),'M0_metrics':m0m,'M1_metrics':m1m,'eta_results':eta_results,'diagnostics':diag,'cache_sha256':sha(out/'cache.npz'),'rankings_sha256':sha(out/'validation_rankings.npz'),'git_sha':git_sha(),'access':{'TRAIN':True,'VALIDATION_LABELS':True,'TEST_LABELS_USED':False},'elapsed_seconds':time.time()-t0}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'cell':f'{a.backbone_seed}x{a.diffusion_seed}','eta_U':{k:v['U_vs_M1'] for k,v in eta_results.items()},'diag':diag},sort_keys=True))
if __name__=='__main__': main()
