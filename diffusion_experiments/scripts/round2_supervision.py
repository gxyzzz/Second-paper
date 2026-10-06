from __future__ import annotations
import argparse, hashlib, json, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch, yaml

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'src'))
from diffusion_experiments.models.round2_residual import make_corrected_target


def sha(p:Path):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def pairset(df): return set(zip(df.userID.astype(int),df.itemID.astype(int)))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); a=ap.parse_args()
    out=Path(a.out); out.mkdir(parents=True,exist_ok=False)
    cfg=yaml.safe_load((ROOT/'diffusion_experiments/configs/round2_baby.yaml').read_text())
    pdx=ROOT/cfg['round1_protocol_dir']; assets=ROOT/cfg['round1_assets_dir']; bdir=ROOT/cfg['round1_backbone_dir']
    proto=json.loads((pdx/'protocol.json').read_text()); audit=json.loads((assets/'audit.json').read_text()); tr=json.loads((bdir/'training.json').read_text())
    if proto['access']['CONFIRM_ACCESSED'] or proto['access']['TEST_ACCESSED'] or audit['access']['CONFIRM_ACCESSED'] or audit['access']['TEST_ACCESSED']:
        raise RuntimeError('closed-set invariant failed')
    # actual-file hash verification, not JSON-to-JSON only
    file_hashes={}
    for key,name in [('fit_edges','fit_edges.csv'),('monitor_edges','monitor_edges.csv'),('probe_edges','probe_edges.csv'),('dev_users','dev_users.csv'),('confirm_users','confirm_users.csv'),('reranker_train_users','reranker_train_users.csv'),('internal_users','internal_users.csv')]:
        actual=sha(pdx/name); file_hashes[name]=actual
        if actual!=proto['hashes'][key]: raise RuntimeError(f'actual hash mismatch: {name}')
    if sha(pdx/'protocol.json')!=cfg['expected_protocol_sha256']: raise RuntimeError('protocol hash mismatch')
    ck=Path(tr['checkpoint'])
    if sha(ck)!=cfg['expected_backbone_sha256'] or tr['checkpoint_sha256']!=cfg['expected_backbone_sha256']:
        raise RuntimeError('backbone checkpoint hash mismatch')
    for n,h in audit['artifacts'].items():
        if sha(assets/n)!=h: raise RuntimeError(f'round1 asset actual hash mismatch: {n}')
    if audit['feature_schema_sha256']!=cfg['expected_feature_schema_sha256']: raise RuntimeError('feature schema mismatch')
    # edge isolation
    fit=pd.read_csv(pdx/'fit_edges.csv'); mon=pd.read_csv(pdx/'monitor_edges.csv'); probe_edges=pd.read_csv(pdx/'probe_edges.csv')
    if pairset(fit)&pairset(probe_edges): raise RuntimeError('probe pair leaked into FIT')
    if set(probe_edges._row.astype(int))&set(mon._row.astype(int)): raise RuntimeError('probe/monitor source row overlap')
    # user/order binding
    probe=np.load(assets/'probe_top100.npz'); tgt=np.load(assets/'probe_targets.npz'); dev=np.load(assets/'dev_top100.npz')
    if not np.array_equal(probe['users'],tgt['users']): raise RuntimeError('probe/target user order mismatch')
    expected_probe=probe_edges.sort_values('userID').userID.to_numpy(np.int64)
    if not np.array_equal(probe['users'],expected_probe): raise RuntimeError('probe asset/source user order mismatch')
    expected_dev=pd.read_csv(pdx/'dev_users.csv').userID.to_numpy(np.int64)
    if not np.array_equal(dev['users'],expected_dev): raise RuntimeError('DEV user order mismatch')
    if np.any(np.diff(probe['s0'],axis=1)>1e-6) or np.any(np.diff(dev['s0'],axis=1)>1e-6): raise RuntimeError('S0 ordering invariant')
    if any(len(set(r.tolist()))!=100 for r in probe['items']) or any(len(set(r.tolist()))!=100 for r in dev['items']): raise RuntimeError('duplicate candidate')
    # fixed Round1 feature scaling: verify both deterministic checkpoints agree exactly
    cks=[]
    for n in ['deterministic_seed202610061','deterministic_seed202610062']:
        cks.append(torch.load(ROOT/f'diffusion_experiments/runs/round1/{n}/best.pt',map_location='cpu',weights_only=False))
    mean=np.asarray(cks[0]['feature_mean'],dtype=np.float32); std=np.asarray(cks[0]['feature_std'],dtype=np.float32)
    if not np.array_equal(mean,np.asarray(cks[1]['feature_mean'],dtype=np.float32)) or not np.array_equal(std,np.asarray(cks[1]['feature_std'],dtype=np.float32)):
        raise RuntimeError('Round1 feature scaling differs across control seeds')
    ranks=tgt['target_rank'].astype(np.int64); rr=tgt['reranker_train'].astype(bool)
    sup=rr&(ranks>=6)&(ranks<=30); sup_rows=np.flatnonzero(sup); target_pos=(ranks[sup]-6).astype(np.int64)
    if len(sup_rows)!=856: raise RuntimeError(f'expected 856 supervised queries, got {len(sup_rows)}')
    win_items=probe['items'][sup_rows,5:30].astype(np.int32); s0w=probe['s0'][sup_rows,5:30].astype(np.float32); users=probe['users'][sup_rows].astype(np.int64); targets=tgt['target_items'][sup_rows].astype(np.int32)
    mon_map={int(u):set(g.itemID.astype(int).tolist()) for u,g in mon.groupby('userID')}
    m=np.zeros((len(sup_rows),25),dtype=bool)
    for i,(u,row,p) in enumerate(zip(users,win_items,targets)):
        known=mon_map.get(int(u),set())
        for j,it in enumerate(row):
            if int(it)!=int(p) and int(it) in known: m[i,j]=True
    affected=m.any(1)
    if int(affected.sum())!=94 or int(m.sum())!=94: raise RuntimeError(f'expected 94 affected queries/coords, got {affected.sum()}/{m.sum()}')
    rstar,active=make_corrected_target(s0w,target_pos,m,float(cfg['residual']['epsilon']),float(cfg['residual']['tau']))
    if np.max(np.abs(rstar[m]))>0: raise RuntimeError('M target residual not exactly zero')
    if np.max(np.abs(rstar.mean(1)))>2e-6: raise RuntimeError('target not zero mean')
    neg=active.copy(); neg[np.arange(len(neg)),target_pos]=False
    if np.any(neg&m): raise RuntimeError('monitor coordinate remains in negative pair')
    sigma=float(np.sqrt((rstar[active]**2).mean()))
    if not np.isfinite(sigma) or sigma<=0: raise RuntimeError('invalid sigma')
    x0=(rstar/sigma).astype(np.float32)
    np.savez_compressed(out/'supervision.npz',sup_rows=sup_rows.astype(np.int32),users=users,target_items=targets,target_pos=target_pos.astype(np.int16),m_mask=m,active_mask=active,rstar=rstar,x0=x0,feature_mean=mean,feature_std=std,sigma_r=np.array(sigma,dtype=np.float32),affected=affected)
    config_hash=sha(ROOT/'diffusion_experiments/configs/round2_baby.yaml')
    git_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    tracked_dirty=subprocess.run(['git','diff','--quiet'],cwd=ROOT).returncode!=0
    manifest={'status':'COMPLETE','protocol_version':cfg['protocol_version'],'mask_version':cfg['supervision_mask_version'],'git_sha':git_sha,'tracked_dirty':tracked_dirty,'config_sha256':config_hash,'protocol_sha256':sha(pdx/'protocol.json'),'backbone_sha256':sha(ck),'feature_schema_sha256':audit['feature_schema_sha256'],'actual_file_hashes':file_hashes,'round1_asset_hashes':{n:sha(assets/n) for n in ['probe_top100.npz','probe_targets.npz','dev_top100.npz']},'supervised_queries':int(len(sup_rows)),'affected_monitor_queries':int(affected.sum()),'monitor_coords':int(m.sum()),'monitor_negative_pairs':int((neg&m).sum()),'m_target_absmax':float(np.max(np.abs(rstar[m]))) if m.any() else 0.0,'target_row_mean_absmax':float(np.max(np.abs(rstar.mean(1)))),'sigma_r':sigma,'feature_mean_sha256':hashlib.sha256(mean.tobytes()).hexdigest(),'feature_std_sha256':hashlib.sha256(std.tobytes()).hexdigest(),'supervision_sha256':sha(out/'supervision.npz'),'access':{'CONFIRM_ACCESSED':False,'TEST_ACCESSED':False,'TEST_USED_FOR_SELECTION':False}}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,sort_keys=True))
if __name__=='__main__': main()
