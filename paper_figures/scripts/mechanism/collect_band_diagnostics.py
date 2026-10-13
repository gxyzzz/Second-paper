from __future__ import annotations
import csv, json, subprocess, sys, hashlib
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
from modules.preference_diffusion import BAND_NAMES, forward_bundle, latent_stats, make_diffusion, mine_current_ids, predict_directions, valid_mask
from pipelines.preference_pipeline import TrainEvents, _calibrate_bands, instantiate_shared

SEED=1000
SUMMARY_PATH='diffusion_experiments/round32_round26_parameter_sweep/evidence/ROUND32_M25_SEED1000.json'
CHECKPOINT=ROOT/'diffusion_experiments/round32_round26_parameter_sweep/outputs/seed1000/M25/best.pt'
RUN_DIR=Path('/tmp/paper_figures_band_diag_baby_seed1000')
OUT=ROOT/'paper_figures/raw_data/mechanism/fig15_16_band_diagnostics.csv'
PROV=ROOT/'paper_figures/raw_data/PROVENANCE.csv'
R32_BRANCH='exp/round32-round26-parameter-sweep-20261011'
R32_COMMIT=subprocess.check_output(['git','rev-parse',R32_BRANCH],cwd=ROOT,text=True).strip()
SOURCE_PATH=SUMMARY_PATH
SUMMARY=json.loads(subprocess.check_output(['git','show',f'{R32_COMMIT}:{SUMMARY_PATH}'],cwd=ROOT))
BEST_EPOCH=int(SUMMARY['best_epoch'])
HALF_MARGIN=float(SUMMARY['config']['half_margin'])
THETA=float(SUMMARY['config']['theta_max_deg'])
EXPECTED_SHA=SUMMARY['checkpoint_sha256']

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''): h.update(chunk)
    return h.hexdigest()

def cat(rows,key): return np.concatenate([np.asarray(r[key]) for r in rows])
def mean(rows,key): return float(np.mean(cat(rows,key)))

def main():
    if not CHECKPOINT.exists(): raise FileNotFoundError(CHECKPOINT)
    actual_sha=sha256(CHECKPOINT)
    if actual_sha!=EXPECTED_SHA: raise RuntimeError(f'checkpoint SHA mismatch: {actual_sha} != {EXPECTED_SHA}')
    ckpt=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    assert ckpt['seed']==SEED and ckpt['variant']=='M25' and int(ckpt['epoch'])==BEST_EPOCH
    model, config, train_data, _, _=instantiate_shared('baby',0,RUN_DIR)
    model.load_state_dict(ckpt['model_state_dict'],strict=True); model.eval()
    events=TrainEvents('baby',model); mask=valid_mask(train_data,model.n_items,model.device)
    net, _, schedule=make_diffusion(SEED,model.device,lr=1e-3)
    net.load_state_dict(ckpt['diffusion_state_dict'],strict=True); net.eval()
    bundle=forward_bundle(model); latent=latent_stats(model)
    direction={b:[] for b in BAND_NAMES}; calib={b:[] for b in BAND_NAMES}
    with torch.no_grad():
        for batch_idx,interaction in enumerate(train_data):
            ids,_,_=mine_current_ids(bundle,interaction,events.histories,mask,SEED,BEST_EPOCH,batch_idx,active=3)
            dirs,ddiag,_=predict_directions(net,schedule,bundle,interaction,latent,ids,202862000+SEED*100000+BEST_EPOCH*1000+batch_idx,local_t=(1,2,3,4,5))
            _,cdiag=_calibrate_bands(bundle,interaction,ids,dirs,HALF_MARGIN,THETA)
            for d in ddiag: direction[d['band']].append(d)
            for d in cdiag: calib[d['band']].append(d)
    rows=[]
    for band in BAND_NAMES:
        dr=direction[band]; cr=calib[band]; da=cat(dr,'DeltaA')
        rows.append(dict(band=band,A_true_mean=mean(dr,'A_true'),A_shuf_mean=mean(dr,'A_shuf'),DeltaA_mean=float(np.mean(da)),P_DeltaA_positive=float(np.mean(da>0)),applied_refinement_fraction=float(np.mean(cat(cr,'applied_refinement'))),actual_score_shift_mean=mean(cr,'actual_score_shift'),cap_hit_fraction=float(np.mean(cat(cr,'cap_hit'))),sample_count=int(len(da))))
    OUT.parent.mkdir(parents=True,exist_ok=True)
    with OUT.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    existing=list(csv.DictReader(PROV.open()))
    fields=list(existing[0])
    existing=[r for r in existing if r['figure_id'] not in ('Fig15','Fig16')]
    additions=[]
    checkpoint_rel=str(CHECKPOINT.relative_to(ROOT))
    for fig,key in [('Fig15','best_epoch; checkpoint_sha256; config; deterministic per-band A_true/A_shuf/DeltaA replay'),('Fig16','best_epoch; checkpoint_sha256; config; deterministic per-band score-shift replay')]:
        additions.append(dict(figure_id=fig,data_file='paper_figures/raw_data/mechanism/fig15_16_band_diagnostics.csv',dataset='Baby',seed='1000',split='TRAIN diagnostic',method_or_variant='M25 best checkpoint deterministic replay',source_branch=R32_BRANCH,source_commit=R32_COMMIT,source_path=SOURCE_PATH,source_key=key,extraction_script='paper_figures/scripts/mechanism/collect_band_diagnostics.py',generated_at=datetime.now(timezone.utc).isoformat(),notes=f'RECALCULATED from local checkpoint {checkpoint_rel}; sha256={EXPECTED_SHA}; no training; no optimizer step; no Test access; all three frozen rank bands replayed at Validation-selected epoch {BEST_EPOCH}.'))
    with PROV.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(existing+additions)
    meta={'dataset':'Baby','seed':SEED,'checkpoint':str(CHECKPOINT.relative_to(ROOT)),'checkpoint_epoch':BEST_EPOCH,'checkpoint_sha256':EXPECTED_SHA,'half_margin':HALF_MARGIN,'theta_max_deg':THETA,'rank_bands':{'Easy':'60-100','Medium':'30-60','Hard':'10-30'},'model_updates':0,'diffusion_updates':0,'TEST_ACCESSED':False,'rows':rows}
    OUT.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta,indent=2))

if __name__=='__main__': main()
