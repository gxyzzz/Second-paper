from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'src'))
from diffusion_experiments.modules.round8_common import cfg_round8,git_sha,sha

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); ap.add_argument('--out',required=True); a=ap.parse_args(); root=Path(a.root); out=Path(a.out)
 if out.exists() and any(out.iterdir()): raise RuntimeError('refuse overwrite nonempty selection lock dir')
 out.mkdir(parents=True,exist_ok=True); cfg=cfg_round8(); cells=[]
 for b in [999,1000]:
  for d in [202610111,202610112]:
   cdir=root/f'validation_b{b}_d{d}'; udir=root/f'locked_unlabeled_b{b}_d{d}'; res=json.loads((cdir/'result.json').read_text()); unl=json.loads((udir/'result.json').read_text()); tr=json.loads((root/f'formal_b{b}_d{d}'/'result.json').read_text()); aa=json.loads((root/f'assets_seed{b}_formal'/'audit.json').read_text())
   if res['generator_sha256']!=tr['generator_sha256'] or unl['generator_sha256']!=tr['generator_sha256']: raise RuntimeError('generator/cache mismatch')
   if unl['access']['VALIDATION_LABELS_USED'] or unl['access']['TEST_LABELS_USED']: raise RuntimeError('locked rankings not label-free')
   cells.append((b,d,cdir,udir,res,unl,tr,aa))
 eta_vals=[float(x) for x in cfg['fusion']['eta_candidates']]; table={}
 for eta in eta_vals:
  vals=[float(x[4]['eta_results'][str(eta)]['U_vs_M1']) for x in cells]; table[str(eta)]={'cell_U':vals,'mean_U':float(np.mean(vals)),'positive_cells':int(sum(v>0 for v in vals))}
 selected=min(eta_vals,key=lambda e:(-table[str(e)]['mean_U'],e)); mean_u=table[str(selected)]['mean_U']; positive=table[str(selected)]['positive_cells']; expand=bool(mean_u>0 and positive>=int(cfg['expansion_rule']['require_positive_cells']))
 ranking_files={}; ranking_hashes={}; cellmeta={}; key=f"m2_eta_{str(selected).replace('.','p')}"
 for b,d,cdir,udir,res,unl,tr,aa in cells:
  z=np.load(udir/'validation_rankings.npz'); cell=f'b{b}_d{d}'
  if key not in z.files: raise RuntimeError(f'missing selected ranking {key}')
  dst=out/f'{cell}_rankings.npz'; np.savez_compressed(dst,users=z['users'].astype(np.int32),m0=z['m0'].astype(np.int32),m1=z['m1'].astype(np.int32),m2=z[key].astype(np.int32))
  ranking_files[cell]=str(dst.resolve().relative_to(ROOT)); ranking_hashes[cell]=sha(dst)
  cellmeta[cell]={'backbone_seed':b,'diffusion_seed':d,'backbone_checkpoint_sha256':aa['source_checkpoint_sha256'],'assets_audit_sha256':sha(root/f'assets_seed{b}_formal'/'audit.json'),'generator_sha256':tr['generator_sha256'],'generator_result_sha256':sha(root/f'formal_b{b}_d{d}'/'result.json'),'validation_result_sha256':sha(cdir/'result.json'),'unlabeled_recompute_result_sha256':sha(udir/'result.json'),'condition_dim':aa['condition_dim'],'K':int(cfg['inference']['K']),'ddim_path':cfg['diffusion']['ddim_path'],'t_edit':int(cfg['diffusion']['t_edit']),'A_rule':cfg['fusion'],'CoLift_config':cfg['coliftrec'],'selected_validation':res['eta_results'][str(selected)]}
 lock={'status':'ROUND8_SELECTION_LOCKED','protocol_version':cfg['protocol_version'],'selected_eta':selected,'eta_validation_table':table,'expansion_decision_before_test':{'expand_sports_elec':expand,'selected_eta_mean_validation_U':mean_u,'positive_cells':positive,'rule':'mean U > 0 and >=3/4 cells U > 0'},'cells':cellmeta,'ranking_files':ranking_files,'ranking_sha256':ranking_hashes,'guide_sha256':sha(ROOT/'diffusion_experiments/ADVISOR_EXPERIMENT_GUIDE.md'),'git_sha':git_sha(),'test_authorized':bool(cfg['access']['test_authorized']),'TEST_LABELS_USED':False,'TEST_TARGET_ITEM_IDS_ACCESSED':False}
 (out/'selection_lock.json').write_text(json.dumps(lock,indent=2)+'\n'); print(json.dumps({'status':lock['status'],'selected_eta':selected,'mean_U':mean_u,'positive_cells':positive,'expand':expand,'lock_sha256':sha(out/'selection_lock.json')},sort_keys=True))
if __name__=='__main__': main()
