from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np,pandas as pd,yaml
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from diffusion_experiments.modules.round6_common import load_edges,sha
from diffusion_experiments.modules.round6_boundary_weight import keyed_seed
from diffusion_experiments.modules.round6r1_risk import query_stats

def split_eval(u,seed,frac): return keyed_seed('risk_split',int(seed),int(u))%1000000>=int(float(frac)*1000000)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); a=ap.parse_args(); out=Path(a.out)
    if out.exists() and any(out.iterdir()): raise RuntimeError('refuse overwrite nonempty output')
    out.mkdir(parents=True,exist_ok=True); cfg=yaml.safe_load((ROOT/'diffusion_experiments/configs/round6r1_baby.yaml').read_text()); rc=cfg['risk']; pdx=ROOT/cfg['protocol_dir']; old=ROOT/cfg['round6_old_risk_cache']
    if sha(old)!=cfg['expected_old_risk_cache_sha256']: raise RuntimeError('old risk cache hash mismatch')
    z=np.load(old); items=z['items'].astype(np.int32); A=z['A_mask'].astype(bool); rho=z['rho'].astype(np.float32); w=z['weights'].astype(np.float32)
    fit=load_edges(pdx/'fit_edges.csv'); probe=load_edges(pdx/'probe_edges.csv').sort_values('userID'); rr=pd.read_csv(pdx/'reranker_train_users.csv').userID.to_numpy(np.int64); target=dict(zip(probe.userID.astype(int),probe.itemID.astype(int)))
    degree=np.bincount(fit.itemID.to_numpy(np.int64),minlength=7050).astype(np.int64); emb=np.load(ROOT/cfg['round1_assets']/'embeddings.npz'); norms=np.linalg.norm(emb['collab_item'].astype(np.float32),axis=1)
    cal=np.array([u for u in rr if not split_eval(u,rc['cal_eval_seed'],rc['cal_fraction'])],np.int64); evl=np.array([u for u in rr if split_eval(u,rc['cal_eval_seed'],rc['cal_fraction'])],np.int64)
    res={}
    for name,users in [('CAL',cal),('EVAL',evl)]:
        st=query_stats(users,rho,w,items,A,target,degree,norms,rc['degree_log1p_tolerance'],rc['norm_log_tolerance']); st.pop('query_pair_values'); st.pop('query_rows'); res[name]=st
    exp={'CAL':{'degree_matched_queries':561,'degree_pair_win':0.5064944216460628},'EVAL':{'degree_matched_queries':224,'degree_pair_win':0.4957162419470303}}
    for sp in exp:
        if res[sp]['degree_matched_queries']!=exp[sp]['degree_matched_queries'] or abs(res[sp]['degree_pair_win']-exp[sp]['degree_pair_win'])>1e-12: raise RuntimeError(f'corrected replay mismatch {sp}: {res[sp]}')
    result={'status':'COMPLETE_CORRECTED_REPLAY','old_cache_sha256':sha(old),'CAL':res['CAL'],'EVAL':res['EVAL'],'expected_reproduced':True,'access':{'DEV':False,'INTERNAL':False,'CONFIRM':False,'TEST':False}}
    (out/'corrected_replay.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({'status':result['status'],'CAL_degree_queries':res['CAL']['degree_matched_queries'],'CAL_degree_win':res['CAL']['degree_pair_win'],'EVAL_degree_queries':res['EVAL']['degree_matched_queries'],'EVAL_degree_win':res['EVAL']['degree_pair_win']},sort_keys=True))
if __name__=='__main__': main()
