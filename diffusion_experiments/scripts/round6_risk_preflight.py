from __future__ import annotations
import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path
import numpy as np,pandas as pd,torch
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from diffusion_experiments.models.round6_user_behavior_diffusion import UserBehaviorDiffusion
from diffusion_experiments.modules.round6_common import cfg_round6,sha,load_edges
from diffusion_experiments.modules.round6_boundary_weight import sample_users,risk_for_users,keyed_seed

def split_eval(u,seed,cal_fraction):
    x=keyed_seed('risk_split',int(seed),int(u))%1000000; return x>=int(float(cal_fraction)*1000000)
def load_model(p,device='cuda:0'):
    z=torch.load(p,map_location='cpu',weights_only=False); dc=z['config']; m=UserBehaviorDiffusion(int(z['x_dim']),int(z['cond_dim']),int(dc['hidden_dim']),int(dc['time_dim']),float(dc['dropout']),int(dc['hidden_layers'])).to(device); m.load_state_dict(z['model']); m.eval(); return m,z

def query_stats(users,rho,w,items,A,target,degree,tol):
    pair=[]; degree_pair=[]; posw=[]; negw=[]; pos_top=[]; all_top=0; all_items=0; matched_q=0; no_deg=0; rows=[]
    for u in users:
        u=int(u); q=np.flatnonzero(items[u]==int(target[u]))
        if not len(q): continue
        j=int(q[0]); idx=np.flatnonzero(A[u] & np.isfinite(rho[u]));
        if j not in set(idx.tolist()) or len(idx)<2: continue
        neg=idx[idx!=j]; pr=float(rho[u,j]); pw=float(w[u,j]); nr=rho[u,neg]; nw=w[u,neg]; wins=float(np.mean((pr>nr).astype(float)+.5*(pr==nr))); pair.append(wins); posw.append(pw); negw.extend(nw.tolist()); matched_q+=1
        top=(rho[u,idx]>=.75); all_top+=int(top.sum()); all_items+=len(idx); pos_top.append(int(pr>=.75))
        dmatch=np.abs(np.log1p(degree[neg])-np.log1p(degree[int(target[u])]))<=float(tol)
        if dmatch.any():
            x=nr[dmatch]; degree_pair.append(float(np.mean((pr>x).astype(float)+.5*(pr==x))))
        else: no_deg+=1
        rows.append((u,wins,pw,int(pr>=.75),len(idx)))
    all_rate=matched_q/max(all_items,1); top_rate=sum(pos_top)/max(all_top,1); enrich=top_rate/max(all_rate,1e-12)
    return {'queries':matched_q,'pair_win':float(np.mean(pair)) if pair else None,'degree_matched_queries':len(degree_pair),'degree_pair_win':float(np.mean(degree_pair)) if degree_pair else None,'degree_no_match_queries':no_deg,'known_positive_mean_w':float(np.mean(posw)) if posw else None,'unobserved_mean_w':float(np.mean(negw)) if negw else None,'all_A_target_rate':float(all_rate),'top_risk_target_rate':float(top_rate),'top_risk_positive_coverage':float(np.mean(pos_top)) if pos_top else None,'top_risk_enrichment':float(enrich),'query_pair_values':pair,'query_rows':rows}

def bootstrap_pair(vals,resamples,seed):
    x=np.asarray(vals,float); rng=np.random.default_rng(seed); out=[]
    for _ in range(int(resamples)):
        ix=rng.integers(0,len(x),len(x)); out.append(float(x[ix].mean()))
    return {'mean':float(x.mean()),'ci95':[float(np.quantile(out,.025)),float(np.quantile(out,.975))],'resamples':int(resamples),'seed':int(seed)}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--assets',required=True); ap.add_argument('--generator',required=True); ap.add_argument('--out',required=True); ap.add_argument('--seed',type=int,required=True); a=ap.parse_args(); out=Path(a.out)
    if out.exists() and any(out.iterdir()): raise RuntimeError('refuse overwrite nonempty output')
    out.mkdir(parents=True,exist_ok=True); t0=time.time(); cfg=cfg_round6(); rc=cfg['risk']; pdx=ROOT/cfg['protocol_dir']; fit=load_edges(pdx/'fit_edges.csv'); probe=load_edges(pdx/'probe_edges.csv').sort_values('userID'); rr=pd.read_csv(pdx/'reranker_train_users.csv').userID.to_numpy(np.int64)
    if a.seed not in [int(x) for x in cfg['generator_seeds']]: raise RuntimeError('unregistered generator seed')
    assets=Path(a.assets); uc=np.load(assets/'user_conditions.npz')['condition'].astype(np.float32); ref=np.load(assets/'reference_L100.npz'); items=ref['items'].astype(np.int32); A=ref['A_mask'].astype(bool); beh=np.load(assets/'teacher_behavior.npz'); cf=beh['standardized'].astype(np.float32); degree=np.bincount(fit.itemID.to_numpy(np.int64),minlength=cf.shape[0]).astype(np.int64)
    model,ck=load_model(Path(a.generator)/'generator.pt'); users=np.arange(len(uc),dtype=np.int64); samples=sample_users(model,uc,users,a.seed,int(rc['K']),ck['config']); risk=risk_for_users(samples,users,items,A,cf,float(rc['temperature']))
    np.savez_compressed(out/'risk_cache.npz',items=items,A_mask=A,weights=risk['weights'],rho=risk['rho'],compatibility=risk['compatibility'],valid_query=risk['valid_query'],users=users.astype(np.int32))
    target=dict(zip(probe.userID.astype(int),probe.itemID.astype(int))); cal=np.array([u for u in rr if not split_eval(u,rc['cal_eval_seed'],rc['cal_fraction'])],np.int64); evl=np.array([u for u in rr if split_eval(u,rc['cal_eval_seed'],rc['cal_fraction'])],np.int64)
    calst=query_stats(cal,risk['rho'],risk['weights'],items,A,target,degree,rc['degree_log1p_tolerance']); evst=query_stats(evl,risk['rho'],risk['weights'],items,A,target,degree,rc['degree_log1p_tolerance'])
    # same generator/noise, shuffle only user-side conditions on EVAL query users
    eligible=np.array([r[0] for r in evst['query_rows']],np.int64); rng=np.random.default_rng(int(rc['cal_eval_seed'])+991); perm=eligible[rng.permutation(len(eligible))] if len(eligible) else eligible; shuf=uc.copy();
    if len(eligible): shuf[eligible]=uc[perm]
    ss=sample_users(model,shuf,eligible,a.seed,int(rc['K']),ck['config']) if len(eligible) else np.empty((0,int(rc['K']),64),np.float32); sr=risk_for_users(ss,eligible,items,A,cf,float(rc['temperature'])) if len(eligible) else {'rho':np.empty((0,100),np.float32),'weights':np.empty((0,100),np.float32)}
    # map shuffled rows back to full user axis for identical query metric implementation
    frho=np.full_like(risk['rho'],np.nan); fw=np.full_like(risk['weights'],np.nan)
    for q,u in enumerate(eligible): frho[int(u)]=sr['rho'][q]; fw[int(u)]=sr['weights'][q]
    shst=query_stats(eligible,frho,fw,items,A,target,degree,rc['degree_log1p_tolerance']) if len(eligible) else {'pair_win':None,'queries':0}
    for st,name,bs in [(calst,'CAL',int(rc['bootstrap_seed'])),(evst,'EVAL',int(rc['bootstrap_seed'])+1)]:
        st['pair_win_bootstrap']=bootstrap_pair(st.pop('query_pair_values'),int(rc['bootstrap_resamples']),bs) if st['queries'] else None; st.pop('query_rows',None)
    shst.pop('query_pair_values',None); shst.pop('query_rows',None)
    gate={'min_eval_queries':evst['queries']>=int(rc['min_eval_queries']),'pair_win':evst['pair_win'] is not None and evst['pair_win']>=float(rc['min_pair_win']),'enrichment':evst['top_risk_enrichment']>=float(rc['min_enrichment']),'weight_direction':evst['known_positive_mean_w']<evst['unobserved_mean_w'],'degree_pair_win':evst['degree_pair_win'] is not None and evst['degree_pair_win']>float(rc['min_degree_pair_win']),'not_weaker_than_shuffled':shst.get('pair_win') is not None and evst['pair_win']>=shst['pair_win']}
    passed=all(gate.values()); result={'status':'RISK_SIGNAL_ESTABLISHED' if passed else 'RISK_SIGNAL_NOT_ESTABLISHED','generator_seed':a.seed,'CAL':calst,'EVAL':evst,'SHUFFLED_EVAL':shst,'gate':gate,'gate_pass':passed,'eligible_eval_users_for_shuffle':int(len(eligible)),'risk_cache_sha256':sha(out/'risk_cache.npz'),'generator_sha256':sha(Path(a.generator)/'generator.pt'),'assets_reference_sha256':sha(assets/'reference_L100.npz'),'elapsed_seconds':time.time()-t0,'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'access':{'DEV_ACCESSED':False,'INTERNAL_ACCESSED':False,'CONFIRM_ACCESSED':False,'TEST_ACCESSED':False}}
    (out/'preflight.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({'status':result['status'],'seed':a.seed,'eval_queries':evst['queries'],'pair_win':evst['pair_win'],'enrichment':evst['top_risk_enrichment'],'mean_w_pos':evst['known_positive_mean_w'],'mean_w_unobs':evst['unobserved_mean_w'],'degree_pair_win':evst['degree_pair_win'],'shuffled_pair_win':shst.get('pair_win'),'gate':gate},sort_keys=True))
if __name__=='__main__': main()
