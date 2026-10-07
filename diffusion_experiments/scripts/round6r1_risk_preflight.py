from __future__ import annotations
import argparse,json,subprocess,sys,time,yaml
from pathlib import Path
import numpy as np,pandas as pd,torch
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'src'))
from diffusion_experiments.models.round6_user_behavior_diffusion import UserBehaviorDiffusion,cosine_alpha_bar
from diffusion_experiments.modules.round6_common import load_edges,sha
from diffusion_experiments.modules.round6_boundary_weight import sample_users,keyed_seed
from diffusion_experiments.modules.round6r1_risk import raw_dot_risk_for_users,query_stats,per_user_spearman,bootstrap_mean,bootstrap_paired_delta,inverse_standardized

def cfg(): return yaml.safe_load((ROOT/'diffusion_experiments/configs/round6r1_baby.yaml').read_text())
def split_eval(u,seed,frac): return keyed_seed('risk_split',int(seed),int(u))%1000000>=int(float(frac)*1000000)
def load_model(p,device='cuda:0'):
    z=torch.load(p,map_location='cpu',weights_only=False); dc=z['config']; m=UserBehaviorDiffusion(int(z['x_dim']),int(z['cond_dim']),int(dc['hidden_dim']),int(dc['time_dim']),float(dc['dropout']),int(dc['hidden_layers'])).to(device); m.load_state_dict(z['model']); m.eval(); return m,z

def denoise_diag(model,events,seed,dc,device='cuda:0'):
    x0=events['x0'].astype(np.float32); cond=events['condition'].astype(np.float32); rng=np.random.default_rng(int(seed)+7001); ix=rng.choice(len(x0),size=min(2048,len(x0)),replace=False); xb=torch.as_tensor(x0[ix],device=device); cb=torch.as_tensor(cond[ix],device=device); alpha=cosine_alpha_bar(int(dc['steps']),float(dc['cosine_s']),device=device); out={}
    model.eval()
    for t0 in [1,10,25,40,50]:
        erng=np.random.default_rng(int(seed)+int(t0)*9176); eps=torch.as_tensor(erng.standard_normal(xb.shape).astype(np.float32),device=device); t=torch.full((len(ix),),t0,device=device,dtype=torch.long); at=alpha[t].unsqueeze(1); xt=at.sqrt()*xb+(1-at).sqrt()*eps
        with torch.no_grad(): pred=model(xt,t,cb)
        mse=float(((pred-xb)**2).mean().cpu()); den=(pred.norm(dim=1)*xb.norm(dim=1)).clamp_min(1e-12); cos=float(((pred*xb).sum(1)/den).mean().cpu()); rms=float(pred.square().mean().sqrt().cpu()); out[str(t0)]={'MSE':mse,'target_cos_mean':cos,'prediction_RMS':rms}
    return {'events':int(len(ix)),'event_index_seed':int(seed)+7001,'by_t':out}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--generator',required=True); ap.add_argument('--out',required=True); ap.add_argument('--seed',type=int,required=True); a=ap.parse_args(); out=Path(a.out)
    if out.exists() and any(out.iterdir()): raise RuntimeError('refuse overwrite nonempty output')
    out.mkdir(parents=True,exist_ok=True); t0=time.time(); c=cfg(); rc=c['risk']; assets=ROOT/c['round6_assets']; pdx=ROOT/c['protocol_dir']; gp=Path(a.generator)
    if sha(assets/'reference_L100.npz')!=c['expected_reference_L100_sha256']: raise RuntimeError('reference hash mismatch')
    if a.seed==202610081 and sha(gp/'generator.pt')!=c['expected_generator081_sha256']: raise RuntimeError('generator081 hash mismatch')
    fit=load_edges(pdx/'fit_edges.csv'); probe=load_edges(pdx/'probe_edges.csv').sort_values('userID'); rr=pd.read_csv(pdx/'reranker_train_users.csv').userID.to_numpy(np.int64); target=dict(zip(probe.userID.astype(int),probe.itemID.astype(int)))
    ref=np.load(assets/'reference_L100.npz'); items=ref['items'].astype(np.int32); A=ref['A_mask'].astype(bool); uc=np.load(assets/'user_conditions.npz')['condition'].astype(np.float32); beh=np.load(assets/'teacher_behavior.npz'); mu=beh['cf_mean'].astype(np.float32); sd=beh['cf_std'].astype(np.float32); valid=beh['valid_dims'].astype(bool)
    embp=ROOT/c['round1_assets']/'embeddings.npz'
    if sha(embp)!=c['expected_embeddings_sha256']: raise RuntimeError('embedding hash mismatch')
    emb=np.load(embp); raw_cf=emb['collab_item'].astype(np.float32); std=beh['standardized'].astype(np.float32); recon=inverse_standardized(std[:,None,:],mu,sd,valid)[:,0]; recon_diff=float(np.max(np.abs(recon-raw_cf)))
    if recon_diff>1e-5: raise RuntimeError(f'inverse standardization mismatch {recon_diff}')
    degree=np.bincount(fit.itemID.to_numpy(np.int64),minlength=len(raw_cf)).astype(np.int64); norms=np.linalg.norm(raw_cf,axis=1)
    model,ck=load_model(gp/'generator.pt'); users=np.arange(len(uc),dtype=np.int64); samples=sample_users(model,uc,users,a.seed,int(rc['K']),ck['config']); risk=raw_dot_risk_for_users(samples,users,items,A,raw_cf,mu,sd,valid)
    np.savez_compressed(out/'risk_cache.npz',users=users.astype(np.int32),items=items,A_mask=A,weights=risk['weights'],rho=risk['rho'],compatibility=risk['compatibility'],valid_query=risk['valid_query'],invalid_reason=risk['invalid_reason'])
    cal=np.array([u for u in rr if not split_eval(u,rc['cal_eval_seed'],rc['cal_fraction'])],np.int64); evl=np.array([u for u in rr if split_eval(u,rc['cal_eval_seed'],rc['cal_fraction'])],np.int64)
    stats={}
    for name,us in [('CAL',cal),('EVAL',evl)]:
        st=query_stats(us,risk['rho'],risk['weights'],items,A,target,degree,norms,rc['degree_log1p_tolerance'],rc['norm_log_tolerance']); st['pair_win_bootstrap']=bootstrap_mean(st['query_pair_values'],rc['bootstrap_resamples'],int(rc['bootstrap_seed'])+(0 if name=='CAL' else 1)); stats[name]=st
    eligible=np.array([x['user'] for x in stats['EVAL']['query_rows']],np.int64); shuf=uc.copy(); rng=np.random.default_rng(int(rc['cal_eval_seed'])+991); perm=eligible[rng.permutation(len(eligible))] if len(eligible) else eligible
    if len(eligible): shuf[eligible]=uc[perm]
    ss=sample_users(model,shuf,eligible,a.seed,int(rc['K']),ck['config']) if len(eligible) else np.empty((0,int(rc['K']),64),np.float32); sr=raw_dot_risk_for_users(ss,eligible,items,A,raw_cf,mu,sd,valid) if len(eligible) else None
    frho=np.full_like(risk['rho'],np.nan); fw=np.full_like(risk['weights'],np.nan); fcomp=np.full_like(risk['compatibility'],np.nan)
    if sr is not None:
        for qi,u in enumerate(eligible): frho[int(u)]=sr['rho'][qi]; fw[int(u)]=sr['weights'][qi]; fcomp[int(u)]=sr['compatibility'][qi]
    sh=query_stats(eligible,frho,fw,items,A,target,degree,norms,rc['degree_log1p_tolerance'],rc['norm_log_tolerance']) if len(eligible) else {'queries':0,'query_pair_values':[],'query_rows':[]}
    paired=bootstrap_paired_delta(stats['EVAL']['query_pair_values'],sh['query_pair_values'],rc['bootstrap_resamples'],int(rc['bootstrap_seed'])+2) if len(eligible) else None
    # score/popularity geometry; diagnostic only
    corr={'degree':per_user_spearman(evl,risk['compatibility'],items,A,degree.astype(np.float32)),'cf_norm':per_user_spearman(evl,risk['compatibility'],items,A,norms.astype(np.float32))}
    # Mean-dot linearity and CPU/GPU check on a fixed subset.
    q_users=evl[:min(32,len(evl))]; q_samples=sample_users(model,uc,q_users,a.seed,int(rc['K']),ck['config']); qraw=inverse_standardized(q_samples,mu,sd,valid); lin=[]; cpu_gpu=[]
    for qi,u in enumerate(q_users):
        pos=np.flatnonzero(A[int(u)]); ids=items[int(u),pos]; left=raw_cf[ids]@qraw[qi].mean(0); right=(qraw[qi]@raw_cf[ids].T).mean(0); lin.append(float(np.max(np.abs(left-right))))
        tg=torch.as_tensor(qraw[qi].mean(0),device='cuda:0'); ig=torch.as_tensor(raw_cf[ids],device='cuda:0'); gpu=(ig@tg).cpu().numpy(); cpu_gpu.append(float(np.max(np.abs(left-gpu))))
    invalid=risk['invalid_reason']; invalid_counts={'A_lt2':int((invalid==1).sum()),'nonfinite':int((invalid==2).sum()),'score_range':int((invalid==3).sum()),'valid':int(risk['valid_query'].sum()),'total':int(len(invalid))}
    events=np.load(assets/'events.npz'); final_diag=denoise_diag(model,events,a.seed,ck['config'])
    gate={'min_eval_queries':stats['EVAL']['queries']>=int(rc['min_eval_queries']),'pair_win':stats['EVAL']['pair_win']>=float(rc['min_pair_win']),'enrichment':stats['EVAL']['top_risk_enrichment']>=float(rc['min_enrichment']),'weight_direction':stats['EVAL']['known_positive_mean_w']<stats['EVAL']['unobserved_pooled_mean_w'],'degree_pair_win':stats['EVAL']['degree_pair_win'] is not None and stats['EVAL']['degree_pair_win']>float(rc['min_degree_pair_win'])}; passed=all(gate.values())
    def clean(st):
        z=dict(st); z.pop('query_pair_values',None); z.pop('query_rows',None); return z
    result={'status':'RISK_DIRECTION_PASS' if passed else 'RISK_DIRECTION_FAIL','protocol_version':c['protocol_version'],'risk_mode':'raw_cf_dot_mean','generator_seed':a.seed,'generator_sha256':sha(gp/'generator.pt'),'assets_reference_sha256':sha(assets/'reference_L100.npz'),'inverse_standardization_max_abs_diff':recon_diff,'valid_dims':int(valid.sum()),'CAL':clean(stats['CAL']),'EVAL':clean(stats['EVAL']),'SHUFFLED_EVAL':clean(sh),'true_minus_shuffled_pair_win':paired,'personalization_status':'PERSONALIZATION_UNRESOLVED' if paired and paired['ci95'][0]<=0<=paired['ci95'][1] else 'DIRECTIONAL_DIAGNOSTIC','correlations':corr,'invalid_counts':invalid_counts,'linearity_max_abs_diff':max(lin) if lin else None,'cpu_gpu_score_max_abs_diff':max(cpu_gpu) if cpu_gpu else None,'final_model_denoising_diagnostic':final_diag,'gate':gate,'gate_pass':passed,'risk_cache_sha256':sha(out/'risk_cache.npz'),'elapsed_seconds':time.time()-t0,'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'access':{'DEV':False,'INTERNAL':False,'CONFIRM':False,'TEST':False}}
    (out/'preflight.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({'status':result['status'],'seed':a.seed,'EVAL_queries':result['EVAL']['queries'],'pair_win':result['EVAL']['pair_win'],'enrichment':result['EVAL']['top_risk_enrichment'],'w_pos':result['EVAL']['known_positive_mean_w'],'w_unobs':result['EVAL']['unobserved_pooled_mean_w'],'degree_pair_win':result['EVAL']['degree_pair_win'],'shuffle_pair_win':result['SHUFFLED_EVAL']['pair_win'],'paired_delta':paired,'gate':gate},sort_keys=True))
if __name__=='__main__': main()
