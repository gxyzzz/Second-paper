from __future__ import annotations
import numpy as np
from scipy.stats import rankdata, spearmanr

def inverse_standardized(samples_std, cf_mean, cf_std, valid_dims):
    x=np.asarray(samples_std,np.float32); mu=np.asarray(cf_mean,np.float32); sd=np.asarray(cf_std,np.float32); valid=np.asarray(valid_dims,bool)
    if x.shape[-1]!=int(valid.sum()): raise ValueError('standardized dimension mismatch')
    out=np.broadcast_to(mu, x.shape[:-1]+(len(mu),)).copy().astype(np.float32)
    out[...,valid]=mu[valid]+sd[valid]*x
    return out

def raw_dot_risk_for_users(samples_std,users,items,A_mask,raw_cf,cf_mean,cf_std,valid_dims):
    users=np.asarray(users,np.int64); L=items.shape[1]; weights=np.full((len(users),L),np.nan,np.float32); rho=np.full_like(weights,np.nan); compat=np.full_like(weights,np.nan); valid_q=np.zeros(len(users),bool); reasons=np.zeros(len(users),np.int8)
    raw_samples=inverse_standardized(samples_std,cf_mean,cf_std,valid_dims)
    for qi,u in enumerate(users):
        pos=np.flatnonzero(A_mask[int(u)]); n=len(pos)
        if n<2: reasons[qi]=1; continue
        s=raw_samples[qi]
        if not np.isfinite(s).all(): reasons[qi]=2; continue
        q=s.mean(0); a=raw_cf[items[int(u),pos]]@q
        if not np.isfinite(a).all(): reasons[qi]=2; continue
        if float(a.max()-a.min())<=1e-6: reasons[qi]=3; continue
        rr=(rankdata(a,method='average')-1)/float(n-1); ww=(1-rr)**2
        compat[qi,pos]=a.astype(np.float32); rho[qi,pos]=rr.astype(np.float32); weights[qi,pos]=ww.astype(np.float32); valid_q[qi]=True
    return {'weights':weights,'rho':rho,'compatibility':compat,'valid_query':valid_q,'invalid_reason':reasons,'raw_sample_mean':raw_samples.mean(1).astype(np.float32)}

def degree_match_mask(items_u,neg_positions,target_item,degree,tol):
    neg_positions=np.asarray(neg_positions,np.int64); ids=np.asarray(items_u,np.int64)[neg_positions]
    return np.abs(np.log1p(degree[ids])-np.log1p(degree[int(target_item)]))<=float(tol)

def joint_degree_norm_mask(items_u,neg_positions,target_item,degree,norms,degree_tol,norm_tol):
    ids=np.asarray(items_u,np.int64)[np.asarray(neg_positions,np.int64)]
    d=np.abs(np.log1p(degree[ids])-np.log1p(degree[int(target_item)]))<=float(degree_tol)
    n=np.abs(np.log(norms[ids]+1e-8)-np.log(norms[int(target_item)]+1e-8))<=float(norm_tol)
    return d&n

def query_stats(users,rho,w,items,A,target,degree,norms,degree_tol,norm_tol):
    pair=[]; dpair=[]; jpair=[]; posw=[]; negw=[]; negw_q=[]; pos_top=[]; all_top=all_items=0; no_deg=no_joint=0; rows=[]
    for u0 in users:
        u=int(u0); q=np.flatnonzero(items[u]==int(target[u]))
        if not len(q): continue
        j=int(q[0]); idx=np.flatnonzero(A[u]&np.isfinite(rho[u]))
        if j not in set(idx.tolist()) or len(idx)<2: continue
        neg=idx[idx!=j]; pr=float(rho[u,j]); pw=float(w[u,j]); nr=rho[u,neg]; nw=w[u,neg]
        win=float(np.mean((pr>nr)+.5*(pr==nr))); pair.append(win); posw.append(pw); negw.extend(nw.tolist()); negw_q.append(float(np.mean(nw)))
        top=rho[u,idx]>=.75; all_top+=int(top.sum()); all_items+=len(idx); pos_top.append(int(pr>=.75))
        dm=degree_match_mask(items[u],neg,int(target[u]),degree,degree_tol)
        if dm.any():
            x=nr[dm]; dpair.append(float(np.mean((pr>x)+.5*(pr==x))))
        else: no_deg+=1
        jm=joint_degree_norm_mask(items[u],neg,int(target[u]),degree,norms,degree_tol,norm_tol)
        if jm.any():
            x=nr[jm]; jpair.append(float(np.mean((pr>x)+.5*(pr==x))))
        else: no_joint+=1
        rows.append({'user':u,'pair_win':win,'positive_w':pw,'neg_mean_w':float(np.mean(nw)),'top_positive':int(pr>=.75),'A_n':int(len(idx))})
    n=len(pair); all_rate=n/max(all_items,1); top_rate=sum(pos_top)/max(all_top,1)
    return {'queries':n,'pair_win':float(np.mean(pair)) if pair else None,'degree_matched_queries':len(dpair),'degree_pair_win':float(np.mean(dpair)) if dpair else None,'degree_no_match_queries':no_deg,'joint_matched_queries':len(jpair),'joint_pair_win':float(np.mean(jpair)) if jpair else None,'joint_no_match_queries':no_joint,'known_positive_mean_w':float(np.mean(posw)) if posw else None,'unobserved_pooled_mean_w':float(np.mean(negw)) if negw else None,'unobserved_query_mean_w':float(np.mean(negw_q)) if negw_q else None,'all_A_target_rate':float(all_rate),'top_risk_target_rate':float(top_rate),'top_risk_positive_coverage':float(np.mean(pos_top)) if pos_top else None,'top_risk_enrichment':float(top_rate/max(all_rate,1e-12)),'query_pair_values':pair,'query_rows':rows}

def per_user_spearman(users,score,items,A,feature):
    vals=[]
    for u0 in users:
        u=int(u0); idx=np.flatnonzero(A[u]&np.isfinite(score[u]));
        if len(idx)<3: continue
        a=score[u,idx]; b=feature[items[u,idx]]
        if np.ptp(a)<=1e-12 or np.ptp(b)<=1e-12: continue
        r=spearmanr(a,b).statistic
        if np.isfinite(r): vals.append(float(r))
    return {'queries':len(vals),'mean':float(np.mean(vals)) if vals else None,'median':float(np.median(vals)) if vals else None}

def bootstrap_mean(vals,resamples,seed):
    x=np.asarray(vals,float); rng=np.random.default_rng(int(seed)); out=np.empty(int(resamples),float)
    for i in range(len(out)): out[i]=x[rng.integers(0,len(x),len(x))].mean()
    return {'mean':float(x.mean()),'ci95':[float(np.quantile(out,.025)),float(np.quantile(out,.975))],'resamples':int(resamples),'seed':int(seed)}

def bootstrap_paired_delta(a,b,resamples,seed):
    x=np.asarray(a,float)-np.asarray(b,float); return bootstrap_mean(x,resamples,seed)
