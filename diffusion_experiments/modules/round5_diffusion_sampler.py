from __future__ import annotations
import hashlib
import numpy as np

def build_blacklists(fit,monitor,n_users):
    out=[set() for _ in range(n_users)]
    for df in (fit,monitor):
        for u,g in df.groupby('userID',sort=False): out[int(u)].update(g.itemID.astype(int).tolist())
    return out

def make_base_plan(users,observed,blacklists,seed,epochs):
    users=np.asarray(users,np.int64); observed=np.asarray(observed,np.int32); N=len(users); neg=np.empty((epochs,N),np.int32); route=np.empty((epochs,N),np.float32); order=np.empty((epochs,N),np.int32)
    for ep in range(epochs):
        rng=np.random.default_rng(int(seed)+1000003*(ep+1)); order[ep]=rng.permutation(N).astype(np.int32); route[ep]=rng.random(N,dtype=np.float32)
        for i,u in enumerate(users):
            while True:
                x=int(observed[int(rng.integers(0,len(observed)))])
                if x not in blacklists[int(u)]: neg[ep,i]=x; break
    return {'base_neg':neg,'route':route,'order':order}

def deterministic_choice(ids,seed,epoch,event_idx):
    ids=np.asarray(ids,np.int32)
    if not len(ids): return -1
    b=f'{int(seed)}|{int(epoch)}|{int(event_idx)}'.encode(); q=int.from_bytes(hashlib.sha256(b).digest()[:8],'little'); return int(ids[q%len(ids)])

def hardness_bins(sorted_ids,count):
    if count<3: return [sorted_ids[:count],sorted_ids[:count],sorted_ids[:count]]
    return [x.astype(np.int32) for x in np.array_split(sorted_ids[:count],3)]
def build_selector_cache(cand,A,trajectory_states,teacher_cf_std,text_unit,visual_unit,student_fu,student_fi,event_users,event_pos,blacklists,observed_items,fit_degree,cfg):
    N=len(event_users); maxpool=15; pools=np.full((N,maxpool),-1,np.int32); counts=np.zeros(N,np.int16); bins=np.full((N,maxpool),-1,np.int8)
    observed_mask=np.zeros(teacher_cf_std.shape[0],bool); observed_mask[np.asarray(observed_items,np.int64)]=True
    tnorm=teacher_cf_std/np.maximum(np.linalg.norm(teacher_cf_std,axis=1,keepdims=True),1e-12); cf_norm=np.linalg.norm(teacher_cf_std,axis=1)
    stats={'events':N,'empty_A':0,'empty_legal':0,'empty_content_support':0,'empty_grounded':0,'known_positive_rejections':0,'grounded_total':0,'pool_lt3':0}; pool_sizes=[]; csims=[]; gcos=[]; hard_all=[]; hard_bins=[[],[],[]]; degrees=[]; cfnorms=[]; unique=set()
    keep_frac=float(cfg['candidate']['content_keep_fraction']); gk=int(cfg['candidate']['grounding_per_state'])
    for e,(u,p) in enumerate(zip(event_users,event_pos)):
        u=int(u); p=int(p); posidx=np.flatnonzero(A[u])
        if not len(posidx): stats['empty_A']+=1; continue
        ids=cand['items'][u,posidx].astype(np.int32); legal=[]; lranks=[]
        for j,it in zip(posidx,ids):
            if not observed_mask[int(it)]: continue
            if int(it) in blacklists[u]: stats['known_positive_rejections']+=1; continue
            legal.append(int(it)); lranks.append(int(j))
        if not legal: stats['empty_legal']+=1; continue
        legal=np.asarray(legal,np.int32); lranks=np.asarray(lranks,np.int32); sim=.5*(text_unit[legal]@text_unit[p]+visual_unit[legal]@visual_unit[p]); nkeep=max(1,int(np.ceil(len(legal)*keep_frac))); ords=np.lexsort((legal,lranks,-sim))[:nkeep]; support=legal[ords]; srank=lranks[ords]; ssim=sim[ords]
        if not len(support): stats['empty_content_support']+=1; continue
        chosen={}
        for state_id,state_arr in trajectory_states.items():
            q=state_arr[e].astype(np.float32); qn=float(np.linalg.norm(q))
            if not np.isfinite(qn) or qn<=1e-12: continue
            co=tnorm[support]@(q/qn); o=np.lexsort((support,srank,-co))[:min(gk,len(support))]
            for z in o:
                it=int(support[z]); val=float(co[z]); rank=int(srank[z]); csi=float(ssim[z]); old=chosen.get(it)
                if old is None or val>old[0]: chosen[it]=(val,rank,csi,int(state_id))
        if not chosen: stats['empty_grounded']+=1; continue
        ids2=np.asarray(list(chosen),np.int32); pos_score=float(student_fu[u]@student_fi[p]); h=np.asarray([float(student_fu[u]@student_fi[int(it)]-pos_score) for it in ids2],np.float32); ranks2=np.asarray([chosen[int(it)][1] for it in ids2],np.int32); o=np.lexsort((ids2,ranks2,h)); ids2=ids2[o]; h=h[o]
        c=min(len(ids2),maxpool); pools[e,:c]=ids2[:c]; counts[e]=c; split=hardness_bins(ids2,c)
        for bi,arr in enumerate(split):
            aset=set(map(int,arr.tolist()))
            for j in range(c):
                if int(pools[e,j]) in aset: bins[e,j]=bi
        if c<3: stats['pool_lt3']+=1
        pool_sizes.append(c); stats['grounded_total']+=c
        for it,hh in zip(ids2[:c],h[:c]):
            unique.add(int(it)); hard_all.append(float(hh)); degrees.append(float(fit_degree[int(it)])); cfnorms.append(float(cf_norm[int(it)])); csims.append(float(chosen[int(it)][2])); gcos.append(float(chosen[int(it)][0]))
        for bi,arr in enumerate(split):
            aset=set(map(int,arr.tolist())); hard_bins[bi].extend([float(hh) for it,hh in zip(ids2[:c],h[:c]) if int(it) in aset])
    stats.update({'mean_pool_size':float(np.mean(pool_sizes)) if pool_sizes else 0.0,'median_pool_size':float(np.median(pool_sizes)) if pool_sizes else 0.0,'unique_grounded_items':int(len(unique)),'mean_content_similarity':float(np.mean(csims)) if csims else None,'mean_teacher_cf_cosine':float(np.mean(gcos)) if gcos else None,'mean_hardness':float(np.mean(hard_all)) if hard_all else None,'hardness_by_bin':[float(np.mean(v)) if v else None for v in hard_bins],'mean_degree':float(np.mean(degrees)) if degrees else None,'mean_teacher_cf_norm':float(np.mean(cfnorms)) if cfnorms else None,'selector_active_rate':float((counts>0).mean())})
    return {'pool_ids':pools,'pool_counts':counts,'pool_bins':bins},stats

def choose_from_cache(cache,event_indices,bin_id,seed,epoch):
    out=np.full(len(event_indices),-1,np.int32)
    for q,e in enumerate(event_indices):
        e=int(e); c=int(cache['pool_counts'][e]); ids=cache['pool_ids'][e,:c]; bb=cache['pool_bins'][e,:c]
        cand=ids[bb==int(bin_id)] if c>=3 else ids
        if len(cand): out[q]=deterministic_choice(cand,seed,epoch,e)
    return out
