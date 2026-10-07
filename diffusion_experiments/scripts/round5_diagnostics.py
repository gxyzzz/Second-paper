from __future__ import annotations
import argparse,hashlib,json,sys,time,subprocess
from pathlib import Path
import numpy as np, torch
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from diffusion_experiments.models.round5_behavior_diffusion import BehaviorDiffusion,ddim_trajectory
from diffusion_experiments.modules.round5_common import *
from diffusion_experiments.modules.round5_diffusion_sampler import build_blacklists,build_selector_cache,choose_from_cache
from diffusion_experiments.scripts.round5_train_generator import event_noise
from diffusion_experiments.scripts.round5_train_backbone import schedule

def qstats(x):
    a=np.asarray(x,dtype=np.float64); a=a[np.isfinite(a)]
    return {'n':int(len(a)),'mean':float(a.mean()) if len(a) else None,'p10':float(np.quantile(a,.1)) if len(a) else None,'p50':float(np.quantile(a,.5)) if len(a) else None,'p90':float(np.quantile(a,.9)) if len(a) else None}
def jacc(a,b):
    a=set(map(int,a)); b=set(map(int,b)); u=a|b
    return len(a&b)/len(u) if u else 1.0

def make_shuffled_states(gen_dir,ev,cfg,seed,sample_ix):
    ck=torch.load(Path(gen_dir)/'generator.pt',map_location='cpu',weights_only=False); dc=cfg['behavior_diffusion']; device=torch.device('cuda:0')
    m=BehaviorDiffusion(int(ck['x_dim']),int(ck['cond_dim']),int(dc['hidden_dim']),int(dc['time_dim']),float(dc['dropout']),int(dc['hidden_layers'])).to(device); m.load_state_dict(ck['model'],strict=True); m.eval(); alpha=ck['alpha_bar'].to(device)
    all_users=ev['users'].astype(np.int64); all_pos=ev['pos_items'].astype(np.int64); all_cond=ev['condition'].astype(np.float32); sample_ix=np.asarray(sample_ix,np.int64); users=all_users[sample_ix]; pos=all_pos[sample_ix]; cond=all_cond[sample_ix]; rng=np.random.default_rng(202610079); perm=rng.permutation(len(cond)); cs=cond.copy(); cs[:,64:]=cond[perm,64:]
    path=[int(x) for x in dc['ddim_path']]; saves=set(int(x) for x in dc['save_states']); states={s:np.empty((len(cond),int(ck['x_dim'])),np.float32) for s in saves}; deltas=[]
    orig=np.load(Path(gen_dir)/'trajectory_states.npz')
    for st in range(0,len(cond),512):
        en=min(st+512,len(cond)); nz=event_noise(users[st:en],pos[st:en],seed,int(ck['x_dim'])); _,sv=ddim_trajectory(m,torch.as_tensor(nz,device=device),torch.as_tensor(cs[st:en],device=device),alpha,path,saves)
        for s in saves:
            a=sv[s].cpu().numpy().astype(np.float32); states[s][st:en]=a; deltas.append(np.abs(a-orig[f'x_{s}'][sample_ix[st:en]]).mean())
    h=hashlib.sha256();
    for s in sorted(states): h.update(states[s].tobytes())
    return states,{'permutation_sha256':hashlib.sha256(perm.astype(np.int32).tobytes()).hexdigest(),'states_sha256':h.hexdigest(),'mean_abs_state_delta':float(np.mean(deltas)),'shuffle_definition':'fixed 10000-event label-free sample; keep positive-item T/V condition dims 0:64; permute user-specific collab/history/length dims 64: within sample using fixed permutation; same event-keyed DDIM noise'}

def state_overlap_sample(cand,A,traj,teacher_cf,text,visual,users,pos,black,observed,cfg,sample_ix):
    obs=np.zeros(len(teacher_cf),bool); obs[observed]=True; tnorm=teacher_cf/np.maximum(np.linalg.norm(teacher_cf,axis=1,keepdims=True),1e-12); gk=int(cfg['candidate']['grounding_per_state']); keep=float(cfg['candidate']['content_keep_fraction']); js=[]; unions=[]; inters=[]
    for e in sample_ix:
        u=int(users[e]); p=int(pos[e]); idx=np.flatnonzero(A[u]); ids=cand['items'][u,idx]; ok=np.array([obs[int(it)] and int(it) not in black[u] for it in ids]); idx=idx[ok]; ids=ids[ok]; sim=.5*(text[ids]@text[p]+visual[ids]@visual[p]); n=max(1,int(np.ceil(len(ids)*keep))); o=np.lexsort((ids,idx,-sim))[:n]; support=ids[o]; srank=idx[o]; sets=[]
        for s in [35,25,15]:
            q=traj[s][e]; qn=max(float(np.linalg.norm(q)),1e-12); co=tnorm[support]@(q/qn); z=np.lexsort((support,srank,-co))[:min(gk,len(support))]; sets.append(support[z].astype(np.int32))
        js.extend([jacc(sets[0],sets[1]),jacc(sets[0],sets[2]),jacc(sets[1],sets[2])]); un=set(map(int,np.concatenate(sets))); it=set(map(int,sets[0]))&set(map(int,sets[1]))&set(map(int,sets[2])); unions.append(len(un)); inters.append(len(it))
    return {'events':int(len(sample_ix)),'pairwise_state_jaccard':qstats(js),'union_size':qstats(unions),'all3_intersection_size':qstats(inters)}

def detailed_sample(cand,cache,traj,teacher_cf,text,visual,users,pos,fit_degree,sample_ix):
    tnorm=teacher_cf/np.maximum(np.linalg.norm(teacher_cf,axis=1,keepdims=True),1e-12); cfn=np.linalg.norm(teacher_cf,axis=1); cr=[]; rr=[]; gaps=[]; cs=[]; cfpos=[]; dis=[]; gcos=[]; deg=[]; norms=[]
    for e in sample_ix:
        u=int(users[e]); p=int(pos[e]); c=int(cache['pool_counts'][e]); ids=cache['pool_ids'][e,:c]
        b10=(float(cand['s0'][u,9])+float(cand['s0'][u,10]))/2; b20=(float(cand['s0'][u,19])+float(cand['s0'][u,20]))/2
        for it in ids:
            it=int(it); cj=np.flatnonzero(cand['items'][u]==it); rj=np.flatnonzero(cand['raw_items'][u]==it)
            if not len(cj) or not len(rj): continue
            j=int(cj[0]); cr.append(j+1); rr.append(int(rj[0])+1); gaps.append(min(abs(float(cand['s0'][u,j])-b10),abs(float(cand['s0'][u,j])-b20)))
            s=.5*(float(text[it]@text[p])+float(visual[it]@visual[p])); cfp=float(tnorm[it]@tnorm[p]); cs.append(s); cfpos.append(cfp); dis.append(abs(s-cfp)); gcos.append(max(float(tnorm[it]@(traj[k][e]/max(float(np.linalg.norm(traj[k][e])),1e-12))) for k in [35,25,15])); deg.append(float(fit_degree[it])); norms.append(float(cfn[it]))
    return {'diagnostic_events':int(len(sample_ix)),'grounded_pairs':int(len(cr)),'colift_rank_1based':qstats(cr),'raw_msca_rank_1based':qstats(rr),'nearest_cutoff_s0_distance':qstats(gaps),'content_cosine':qstats(cs),'positive_teacher_cf_cosine':qstats(cfpos),'semantic_cf_abs_disagreement':qstats(dis),'generated_state_grounding_cosine':qstats(gcos),'generated_state_grounding_distance_1_minus_cos':qstats(1-np.asarray(gcos)),'fit_degree':qstats(deg),'teacher_cf_norm':qstats(norms)}

def compare_caches(a,b):
    js=[]; exact=0; ca=a['pool_counts']; cb=b['pool_counts']
    for i in range(len(ca)):
        x=a['pool_ids'][i,:int(ca[i])]; y=b['pool_ids'][i,:int(cb[i])]; js.append(jacc(x,y)); exact+=set(map(int,x))==set(map(int,y))
    return {'events':int(len(ca)),'exact_pool_set_fraction':float(exact/len(ca)),'pool_jaccard':qstats(js),'mean_pool_count_original':float(ca.mean()),'mean_pool_count_shuffled':float(cb.mean())}

def uniform_comparison(run_dir,teacher_cf,fit_degree,black,users,cfg):
    plan=np.load(Path(run_dir)/'base_plan.npz'); cf_norm=np.linalg.norm(teacher_cf,axis=1); sel=[]; uni=[]; bad=0; planned=0
    caches={r:np.load(Path(run_dir)/'refresh'/f'cache_epoch{r:02d}.npz') for r in [0,5,10,15]}
    for ep in range(1,21):
        prob,bi,_=schedule(cfg,ep); ids=np.flatnonzero(plan['route'][ep-1]<prob); planned+=len(ids); r=max(x for x in caches if x<=ep-1); s=choose_from_cache(caches[r],ids,bi,202610071,ep); ok=s>=0; ids=ids[ok]; s=s[ok]; sel.extend(map(int,s)); uni.extend(map(int,plan['base_neg'][ep-1,ids])); bad+=sum(int(it) in black[int(users[e])] for e,it in zip(ids,s))
    sel=np.asarray(sel,np.int32); uni=np.asarray(uni,np.int32)
    return {'planned':int(planned),'selected':int(len(sel)),'illegal_selected_known_positive':int(bad),'same_as_uniform_fraction':float(np.mean(sel==uni)),'selected_unique_items':int(len(np.unique(sel))),'uniform_unique_items':int(len(np.unique(uni))),'selected_degree':qstats(fit_degree[sel]),'uniform_degree':qstats(fit_degree[uni]),'selected_teacher_cf_norm':qstats(cf_norm[sel]),'uniform_teacher_cf_norm':qstats(cf_norm[uni])}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--assets',required=True); ap.add_argument('--run-root',required=True); ap.add_argument('--out',required=True); a=ap.parse_args(); out=Path(a.out)
    if out.exists(): raise RuntimeError('refuse overwrite diagnostic output')
    out.mkdir(parents=True); t0=time.time()
    cfg=cfg_round5(); seed=202610071; asset=Path(a.assets); rr=Path(a.run_root); pdx=ROOT/cfg['protocol_dir']; fit=load_edges(pdx/'fit_edges.csv'); mon=load_edges(pdx/'monitor_edges.csv'); n_users=int(fit.userID.max())+1; n_items=int(np.load(ROOT/'data/baby/text_feat.npy',mmap_mode='r').shape[0]); histories=histories_from_fit(fit,n_users); black=build_blacklists(fit,mon,n_users); side=prepare_side(histories,n_items); ev=np.load(asset/'events.npz'); users=ev['users'].astype(np.int64); pos=ev['pos_items'].astype(np.int64); beh=np.load(asset/'teacher_behavior.npz'); teacher_cf=beh['standardized'].astype(np.float32); observed=beh['observed_items'].astype(np.int32); emb=np.load(asset/'teacher_embeddings.npz'); fu=emb['final_user'].astype(np.float32); fi=emb['final_item'].astype(np.float32); fit_degree=np.bincount(fit.itemID.to_numpy(np.int64),minlength=n_items).astype(np.int64); c=np.load(asset/'teacher_L500.npz'); cand={k:c[k] for k in c.files}; A=cand['A_mask'].astype(bool); gen=rr/f'generator_seed{seed}'; origz=np.load(gen/'trajectory_states.npz'); orig={s:origz[f'x_{s}'].astype(np.float32) for s in [35,25,15]}
    rng=np.random.default_rng(202610078); sample=np.sort(rng.choice(len(users),size=min(10000,len(users)),replace=False)); original_cache=np.load(rr/f'D_CURR_seed{seed}'/'refresh'/'cache_epoch00.npz')
    overlap_orig=state_overlap_sample(cand,A,orig,teacher_cf,side['text_unit'],side['visual_unit'],users,pos,black,observed,cfg,sample); detailed=detailed_sample(cand,original_cache,orig,teacher_cf,side['text_unit'],side['visual_unit'],users,pos,fit_degree,sample)
    shuffled,shmeta=make_shuffled_states(gen,ev,cfg,seed,sample); shuffled_cache,shstats=build_selector_cache(cand,A,shuffled,teacher_cf,side['text_unit'],side['visual_unit'],fu,fi,users[sample],pos[sample],black,observed,fit_degree,cfg); overlap_shuf=state_overlap_sample(cand,A,shuffled,teacher_cf,side['text_unit'],side['visual_unit'],users[sample],pos[sample],black,observed,cfg,np.arange(len(sample))); original_sample_cache={'pool_ids':original_cache['pool_ids'][sample],'pool_counts':original_cache['pool_counts'][sample],'pool_bins':original_cache['pool_bins'][sample]}; poolcmp=compare_caches(original_sample_cache,shuffled_cache)
    uc=uniform_comparison(rr/f'D_CURR_seed{seed}',teacher_cf,fit_degree,black,users,cfg); gens={str(s):json.load(open(rr/f'generator_seed{s}'/'result.json')) for s in cfg['formal_seeds']}; druns={str(s):json.load(open(rr/f'D_CURR_seed{s}'/'result.json')) for s in cfg['formal_seeds']}
    result={'status':'COMPLETE_POSTHOC_MECHANISM_DIAGNOSTIC','seed':seed,'sample_definition':{'size':int(len(sample)),'rng_seed':202610078,'label_free':True},'generator_summary':{s:{'last_loss':g['history'][-1]['loss'],'condition_sensitivity_mean_abs_delta':g['condition_sensitivity_mean_abs_delta'],'trajectory_norms':g['trajectory_norms'],'trajectory_cosine_to_target':g['trajectory_cosine_to_target'],'replay_max_abs_diff':g['replay_max_abs_diff']} for s,g in gens.items()},'original_state_overlap':overlap_orig,'grounded_candidate_detailed_sample':detailed,'shuffled_user_condition':{**shmeta,'state_overlap':overlap_shuf,'selector_stats':shstats,'vs_original_pool':poolcmp},'selected_vs_uniform':uc,'formal_selector_full_event_stats':{s:{'total_planned_routes':d['total_planned_routes'],'total_actual_replacements':d['total_actual_replacements'],'total_fallbacks':d['total_fallbacks'],'selector_nonfallback_fraction':d['selector_nonfallback_fraction'],'unique_replaced_users':d['unique_replaced_users'],'unique_replaced_items':d['unique_replaced_items'],'refresh_stats':d['refresh_stats']} for s,d in druns.items()},'known_positive_policy':'FIT+monitor positives excluded before grounding; DEV/INTERNAL/Test labels never used to filter training negatives; unknown false negatives remain a limitation','access':{'DEV_LABELS_ACCESSED':False,'INTERNAL_LABELS_ACCESSED':False,'CONFIRM_ACCESSED':False,'TEST_LABELS_USED':False},'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'diagnostics_script_sha256':sha(Path(__file__)),'assets_audit_sha256':sha(asset/'audit.json'),'generator_result_sha256':sha(gen/'result.json'),'d_curr_result_sha256':sha(rr/f'D_CURR_seed{seed}'/'result.json'),'elapsed_seconds':float(time.time()-t0)}; (out/'diagnostics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'orig_jaccard':overlap_orig['pairwise_state_jaccard']['mean'],'shuffle_delta':shmeta['mean_abs_state_delta'],'pool_exact_after_shuffle':poolcmp['exact_pool_set_fraction'],'selected_vs_uniform_same':uc['same_as_uniform_fraction']},sort_keys=True))
if __name__=='__main__': main()
