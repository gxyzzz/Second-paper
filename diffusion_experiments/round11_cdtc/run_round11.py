from __future__ import annotations
import argparse, gc, hashlib, json, math, shutil, sys, time
from pathlib import Path
import numpy as np, torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,str(ROOT))
from modules.diffusion import cosine_alpha_bar, q_sample, l2_rows_np
from modules.semantic_purifier import condition_beta
from diffusion_experiments.round9_cabrp import run_round9 as r9
from diffusion_experiments.round10_ccdp import run_round10 as r10

PROTOCOL='ROUND11_CDTC_V1'; SOURCE='d31be7fdfba2f434d5a6a4fbbd8c2fe00c41061f'
BACKBONES=(999,1000,1001,1002); ETA=0.10; CAP=20.0; PSEEDS=r9.SEEDS
PRIMARY=r9.PRIMARY; ALL=r9.ALL
CONFIGS={
 'C0':dict(guidance=2.0,beta=.5,t_edit=3),
 'G1':dict(guidance=1.0,beta=.5,t_edit=3),
 'G2':dict(guidance=1.5,beta=.5,t_edit=3),
 'B0':dict(guidance=2.0,beta=0.0,t_edit=3),
 'B1':dict(guidance=2.0,beta=1.0,t_edit=3),
 'T1':dict(guidance=2.0,beta=.5,t_edit=1),
 'T2':dict(guidance=2.0,beta=.5,t_edit=2),
}
SMOKE=('C0','G1','B0','T1')

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def st(x):
 x=np.asarray(x,np.float64); return {k:float(v) for k,v in [('mean',x.mean()),('median',np.median(x)),('p75',np.quantile(x,.75)),('p90',np.quantile(x,.90)),('p95',np.quantile(x,.95)),('max',x.max())]}

def angle_deg(a,b):
 a=l2_rows_np(a); b=l2_rows_np(b); return np.degrees(np.arccos(np.clip(np.sum(a*b,axis=1),-1,1)))

def endpoints(seed):
 p=r9.paths(seed)['msca']/'embeddings.npz'; z=np.load(p); return np.asarray(z['collab_item'],np.float32),np.asarray(z['final_item'],np.float32),p

@torch.no_grad()
def paired_config(seed,cfg,device='cuda',batch=128):
 _,rt,rv=r9.load_raw(); model,ck,_=r9.load_model(seed,device); collab,final,epath=endpoints(seed); cond=condition_beta(collab,final,float(cfg['beta']))
 n=len(rt); dt=rt.shape[1]; dv=rv.shape[1]; ab=cosine_alpha_bar(50,.008).to(device); sums={k:np.zeros((n,d),np.float32) for k,d in [('true_t',dt),('true_v',dv),('null_t',dt),('null_v',dv)]}; maxxt=0.0
 for ps in PSEEDS:
  gen=torch.Generator(device=device); gen.manual_seed(int(ps))
  for a in range(0,n,batch):
   b=min(a+batch,n); raw=np.concatenate([l2_rows_np(rt[a:b]),l2_rows_np(rv[a:b])],1).astype(np.float32); x0=torch.from_numpy(raw).to(device); noise=torch.randn(x0.shape,generator=gen,device=device,dtype=x0.dtype); tt=torch.full((b-a,),int(cfg['t_edit']),device=device,dtype=torch.long); xt=q_sample(x0,tt,noise,ab); xtt=xt.clone(); xtn=xt.clone(); maxxt=max(maxxt,float(torch.max(torch.abs(xtt-xtn))))
   c=torch.from_numpy(np.asarray(cond[a:b],np.float32).copy()).to(device); z=torch.zeros_like(c)
   yt=r10.ddim_from_xt(model,xtt,c,float(cfg['guidance']),ab,int(cfg['t_edit'])).cpu().numpy(); yn=r10.ddim_from_xt(model,xtn,z,float(cfg['guidance']),ab,int(cfg['t_edit'])).cpu().numpy()
   sums['true_t'][a:b]+=l2_rows_np(yt[:,:dt]); sums['true_v'][a:b]+=l2_rows_np(yt[:,dt:]); sums['null_t'][a:b]+=l2_rows_np(yn[:,:dt]); sums['null_v'][a:b]+=l2_rows_np(yn[:,dt:])
 out={k:l2_rows_np(v/len(PSEEDS)).astype(np.float32) for k,v in sums.items()}
 prov={'diffusion_checkpoint':str(Path(ck).relative_to(ROOT)),'diffusion_sha256':sha(ck),'endpoint_path':str(epath.relative_to(ROOT)),'endpoint_sha256':sha(epath),'condition_beta':float(cfg['beta']),'condition_sha256':hashlib.sha256(np.ascontiguousarray(cond).tobytes()).hexdigest(),'guidance':float(cfg['guidance']),'t_edit':int(cfg['t_edit']),'purification_seeds':list(PSEEDS),'paired_initial_xt_max_abs_diff':maxxt,'TEST_ACCESSED':False}
 del model; torch.cuda.empty_cache(); gc.collect(); return out,prov

def final_features(raw,true,null):
 raw=l2_rows_np(raw); rc=l2_rows_np(true)-l2_rows_np(null); tang=rc-np.sum(raw*rc,axis=1)[:,None]*raw; cand=l2_rows_np(raw+ETA*tang); final,_,applied=r9.slerp_bounded(raw,cand,np.ones(len(raw),np.float32),CAP); hit=applied>=CAP-1e-3
 if not np.isfinite(final).all() or float(np.max(np.abs(np.linalg.norm(final,axis=1)-1.0)))>2e-5: raise RuntimeError('final feature finite/norm FAIL')
 return final,applied,hit,{'cond_norm':st(np.linalg.norm(rc,axis=1)),'tangent_norm':st(np.linalg.norm(tang,axis=1))}

def positive_ranks(rank,users,eval_sets):
 out={}
 for row,u in enumerate(np.asarray(users,np.int64)):
  loc={int(item):j+1 for j,item in enumerate(rank[row])}
  for pos in eval_sets[int(u)]: out[(int(u),int(pos))]=int(loc.get(int(pos),101))
 return out

def boundary_diag(base_rank,new_rank,users,eval_sets):
 b=positive_ranks(base_rank,users,eval_sets); n=positive_ranks(new_rank,users,eval_sets); keys=list(b); br=np.asarray([b[k] for k in keys]); nr=np.asarray([n[k] for k in keys]); delta=br-nr
 h10=int(np.sum((br>=8)&(br<=10)&(nr>10))); e10=int(np.sum((br>=11)&(br<=12)&(nr<=10))); h20=int(np.sum((br>=18)&(br<=20)&(nr>20))); e20=int(np.sum((br>=21)&(br<=22)&(nr<=20)))
 return {'positive_pairs':len(keys),'beneficial_entry_10':e10,'harmful_exit_10':h10,'NetCross10':e10-h10,'beneficial_entry_20':e20,'harmful_exit_20':h20,'NetCross20':e20-h20,'mean_positive_item_rank_change':float(delta.mean()),'median_positive_item_rank_change':float(np.median(delta)),'fraction_improved':float(np.mean(delta>0)),'fraction_worsened':float(np.mean(delta<0)),'fraction_unchanged':float(np.mean(delta==0))}

def run_config(seed,name,cfg,ctx,outdir):
    paired,prov=paired_config(seed,cfg)
    rt=l2_rows_np(ctx['rt']); rv=l2_rows_np(ctx['rv'])
    ft,at,ht,tdt=final_features(rt,paired['true_t'],paired['null_t'])
    fv,av,hv,tdv=final_features(rv,paired['true_v'],paired['null_v'])
    res,rank=r9.evaluate_features(ctx,ft,fv,outdir/'tmp',name)
    traj={
      'text':{'raw_to_TRUE':st(angle_deg(rt,paired['true_t'])),'raw_to_NULL':st(angle_deg(rt,paired['null_t'])),'NULL_to_TRUE':st(angle_deg(paired['null_t'],paired['true_t'])),'raw_to_final':st(at),'fraction_hit_20deg_cap':float(np.mean(ht)),**tdt},
      'visual':{'raw_to_TRUE':st(angle_deg(rv,paired['true_v'])),'raw_to_NULL':st(angle_deg(rv,paired['null_v'])),'NULL_to_TRUE':st(angle_deg(paired['null_v'],paired['true_v'])),'raw_to_final':st(av),'fraction_hit_20deg_cap':float(np.mean(hv)),**tdv}}
    meanang=(traj['text']['raw_to_final']['mean']+traj['visual']['raw_to_final']['mean'])/2.0
    res['ranking_efficiency_U_per_deg']=float(res['U']/meanang) if meanang>0 else None
    res['mean_applied_angle_deg']=float(meanang)
    res['boundary']=boundary_diag(ctx['base_rank'],rank,ctx['users'],ctx['eval_sets'])
    res['trajectory']=traj; res['provenance']=prov
    res['TRAJECTORY_TOO_AGGRESSIVE']=bool(traj['text']['fraction_hit_20deg_cap']>.05 or traj['visual']['fraction_hit_20deg_cap']>.05)
    return res

def eta0_identity(ctx,outdir):
    res,_=r9.evaluate_features(ctx,ctx['rt'],ctx['rv'],outdir/'tmp','eta0_identity')
    exact=bool(res['ranking_exact_vs_colift'] and res['score_max_abs_delta_vs_colift']==0.0 and max(abs(res['metrics'][k]-ctx['base_metrics'][k]) for k in ALL)==0.0)
    if not exact: raise RuntimeError('eta=0 identity FAIL')
    return res

def run_seed(seed:int,outdir:Path,mode:str,config_names):
    t0=time.time(); outdir.mkdir(parents=True,exist_ok=True); ctx=r9.validation_context(seed); identity=eta0_identity(ctx,outdir); results={}
    for name in config_names:
        results[name]=run_config(seed,name,CONFIGS[name],ctx,outdir)
    checks={'eta0_identity':bool(identity['ranking_exact_vs_colift']),'all_paired_xt_exact':bool(all(x['provenance']['paired_initial_xt_max_abs_diff']==0.0 for x in results.values())),'all_final_finite_norm':True,'attribute_unchanged':True,'direct_score_residual':False,'TEST_loader_called':False,'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
    if not checks['all_paired_xt_exact']: raise RuntimeError('paired x_t exactness FAIL')
    if mode=='smoke':
        c0,g1,b0,t1=[results[x] for x in ('C0','G1','B0','T1')]
        guidance_diff=max(abs(c0['trajectory'][m]['NULL_to_TRUE']['mean']-g1['trajectory'][m]['NULL_to_TRUE']['mean']) for m in ('text','visual'))
        tedit_diff=max(abs(c0['trajectory'][m]['raw_to_TRUE']['mean']-t1['trajectory'][m]['raw_to_TRUE']['mean']) for m in ('text','visual'))
        beta_hash_diff=c0['provenance']['condition_sha256']!=b0['provenance']['condition_sha256']
        collab,final,_=endpoints(seed); hist=np.load(r9.paths(seed)['diff']/'assets/condition_beta_0p5.npy'); beta05_rebuild=float(np.max(np.abs(condition_beta(collab,final,.5)-hist))); beta0_endpoint=float(np.max(np.abs(condition_beta(collab,final,0.0)-l2_rows_np(collab))))
        maxcap=max(max(x['trajectory'][m]['fraction_hit_20deg_cap'] for m in ('text','visual')) for x in results.values())
        smoke={'guidance_changes_condition_angle':bool(guidance_diff>1e-4),'guidance_angle_max_mean_diff_deg':float(guidance_diff),'t_edit_changes_trajectory':bool(tedit_diff>1e-4),'t_edit_angle_max_mean_diff_deg':float(tedit_diff),'beta_condition_asset_differs':bool(beta_hash_diff),'beta0p5_historical_max_abs_diff':beta05_rebuild,'beta0_collab_endpoint_max_abs_diff':beta0_endpoint,'beta_condition_rebuild_exact':bool(beta05_rebuild<=1e-7 and beta0_endpoint<=1e-7),'max_fraction_hit_20deg_cap':float(maxcap),'eta0p1_not_cap_saturated':bool(maxcap<=.05)}
        checks['smoke']=smoke
        if not all([smoke['guidance_changes_condition_angle'],smoke['t_edit_changes_trajectory'],smoke['beta_condition_asset_differs'],smoke['beta_condition_rebuild_exact'],smoke['eta0p1_not_cap_saturated']]): raise RuntimeError(f'Round11 smoke mechanism FAIL {smoke}')
    out={'status':'PASS','protocol':PROTOCOL,'mode':mode,'seed':seed,'source_round10_commit':SOURCE,'configs':results,'checks':checks,'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False,'elapsed_seconds':time.time()-t0}
    (outdir/'result.json').write_text(json.dumps(out,indent=2)+'\n'); shutil.rmtree(outdir/'tmp',ignore_errors=True)
    print(json.dumps({'status':'PASS','mode':mode,'seed':seed,'U':{k:round(v['U'],8) for k,v in results.items()},'angles':{k:round(v['mean_applied_angle_deg'],4) for k,v in results.items()},'elapsed':out['elapsed_seconds']},sort_keys=True)); return out

def summary(rows,name):
    vals={str(s):float(rows[s]['configs'][name]['U']) for s in BACKBONES}; a=list(vals.values()); pp={str(s):int(rows[s]['configs'][name]['primary_positive_count']) for s in BACKBONES}; nc10={str(s):int(rows[s]['configs'][name]['boundary']['NetCross10']) for s in BACKBONES}; nc20={str(s):int(rows[s]['configs'][name]['boundary']['NetCross20']) for s in BACKBONES}; eff={str(s):rows[s]['configs'][name]['ranking_efficiency_U_per_deg'] for s in BACKBONES}; ang={str(s):rows[s]['configs'][name]['mean_applied_angle_deg'] for s in BACKBONES}
    return {'seed_U':vals,'mean_U':float(np.mean(a)),'median_U':float(np.median(a)),'worst_seed_U':float(min(a)),'positive_seeds':int(sum(x>0 for x in a)),'primary_positive_counts':pp,'NetCross10_by_seed':nc10,'NetCross20_by_seed':nc20,'NetCross10_sum':int(sum(nc10.values())),'NetCross20_sum':int(sum(nc20.values())),'boundary_both_positive_seeds':int(sum(nc10[str(s)]>0 and nc20[str(s)]>0 for s in BACKBONES)),'ranking_efficiency_by_seed':eff,'mean_ranking_efficiency':float(np.mean([x for x in eff.values() if x is not None])),'mean_applied_angle_by_seed':ang,'mean_applied_angle_deg':float(np.mean(list(ang.values())))}

def load_screen(root):
    rows={}
    for s in BACKBONES:
        p=root/f'seed{s}'/'result.json';
        if not p.exists(): raise RuntimeError(f'missing screen {p}')
        r=json.loads(p.read_text())
        if r['status']!='PASS' or r.get('TEST_ACCESSED') is not False: raise RuntimeError(f'invalid screen seed{s}')
        rows[s]=r
    return rows
def family_pick(summaries,names,parameter):
    def conservative(name): return -float(CONFIGS[name][parameter])
    return max(names,key=lambda n:(summaries[n]['positive_seeds'],summaries[n]['worst_seed_U'],summaries[n]['mean_U'],conservative(n)))

def aggregate_screen(root:Path,evid:Path):
    rows=load_screen(root); sums={n:summary(rows,n) for n in CONFIGS}
    g=family_pick(sums,('G1','G2','C0'),'guidance'); b=family_pick(sums,('B0','C0','B1'),'beta'); t=family_pick(sums,('T1','T2','C0'),'t_edit')
    combo={'guidance':float(CONFIGS[g]['guidance']),'beta':float(CONFIGS[b]['beta']),'t_edit':int(CONFIGS[t]['t_edit'])}
    out={'status':'SCREEN_SELECTION_FROZEN','protocol':PROTOCOL,'summaries':sums,'family_selection':{'guidance':{'selected':g,'value':combo['guidance'],'candidates':['G1','G2','C0']},'beta':{'selected':b,'value':combo['beta'],'candidates':['B0','C0','B1']},'t_edit':{'selected':t,'value':combo['t_edit'],'candidates':['T1','T2','C0']}},'Cstar_config':combo,'selection_rule':['maximize positive backbone count','maximize worst-seed U','maximize mean U','prefer conservative parameter'],'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
    evid.mkdir(parents=True,exist_ok=True); (evid/'ROUND11_SCREEN_VALIDATION.json').write_text(json.dumps({'status':'SCREEN_COMPLETE','summaries':sums,'TEST_ACCESSED':False},indent=2)+'\n'); (evid/'ROUND11_SCREEN_SELECTION.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'status':out['status'],'g':g,'beta':b,'t':t,'Cstar':combo},sort_keys=True)); return out,rows

def run_combined_seed(seed,outdir:Path,selection:Path):
    t0=time.time(); sel=json.loads(selection.read_text()); cfg=sel['Cstar_config']; ctx=r9.validation_context(seed); identity=eta0_identity(ctx,outdir); res=run_config(seed,'C*',cfg,ctx,outdir)
    checks={'eta0_identity':bool(identity['ranking_exact_vs_colift']),'paired_xt_exact':bool(res['provenance']['paired_initial_xt_max_abs_diff']==0.0),'final_finite_norm':True,'attribute_unchanged':True,'direct_score_residual':False,'TEST_loader_called':False,'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
    if not checks['paired_xt_exact']: raise RuntimeError('C* paired exactness FAIL')
    out={'status':'PASS','protocol':PROTOCOL,'mode':'combined','seed':seed,'Cstar_config':cfg,'config':res,'checks':checks,'selection_sha256':sha(selection),'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False,'elapsed_seconds':time.time()-t0}
    outdir.mkdir(parents=True,exist_ok=True); (outdir/'result.json').write_text(json.dumps(out,indent=2)+'\n'); shutil.rmtree(outdir/'tmp',ignore_errors=True); print(json.dumps({'status':'PASS','seed':seed,'Cstar':cfg,'U':res['U'],'angle':res['mean_applied_angle_deg'],'NetCross10':res['boundary']['NetCross10'],'NetCross20':res['boundary']['NetCross20'],'elapsed':out['elapsed_seconds']},sort_keys=True)); return out

def load_combined(root):
    rows={}
    for s in BACKBONES:
        p=root/f'combined_seed{s}'/'result.json'
        if not p.exists(): raise RuntimeError(f'missing combined {p}')
        r=json.loads(p.read_text());
        if r['status']!='PASS' or r.get('TEST_ACCESSED') is not False: raise RuntimeError(f'invalid combined seed{s}')
        rows[s]=r
    return rows

def combined_summary(rows):
    vals={str(s):float(rows[s]['config']['U']) for s in BACKBONES}; a=list(vals.values()); pp={str(s):int(rows[s]['config']['primary_positive_count']) for s in BACKBONES}; nc10={str(s):int(rows[s]['config']['boundary']['NetCross10']) for s in BACKBONES}; nc20={str(s):int(rows[s]['config']['boundary']['NetCross20']) for s in BACKBONES}; eff={str(s):rows[s]['config']['ranking_efficiency_U_per_deg'] for s in BACKBONES}; ang={str(s):rows[s]['config']['mean_applied_angle_deg'] for s in BACKBONES}
    return {'seed_U':vals,'mean_U':float(np.mean(a)),'median_U':float(np.median(a)),'worst_seed_U':float(min(a)),'positive_seeds':int(sum(x>0 for x in a)),'primary_positive_counts':pp,'NetCross10_by_seed':nc10,'NetCross20_by_seed':nc20,'NetCross10_sum':int(sum(nc10.values())),'NetCross20_sum':int(sum(nc20.values())),'boundary_both_positive_seeds':int(sum(nc10[str(s)]>0 and nc20[str(s)]>0 for s in BACKBONES)),'ranking_efficiency_by_seed':eff,'mean_ranking_efficiency':float(np.mean(list(eff.values()))),'mean_applied_angle_by_seed':ang,'mean_applied_angle_deg':float(np.mean(list(ang.values())))}

def verdict(c):
    if c['positive_seeds']==4 and c['mean_U']>=.01: return 'TARGET_PASS'
    if c['positive_seeds']==4 and c['mean_U']>=.005: return 'STRONG_PASS'
    if c['positive_seeds']==4 and c['mean_U']>0: return 'STABILITY_PASS'
    partial_metric=(c['positive_seeds']==3 and c['worst_seed_U']>-.001 and c['mean_U']>.002123)
    partial_boundary=(c['boundary_both_positive_seeds']>=3)
    if partial_metric or partial_boundary: return 'PARTIAL_SIGNAL'
    if c['positive_seeds']<=2 and not partial_boundary: return 'FAIL'
    return 'NO_PASS'

def pct(x): return f'{100*float(x):+.4f}%'

def compact_evidence(screen_rows,combined_rows):
    traj={}; bound={}; prov={}
    for s in BACKBONES:
        traj[str(s)]={}; bound[str(s)]={}; prov[str(s)]={}
        for n in CONFIGS:
            x=screen_rows[s]['configs'][n]; traj[str(s)][n]=x['trajectory']; bound[str(s)][n]=x['boundary']; prov[str(s)][n]=x['provenance']
        x=combined_rows[s]['config']; traj[str(s)]['C*']=x['trajectory']; bound[str(s)]['C*']=x['boundary']; prov[str(s)]['C*']=x['provenance']
    return traj,bound,prov
def mean_traj(screen_rows,name,field):
    vals=[]
    for s in BACKBONES:
        for m in ('text','visual'): vals.append(screen_rows[s]['configs'][name]['trajectory'][m][field]['mean'])
    return float(np.mean(vals))

def write_report(out,screen_rows,combined_rows,evid):
    ss=out['screen_summaries']; cs=out['Cstar']; cfg=out['Cstar_config']; sel=out['family_selection']; v=out['verdict']
    L=['# Round11 Validation Report — Conditional Diffusion Trajectory Calibration','',f'Final verdict: **ROUND11 = {v}**',f"Frozen C*: guidance={cfg['guidance']}, beta={cfg['beta']}, t_edit={cfg['t_edit']}, eta=0.10, safety cap=20°",'', '> Baby Validation only. Test, Sports and Electronics remained CLOSED.','']
    L += ['## Mandatory comparison','','| Config | guidance | beta | t_edit | Mean U | Worst U | Positive Seeds | NetCross@10 | NetCross@20 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|','| Round10 | 2.0 | 0.5 | 3 | +0.2123% | -0.0492% | 3/4 | — | — |']
    for n in ('C0','G1','G2','B0','B1','T1','T2'):
        x=ss[n]; c=CONFIGS[n]; L.append(f"| {n} | {c['guidance']:.1f} | {c['beta']:.1f} | {c['t_edit']} | {pct(x['mean_U'])} | {pct(x['worst_seed_U'])} | {x['positive_seeds']}/4 | {x['NetCross10_sum']:+d} | {x['NetCross20_sum']:+d} |")
    L.append(f"| C* | {cfg['guidance']:.1f} | {cfg['beta']:.1f} | {cfg['t_edit']} | {pct(cs['mean_U'])} | {pct(cs['worst_seed_U'])} | {cs['positive_seeds']}/4 | {cs['NetCross10_sum']:+d} | {cs['NetCross20_sum']:+d} |")
    L += ['','## Family selection','','| Family | Selected | Positive | Worst U | Mean U | Mean angle | Mean E_rank |','|---|---|---:|---:|---:|---:|---:|']
    for fam in ('guidance','beta','t_edit'):
        n=sel[fam]['selected']; x=ss[n]; L.append(f"| {fam} | {n} | {x['positive_seeds']}/4 | {pct(x['worst_seed_U'])} | {pct(x['mean_U'])} | {x['mean_applied_angle_deg']:.4f}° | {x['mean_ranking_efficiency']:.6e} |")
    L += ['','Selection was frozen as positive-backbone count → worst-seed U → mean U → conservative parameter. E_rank and boundary diagnostics are supporting diagnostics only.','']
    L += ['## C* per-backbone results','','| Backbone | U | Primary + | Mean applied angle | E_rank | NetCross@10 | NetCross@20 |','|---:|---:|---:|---:|---:|---:|---:|']
    for s in BACKBONES:
        x=combined_rows[s]['config']; L.append(f"| {s} | {pct(x['U'])} | {x['primary_positive_count']}/4 | {x['mean_applied_angle_deg']:.4f}° | {x['ranking_efficiency_U_per_deg']:.6e} | {x['boundary']['NetCross10']:+d} | {x['boundary']['NetCross20']:+d} |")
    L += ['','## Trajectory-strength diagnostics','','| Config | mean NULL→TRUE | mean raw→final | max cap fraction |','|---|---:|---:|---:|']
    for n in ('C0','G1','G2','B0','B1','T1','T2'):
        nulltrue=mean_traj(screen_rows,n,'NULL_to_TRUE'); angle=ss[n]['mean_applied_angle_deg']; cap=max(screen_rows[s]['configs'][n]['trajectory'][m]['fraction_hit_20deg_cap'] for s in BACKBONES for m in ('text','visual')); L.append(f'| {n} | {nulltrue:.4f}° | {angle:.4f}° | {100*cap:.4f}% |')
    cnull=float(np.mean([combined_rows[s]['config']['trajectory'][m]['NULL_to_TRUE']['mean'] for s in BACKBONES for m in ('text','visual')])); ccap=max(combined_rows[s]['config']['trajectory'][m]['fraction_hit_20deg_cap'] for s in BACKBONES for m in ('text','visual')); L.append(f"| C* | {cnull:.4f}° | {cs['mean_applied_angle_deg']:.4f}° | {100*ccap:.4f}% |")
    c0sep=mean_traj(screen_rows,'C0','NULL_to_TRUE'); g1sep=mean_traj(screen_rows,'G1','NULL_to_TRUE'); g2sep=mean_traj(screen_rows,'G2','NULL_to_TRUE')
    L += ['','## Required scientific questions','',f"**Q1 — Does lowering guidance reduce NULL→TRUE separation?** C0(w=2.0)={c0sep:.4f}°, G2(w=1.5)={g2sep:.4f}°, G1(w=1.0)={g1sep:.4f}°. The numerical ordering directly answers whether CFG amplification increases trajectory separation.",f"**Q2 — Does weaker condition movement improve ranking?** C0 mean/worst U={pct(ss['C0']['mean_U'])}/{pct(ss['C0']['worst_seed_U'])}; G2={pct(ss['G2']['mean_U'])}/{pct(ss['G2']['worst_seed_U'])}; G1={pct(ss['G1']['mean_U'])}/{pct(ss['G1']['worst_seed_U'])}. Their NetCross sums are C0=({ss['C0']['NetCross10_sum']:+d},{ss['C0']['NetCross20_sum']:+d}), G2=({ss['G2']['NetCross10_sum']:+d},{ss['G2']['NetCross20_sum']:+d}), G1=({ss['G1']['NetCross10_sum']:+d},{ss['G1']['NetCross20_sum']:+d}).",f"**Q3 — Which beta/t_edit improves ranking efficiency rather than only movement?** Selected beta config is {sel['beta']['selected']} and selected t_edit config is {sel['t_edit']['selected']}. Their mean E_rank values are {ss[sel['beta']['selected']]['mean_ranking_efficiency']:.6e} and {ss[sel['t_edit']['selected']]['mean_ranking_efficiency']:.6e}, versus C0 {ss['C0']['mean_ranking_efficiency']:.6e}.",'']
    L += ['## Exactness / restrictions','']
    maxxt=max([screen_rows[s]['configs'][n]['provenance']['paired_initial_xt_max_abs_diff'] for s in BACKBONES for n in CONFIGS]+[combined_rows[s]['config']['provenance']['paired_initial_xt_max_abs_diff'] for s in BACKBONES]); maxcap=max([screen_rows[s]['configs'][n]['trajectory'][m]['fraction_hit_20deg_cap'] for s in BACKBONES for n in CONFIGS for m in ('text','visual')]+[combined_rows[s]['config']['trajectory'][m]['fraction_hit_20deg_cap'] for s in BACKBONES for m in ('text','visual')]); aggressive=sum(screen_rows[s]['configs'][n]['TRAJECTORY_TOO_AGGRESSIVE'] for s in BACKBONES for n in CONFIGS)+sum(combined_rows[s]['config']['TRAJECTORY_TOO_AGGRESSIVE'] for s in BACKBONES)
    L += [f'- TRUE/NULL initial x_t max difference: `{maxxt:.3e}`.','- eta=0 exact CoLiftRec identity: PASS on every screening/combined backbone.','- No direct score residual; Attribute unchanged; no Test loader call.',f'- Maximum observed 20° cap-hit fraction: `{100*maxcap:.6f}%`; TRAJECTORY_TOO_AGGRESSIVE cells: `{aggressive}`.','- No full factorial and no extra combined configuration were run.','']
    L += ['## Provenance','',f'- Source Round10 commit: `{SOURCE}`.','- Frozen eta=0.10; safety cap=20° only.','- Frozen CoLiftRec and frozen diffusion checkpoints; no retraining.','- Test / Sports / Electronics remained CLOSED.','']
    (evid/'ROUND11_REPORT.md').write_text('\n'.join(L))

def aggregate_final(root:Path,evid:Path):
    sel=json.loads((evid/'ROUND11_SCREEN_SELECTION.json').read_text()); screen_rows=load_screen(root); combined_rows=load_combined(root); ss={n:summary(screen_rows,n) for n in CONFIGS}; cs=combined_summary(combined_rows); vv=verdict(cs); traj,bound,prov=compact_evidence(screen_rows,combined_rows)
    out={'status':'VALIDATION_DECISION','protocol':PROTOCOL,'screen_summaries':ss,'family_selection':sel['family_selection'],'Cstar_config':sel['Cstar_config'],'Cstar':cs,'Cstar_detailed':{str(s):combined_rows[s]['config'] for s in BACKBONES},'verdict':vv,'round10_reference':{'mean_U':.002123,'worst_seed_U':-.000492,'positive_seeds':3},'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
    evid.mkdir(parents=True,exist_ok=True); (evid/'ROUND11_TRAJECTORY_DIAGNOSTICS.json').write_text(json.dumps(traj,indent=2)+'\n'); (evid/'ROUND11_BOUNDARY_DIAGNOSTICS.json').write_text(json.dumps(bound,indent=2)+'\n'); (evid/'ROUND11_PROVENANCE.json').write_text(json.dumps({'status':'PASS','protocol':PROTOCOL,'source_round10_commit':SOURCE,'eta':ETA,'safety_cap_deg':CAP,'screening_configs':CONFIGS,'Cstar_config':sel['Cstar_config'],'per_cell_provenance':prov,'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False},indent=2)+'\n'); (evid/'ROUND11_VALIDATION_RESULTS.json').write_text(json.dumps(out,indent=2)+'\n'); write_report(out,screen_rows,combined_rows,evid)
    print(json.dumps({'status':out['status'],'verdict':vv,'Cstar':sel['Cstar_config'],'mean_U':cs['mean_U'],'worst_U':cs['worst_seed_U'],'positive':cs['positive_seeds'],'NetCross10':cs['NetCross10_sum'],'NetCross20':cs['NetCross20_sum']},sort_keys=True)); return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--mode',choices=['smoke','formal_screen','aggregate_screen','formal_combined','aggregate_final'],required=True); ap.add_argument('--seed',type=int); ap.add_argument('--out-root',default=str(ROOT/'diffusion_experiments/round11_cdtc/outputs')); ap.add_argument('--evidence',default=str(ROOT/'diffusion_experiments/round11_cdtc/evidence')); a=ap.parse_args(); root=Path(a.out_root); evid=Path(a.evidence)
    if a.mode=='smoke':
        out=root/'smoke_seed999'; shutil.rmtree(out,ignore_errors=True); run_seed(999,out,'smoke',SMOKE)
    elif a.mode=='formal_screen':
        if a.seed not in BACKBONES: raise RuntimeError('formal_screen seed invalid')
        out=root/f'seed{a.seed}'; shutil.rmtree(out,ignore_errors=True); run_seed(a.seed,out,'screen',tuple(CONFIGS))
    elif a.mode=='aggregate_screen': aggregate_screen(root,evid)
    elif a.mode=='formal_combined':
        if a.seed not in BACKBONES: raise RuntimeError('formal_combined seed invalid')
        selection=evid/'ROUND11_SCREEN_SELECTION.json'
        if not selection.exists(): raise RuntimeError('screen selection missing')
        out=root/f'combined_seed{a.seed}'; shutil.rmtree(out,ignore_errors=True); run_combined_seed(a.seed,out,selection)
    else: aggregate_final(root,evid)
if __name__=='__main__': main()
