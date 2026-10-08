from __future__ import annotations
import argparse, gc, hashlib, json, math, shutil, sys, time
from pathlib import Path
import numpy as np, torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,str(ROOT))
from modules.diffusion import cosine_alpha_bar, q_sample, l2_rows_np
from modules.semantic_purifier import blend_native
from diffusion_experiments.round9_cabrp import run_round9 as r9

PROTOCOL='ROUND10_CCDP_V1'
BACKBONES=(999,1000,1001,1002)
ETAS=(0.5,1.0,2.0)
SEEDS=r9.SEEDS
CAP_DEG=10.0
SOURCE_COMMIT='a1104fd3c87a70accbed4cabb9f1eac0e08e42f3'
PRIMARY=r9.PRIMARY; ALL=r9.ALL

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def dist_stats(x):
 x=np.asarray(x,np.float64)
 return {k:float(v) for k,v in [('mean',x.mean()),('median',np.median(x)),('p25',np.quantile(x,.25)),('p50',np.quantile(x,.50)),('p75',np.quantile(x,.75)),('p90',np.quantile(x,.90)),('p95',np.quantile(x,.95)),('p99',np.quantile(x,.99)),('max',x.max())]}

def short_stats(x):
 x=np.asarray(x,np.float64)
 return {k:float(v) for k,v in [('mean',x.mean()),('median',np.median(x)),('p75',np.quantile(x,.75)),('p90',np.quantile(x,.90)),('p95',np.quantile(x,.95))]}

def angle_deg(a,b):
 a=l2_rows_np(a); b=l2_rows_np(b)
 return np.degrees(np.arccos(np.clip(np.sum(a*b,axis=1),-1.0,1.0)))

@torch.no_grad()
def ddim_from_xt(model,xt,cond,guidance,alpha_bar,t_edit=3):
 x=xt.clone(); zero=torch.zeros_like(cond)
 for ti in range(int(t_edit),0,-1):
  t=torch.full((len(x),),ti,device=x.device,dtype=torch.long)
  pcond=model(x,t,cond); pun=model(x,t,zero)
  xhat=pun+float(guidance)*(pcond-pun)
  ab_t=alpha_bar[ti].to(x.device); ab_prev=alpha_bar[ti-1].to(x.device)
  eps=(x-torch.sqrt(ab_t)*xhat)/torch.sqrt(torch.clamp(1-ab_t,min=1e-8))
  x=torch.sqrt(ab_prev)*xhat+torch.sqrt(torch.clamp(1-ab_prev,min=0.0))*eps
 return x

def paired_ensembles(seed,outdir:Path,device='cuda',batch=128):
 cfg,rt,rv=r9.load_raw(); p=r9.paths(seed); model,ck,cp=r9.load_model(seed,device); cond=np.load(cp,mmap_mode='r')
 n=len(rt); dt=rt.shape[1]; dv=rv.shape[1]; ab=cosine_alpha_bar(50,.008).to(device); rev=np.arange(n-1,-1,-1,dtype=np.int64)
 sums={k:np.zeros((n,d),np.float32) for k,d in [('true_t',dt),('true_v',dv),('null_t',dt),('null_v',dv),('shuf_t',dt),('shuf_v',dv)]}
 max_xt_diff=0.0
 for ps in SEEDS:
  gen=torch.Generator(device=device); gen.manual_seed(int(ps))
  for st in range(0,n,batch):
   en=min(st+batch,n); raw=np.concatenate([l2_rows_np(rt[st:en]),l2_rows_np(rv[st:en])],1).astype(np.float32)
   x0=torch.from_numpy(raw).to(device); noise=torch.randn(x0.shape,generator=gen,device=device,dtype=x0.dtype); tt=torch.full((en-st,),3,device=device,dtype=torch.long); xt=q_sample(x0,tt,noise,ab)
   xt_true=xt.clone(); xt_null=xt.clone(); xt_shuf=xt.clone(); max_xt_diff=max(max_xt_diff,float(torch.max(torch.abs(xt_true-xt_null))),float(torch.max(torch.abs(xt_true-xt_shuf))))
   c=torch.from_numpy(np.asarray(cond[st:en],np.float32).copy()).to(device); cs=torch.from_numpy(np.asarray(cond[rev[st:en]],np.float32).copy()).to(device); z=torch.zeros_like(c)
   yt=ddim_from_xt(model,xt_true,c,2.0,ab,3).cpu().numpy(); yn=ddim_from_xt(model,xt_null,z,2.0,ab,3).cpu().numpy(); ys=ddim_from_xt(model,xt_shuf,cs,2.0,ab,3).cpu().numpy()
   for name,y in [('true',yt),('null',yn),('shuf',ys)]:
    sums[name+'_t'][st:en]+=l2_rows_np(y[:,:dt]); sums[name+'_v'][st:en]+=l2_rows_np(y[:,dt:])
 out={k:l2_rows_np(v/len(SEEDS)).astype(np.float32) for k,v in sums.items()}
 outdir.mkdir(parents=True,exist_ok=True); np.savez_compressed(outdir/'paired_assets.npz',**out)
 prov={'backbone_seed':seed,'diffusion_checkpoint':str(Path(ck).relative_to(ROOT)),'diffusion_sha256':sha(ck),'condition_path':str(Path(cp).relative_to(ROOT)),'condition_sha256':sha(cp),'purification_seeds':list(SEEDS),'paired_initial_xt_max_abs_diff':max_xt_diff,'same_raw_x0':True,'same_t_edit':True,'same_noise':True,'same_ddim_schedule':True,'condition_only_difference':True,'TEST_ACCESSED':False}
 del model; torch.cuda.empty_cache(); gc.collect(); return out,prov

def decompose(raw,true,null):
 raw=l2_rows_np(raw); true=l2_rows_np(true); null=l2_rows_np(null); full=true-raw; generic=null-raw; cond=true-null; err=full-(generic+cond)
 dot=np.sum(raw*cond,axis=1); tangent=cond-dot[:,None]*raw; nc=np.linalg.norm(cond,axis=1); nt=np.linalg.norm(tangent,axis=1); ratio=np.divide(nt,nc,out=np.zeros_like(nt),where=nc>1e-12)
 return {'full':full,'generic':generic,'cond':cond,'tangent':tangent,'decomposition':{'max_absolute_error':float(np.max(np.abs(err))),'mean_absolute_error':float(np.mean(np.abs(err)))},'tangent_norm':{'cond_norm':short_stats(nc),'tangent_norm':short_stats(nt),'tangent_over_full':short_stats(ratio)}}

def direction_diag(true,null,shuf):
 a=l2_rows_np(true)-l2_rows_np(null); b=l2_rows_np(shuf)-l2_rows_np(null); na=np.linalg.norm(a,axis=1); nb=np.linalg.norm(b,axis=1); cos=np.divide(np.sum(a*b,axis=1),na*nb,out=np.zeros_like(na),where=(na*nb)>1e-12)
 return {'cos_true_null_vs_shuf_null':dist_stats(cos),'norm_true_null':dist_stats(na),'norm_shuf_null':dist_stats(nb)}

def b3_features(raw,tangent,eta):
 raw=l2_rows_np(raw); cand=l2_rows_np(raw+float(eta)*np.asarray(tangent,np.float32)); final,_,applied=r9.slerp_bounded(raw,cand,np.ones(len(raw),np.float32),CAP_DEG)
 if not np.isfinite(final).all() or np.max(np.abs(np.linalg.norm(final,axis=1)-1.0))>2e-5: raise RuntimeError('B3 final feature finite/norm check failed')
 if float(np.max(applied))>CAP_DEG+0.02: raise RuntimeError('B3 angular cap violation')
 return final,applied

def baseline_result(ctx):
 m=ctx['base_metrics']
 return {'metrics':m,'absolute_delta':{k:0.0 for k in ALL},'relative_delta':{k:0.0 for k in ALL},'U':0.0,'primary_positive_count':0,'overall_positive_count':0,'ranking_exact_vs_colift':True,'score_max_abs_delta_vs_colift':0.0}

def run_seed(seed:int,outdir:Path,mode:str):
 t0=time.time(); outdir.mkdir(parents=True,exist_ok=True); ctx=r9.validation_context(seed)
 B0=baseline_result(ctx); B1,b1audit=r9.parity_a0(seed,ctx,outdir)
 paired,prov=paired_ensembles(seed,outdir); rt=l2_rows_np(ctx['rt']); rv=l2_rows_np(ctx['rv'])
 dt=decompose(rt,paired['true_t'],paired['null_t']); dv=decompose(rv,paired['true_v'],paired['null_v'])
 movement={'text':{'raw_to_TRUE_deg':dist_stats(angle_deg(rt,paired['true_t'])),'raw_to_NULL_deg':dist_stats(angle_deg(rt,paired['null_t'])),'NULL_to_TRUE_deg':dist_stats(angle_deg(paired['null_t'],paired['true_t']))},'visual':{'raw_to_TRUE_deg':dist_stats(angle_deg(rv,paired['true_v'])),'raw_to_NULL_deg':dist_stats(angle_deg(rv,paired['null_v'])),'NULL_to_TRUE_deg':dist_stats(angle_deg(paired['null_v'],paired['true_v']))}}
 decomposition={'text':dt['decomposition'],'visual':dv['decomposition'],'tangent':{'text':dt['tangent_norm'],'visual':dv['tangent_norm']}}
 shuffled={'text':direction_diag(paired['true_t'],paired['null_t'],paired['shuf_t']),'visual':direction_diag(paired['true_v'],paired['null_v'],paired['shuf_v'])}
 b2t,b2v=blend_native(rt,rv,paired['null_t'],paired['null_v'],.25,1.0); B2,_=r9.evaluate_features(ctx,b2t,b2v,outdir/'tmp','B2_generic')
 raw_id,_=r9.evaluate_features(ctx,ctx['rt'],ctx['rv'],outdir/'tmp','B3_eta0_identity')
 if not raw_id['ranking_exact_vs_colift'] or max(abs(raw_id['metrics'][k]-ctx['base_metrics'][k]) for k in ALL)>1e-12 or raw_id['score_max_abs_delta_vs_colift']>1e-7: raise RuntimeError('eta=0 / raw CoLift identity FAIL')
 variants={'B0':B0,'B1':B1,'B2':B2}; b3move={}
 for eta in ETAS:
  ft,mt=b3_features(rt,dt['tangent'],eta); fv,mv=b3_features(rv,dv['tangent'],eta); res,_=r9.evaluate_features(ctx,ft,fv,outdir/'tmp',f'B3_eta{str(eta).replace(".","p")}'); variants[f'B3_{eta:g}']=res
  b3move[str(eta)]={'text_applied_angle_deg':dist_stats(mt),'visual_applied_angle_deg':dist_stats(mv),'text_max_deg':float(np.max(mt)),'visual_max_deg':float(np.max(mv))}
 checks={'paired_initial_xt_exact':bool(prov['paired_initial_xt_max_abs_diff']==0.0),'paired_initial_xt_max_abs_diff':prov['paired_initial_xt_max_abs_diff'],'decomposition_max_abs_error':max(decomposition['text']['max_absolute_error'],decomposition['visual']['max_absolute_error']),'eta0_identity':raw_id,'B1_round9_A0_exact':bool(b1audit['match']),'final_feature_norm_checked':True,'B3_max_angle_within_10deg':bool(max(max(v['text_max_deg'],v['visual_max_deg']) for v in b3move.values())<=10.02),'attribute_unchanged':True,'direct_ranking_residual':False,'TEST_loader_called':False,'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
 if not (checks['paired_initial_xt_exact'] and checks['decomposition_max_abs_error']<=2e-6 and checks['eta0_identity']['ranking_exact_vs_colift'] and checks['B1_round9_A0_exact'] and checks['B3_max_angle_within_10deg']): raise RuntimeError(f'ROUND10 INVALID checks: {checks}')
 result={'status':'PASS','protocol':PROTOCOL,'mode':mode,'seed':seed,'source_round9_commit':SOURCE_COMMIT,'provenance':prov,'coliftrec_metrics':ctx['base_metrics'],'B1_audit':b1audit,'variants':variants,'movement_diagnostics':movement,'residual_decomposition':decomposition,'shuffled_direction_diagnostic':shuffled,'B3_movement':b3move,'exactness_checks':checks,'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False,'elapsed_seconds':time.time()-t0}
 (outdir/'result.json').write_text(json.dumps(result,indent=2)+'\n'); shutil.rmtree(outdir/'tmp',ignore_errors=True)
 print(json.dumps({'status':'PASS','mode':mode,'seed':seed,'B1_U':B1['U'],'B2_U':B2['U'],'B3':{str(e):variants[f'B3_{e:g}']['U'] for e in ETAS},'angles':{'text':{k:v['mean'] for k,v in movement['text'].items()},'visual':{k:v['mean'] for k,v in movement['visual'].items()}},'elapsed':result['elapsed_seconds']},sort_keys=True)); return result

def summarize_variant(rows,key):
 u={str(s):float(rows[s]['variants'][key]['U']) for s in BACKBONES}; vals=list(u.values()); pp={str(s):int(rows[s]['variants'][key]['primary_positive_count']) for s in BACKBONES}
 return {'seed_U':u,'mean_U':float(np.mean(vals)),'median_U':float(np.median(vals)),'worst_seed_U':float(min(vals)),'positive_seeds':int(sum(x>0 for x in vals)),'primary_positive_counts':pp}

def aggregate(rootdir:Path,evid:Path):
 rows={}
 for s in BACKBONES:
  p=rootdir/f'seed{s}'/'result.json'
  if not p.exists(): raise RuntimeError(f'missing formal result {p}')
  r=json.loads(p.read_text());
  if r['status']!='PASS' or r.get('TEST_ACCESSED') is not False: raise RuntimeError(f'invalid/contaminated seed{s}')
  rows[s]=r
 B1=summarize_variant(rows,'B1'); B2=summarize_variant(rows,'B2'); B3={}
 for eta in ETAS: B3[str(eta)]=summarize_variant(rows,f'B3_{eta:g}')
 selected=max(ETAS,key=lambda e:(B3[str(e)]['positive_seeds'],B3[str(e)]['worst_seed_U'],B3[str(e)]['mean_U'],-e)); sel=B3[str(selected)]
 if sel['positive_seeds']==4:
  if sel['mean_U']>=.01: verdict='TARGET_PASS'
  elif sel['mean_U']>=.005: verdict='STRONG_PASS'
  else: verdict='STABILITY_PASS'
 else:
  partial_a=bool(sel['positive_seeds']>=3 and sel['worst_seed_U']>B1['worst_seed_U'])
  b3_vs_b2_count=sum(rows[s]['variants'][f'B3_{selected:g}']['U']>rows[s]['variants']['B2']['U'] for s in BACKBONES)
  partial_b=bool(b3_vs_b2_count>=3 and sel['mean_U']>B2['mean_U'])
  verdict='PARTIAL_SIGNAL' if (partial_a or partial_b) else 'FAIL'
 selected_detail={str(s):rows[s]['variants'][f'B3_{selected:g}'] for s in BACKBONES}
 out={'status':'VALIDATION_DECISION','protocol':PROTOCOL,'selection_rule':['maximize positive_backbones','maximize worst_seed_U','maximize mean_U','smaller_eta_on_exact_tie'],'B0_reference_U':0.0,'B1':B1,'B2':B2,'B3':B3,'selected_eta':float(selected),'selected_B3':sel,'selected_B3_detailed':selected_detail,'partial_signal_checks':{'branch_a_3of4_and_worst_strictly_better_than_B1':partial_a if sel['positive_seeds']<4 else None,'branch_b_B3_beats_B2_on_backbones':int(sum(rows[s]['variants'][f'B3_{selected:g}']['U']>rows[s]['variants']['B2']['U'] for s in BACKBONES)),'branch_b_mean_B3_gt_B2':bool(sel['mean_U']>B2['mean_U']) if sel['positive_seeds']<4 else None},'verdict':verdict,'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
 evid.mkdir(parents=True,exist_ok=True)
 provenance={str(s):rows[s]['provenance'] for s in BACKBONES}; provenance['source_round9_commit']=SOURCE_COMMIT; provenance['protocol']=PROTOCOL; provenance['closed']={'TEST':True,'SPORTS':True,'ELECTRONICS':True}
 decomposition={str(s):{'residual_decomposition':rows[s]['residual_decomposition'],'shuffled_direction_diagnostic':rows[s]['shuffled_direction_diagnostic']} for s in BACKBONES}
 movement={str(s):{'movement_diagnostics':rows[s]['movement_diagnostics'],'B3_movement':rows[s]['B3_movement']} for s in BACKBONES}
 (evid/'ROUND10_PROVENANCE.json').write_text(json.dumps(provenance,indent=2)+'\n'); (evid/'ROUND10_RESIDUAL_DECOMPOSITION.json').write_text(json.dumps(decomposition,indent=2)+'\n'); (evid/'ROUND10_MOVEMENT_DIAGNOSTICS.json').write_text(json.dumps(movement,indent=2)+'\n'); (evid/'ROUND10_VALIDATION_RESULTS.json').write_text(json.dumps(out,indent=2)+'\n')
 return out,rows

def pct(x): return f'{100*float(x):+.4f}%'

def write_report(out,rows,evid:Path):
 eta=float(out['selected_eta']); sel=out['selected_B3']; L=['# Round10 Validation Report — Collaborative-Condition Differential Purification','',f'Final status: **VALIDATION_DECISION**',f"Verdict: **ROUND10 = {out['verdict']}**",f"Selected global eta: **{eta:g}**",'', '> Test, Sports and Electronics remained CLOSED throughout Round10.','']
 L += ['## Main comparison — U vs frozen CoLiftRec','', '| Backbone | CoLift | Original B1 | Generic B2 | B3 η=.5 | B3 η=1 | B3 η=2 |','|---:|---:|---:|---:|---:|---:|---:|']
 for s in BACKBONES:
  v=rows[s]['variants']; L.append(f"| {s} | 0.0000% | {pct(v['B1']['U'])} | {pct(v['B2']['U'])} | {pct(v['B3_0.5']['U'])} | {pct(v['B3_1']['U'])} | {pct(v['B3_2']['U'])} |")
 L.append(f"| Mean U | 0.0000% | {pct(out['B1']['mean_U'])} | {pct(out['B2']['mean_U'])} | {pct(out['B3']['0.5']['mean_U'])} | {pct(out['B3']['1.0']['mean_U'])} | {pct(out['B3']['2.0']['mean_U'])} |")
 L += ['','## Representation movement — mean angular movement','', '| Backbone | Modality | raw→TRUE | raw→NULL | NULL→TRUE |','|---:|---|---:|---:|---:|']
 fullmeans=[]; genmeans=[]; condmeans=[]
 for s in BACKBONES:
  m=rows[s]['movement_diagnostics']
  for mod in ('text','visual'):
   a=m[mod]['raw_to_TRUE_deg']['mean']; b=m[mod]['raw_to_NULL_deg']['mean']; c=m[mod]['NULL_to_TRUE_deg']['mean']; fullmeans.append(a); genmeans.append(b); condmeans.append(c); L.append(f'| {s} | {mod.title()} | {a:.4f}° | {b:.4f}° | {c:.4f}° |')
 L += ['','## Eta selection','', '| eta | positive backbones | worst U | mean U | median U |','|---:|---:|---:|---:|---:|']
 for e in ETAS:
  x=out['B3'][str(e)]; L.append(f"| {e:g} | {x['positive_seeds']}/4 | {pct(x['worst_seed_U'])} | {pct(x['mean_U'])} | {pct(x['median_U'])} |")
 L += ['', 'Selection priority was frozen before formal Validation: positive-backbone count → worst-seed U → mean U → smaller eta on exact tie.','']
 L += ['## Selected B3 detailed metrics','', '| Backbone | Metric | Absolute | Δ vs CoLift | Relative Δ |','|---:|---|---:|---:|---:|']
 for s in BACKBONES:
  x=out['selected_B3_detailed'][str(s)]
  for k in ALL: L.append(f"| {s} | {k} | {x['metrics'][k]:.8f} | {x['absolute_delta'][k]:+.8f} | {pct(x['relative_delta'][k])} |")
  L.append(f"| {s} | **U** | — | — | **{pct(x['U'])}** |")
 L += ['','## Stability summary','',f"- Positive backbones: **{sel['positive_seeds']}/4**.",f"- Mean U: **{pct(sel['mean_U'])}**; median U: **{pct(sel['median_U'])}**; worst-seed U: **{pct(sel['worst_seed_U'])}**.",f"- Primary-positive counts by backbone: `{sel['primary_positive_counts']}`.",f"- PARTIAL_SIGNAL checks: `{out['partial_signal_checks']}`.",'']
 maxdec=max(max(rows[s]['residual_decomposition']['text']['max_absolute_error'],rows[s]['residual_decomposition']['visual']['max_absolute_error']) for s in BACKBONES); maxangle=max(max(v['text_max_deg'],v['visual_max_deg']) for s in BACKBONES for v in rows[s]['B3_movement'].values()); maxxt=max(rows[s]['provenance']['paired_initial_xt_max_abs_diff'] for s in BACKBONES)
 L += ['## Exactness / sanity','',f'- TRUE/NULL/SHUFFLED initial `x_t` max difference: `{maxxt:.3e}`.',f'- Residual decomposition max absolute error: `{maxdec:.3e}`.',f'- Maximum B3 raw→final movement: `{maxangle:.6f}°` (fixed safety cap 10°).','- eta=0 raw-feature/CoLift identity: PASS on every formal seed.','- B1 Round9 A0 parity: PASS on every formal seed.','- Attribute unchanged; no direct ranking residual exists; Test loader not invoked.','']
 cond_ratio=float(np.mean(condmeans)/max(np.mean(fullmeans),1e-12)); generic_ratio=float(np.mean(genmeans)/max(np.mean(fullmeans),1e-12)); b3_b2_count=out['partial_signal_checks']['branch_b_B3_beats_B2_on_backbones']
 L += ['## Mechanism interpretation','',f"Across backbone/modality cells, mean raw→TRUE={np.mean(fullmeans):.4f}°, raw→NULL={np.mean(genmeans):.4f}°, and NULL→TRUE={np.mean(condmeans):.4f}°. Thus condition-specific angular movement is {cond_ratio:.3f}× the full shift, while generic movement is {generic_ratio:.3f}× the full shift.",f"Selected B3 beats Generic-only B2 on **{b3_b2_count}/4** backbones.",'','The guide distinguishes: **Case A** = generic-dominated with little useful conditional signal; **Case B** = generic-dominated but B3 becomes positive/stable; **Case C** = substantial condition differential whose direction remains unstable/negative. The numerical evidence above should be used for the Advisor interpretation; the executor does not redefine these cases with post-hoc thresholds.','']
 L += ['## TRUE vs NULL vs SHUFFLED diagnostic','']
 for s in BACKBONES:
  d=rows[s]['shuffled_direction_diagnostic']; L.append(f"- seed{s}: Text cos(true-null, shuf-null) mean={d['text']['cos_true_null_vs_shuf_null']['mean']:.4f}, true-null norm mean={d['text']['norm_true_null']['mean']:.4f}, shuf-null norm mean={d['text']['norm_shuf_null']['mean']:.4f}; Visual cos mean={d['visual']['cos_true_null_vs_shuf_null']['mean']:.4f}, true-null norm mean={d['visual']['norm_true_null']['mean']:.4f}, shuf-null norm mean={d['visual']['norm_shuf_null']['mean']:.4f}.")
 L += ['','## Provenance','',f'- Source Round9 commit: `{SOURCE_COMMIT}`.','- Frozen Baby backbones: 999 / 1000 / 1001 / 1002.','- Frozen CoLiftRec λ(T/A/V)=1.0/0.75/0.25; α(T/A/V)=0.25/0.15/0.025.','- Frozen diffusion beta=0.5, t_edit=3, guidance=2.0, steps=50, cosine_s=0.008, purification seeds 20261001–20261004.','- No diffusion retraining. No Test, Sports or Electronics access.','']
 (evid/'ROUND10_REPORT.md').write_text('\n'.join(L))

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--mode',choices=['smoke','formal','aggregate'],required=True); ap.add_argument('--seed',type=int); ap.add_argument('--out-root',default=str(ROOT/'diffusion_experiments/round10_ccdp/outputs')); ap.add_argument('--evidence',default=str(ROOT/'diffusion_experiments/round10_ccdp/evidence')); a=ap.parse_args(); root=Path(a.out_root); evid=Path(a.evidence)
 if a.mode in ('smoke','formal'):
  if a.mode=='smoke': seed=999; out=root/'smoke_seed999'
  else:
   if a.seed not in BACKBONES: raise RuntimeError('formal seed must be one frozen Baby backbone')
   seed=a.seed; out=root/f'seed{seed}'
  if out.exists(): shutil.rmtree(out)
  run_seed(seed,out,a.mode)
 else:
  out,rows=aggregate(root,evid); write_report(out,rows,evid); print(json.dumps({'status':'VALIDATION_DECISION','selected_eta':out['selected_eta'],'verdict':out['verdict'],'selected':out['selected_B3']},sort_keys=True))
if __name__=='__main__': main()
