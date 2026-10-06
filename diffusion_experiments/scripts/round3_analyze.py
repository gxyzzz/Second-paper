from __future__ import annotations
import argparse,csv,hashlib,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch,yaml
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'src'))
from diffusion_experiments.scripts import round3_train as T
from diffusion_experiments.models.round3_candidate_energy import DenoiserCore,ConditionalDenoiser,DirectEvidenceScorer,probe_specs,apply_evidence,evidence_advantage
from modules.ranking import metrics_at,metric_arrays,sha256_file
PRIMARY=('R10','N10','R20','N20'); ALL=PRIMARY+('R50','N50')
def sha(p): return sha256_file(Path(p))
def load_json(p): return json.loads(Path(p).read_text())
def per_user_metrics(items,users,labels):
    _,_,hit,pos_len=metric_arrays(items,users,labels,max_k=20)
    ranks=np.arange(1,21,dtype=np.float64)[None,:]
    recall=np.cumsum(hit,1)/pos_len[:,None]
    dcg=np.cumsum(hit/np.log2(ranks+1),1)
    idcg=np.cumsum(np.ones((len(users),20))/np.log2(ranks+1),1)
    for i,n in enumerate(pos_len):
        cut=min(int(n),20)
        if cut<20:idcg[i,cut:]=idcg[i,cut-1]
    nd=dcg/idcg
    return np.stack([recall[:,9],nd[:,9],recall[:,19],nd[:,19]],1)
def U_from_arrays(base,new):
    b=base.mean(0); n=new.mean(0); return float(np.mean((n-b)/np.maximum(b,1e-12)))
def bootstrap_compare(base,a,b=None,resamples=1000,seed=202610069):
    rng=np.random.default_rng(seed); vals=[]; n=len(base)
    for _ in range(resamples):
        ix=rng.integers(0,n,n); ua=U_from_arrays(base[ix],a[ix]); ub=0.0 if b is None else U_from_arrays(base[ix],b[ix]); vals.append(ua-ub)
    vals=np.asarray(vals); point=U_from_arrays(base,a)-(0.0 if b is None else U_from_arrays(base,b))
    return {'delta_U':point,'ci95':[float(np.quantile(vals,.025)),float(np.quantile(vals,.975))],'positive_replicate_fraction':float((vals>0).mean()),'resamples':int(resamples),'seed':int(seed),'note':'conditional paired user bootstrap; positive fraction is not p-value/probability'}
def hit_transitions(base,new,users,labels,k):
    inc=dec=z2h=h2z=0
    for i,u in enumerate(users):
        lab=labels[int(u)]; b=sum(int(x) in lab for x in base[i,:k]); n=sum(int(x) in lab for x in new[i,:k]); inc+=n>b; dec+=n<b; z2h+=(b==0 and n>0); h2z+=(b>0 and n==0)
    return {'increase':int(inc),'decrease':int(dec),'zero_to_hit':int(z2h),'hit_to_zero':int(h2z),'net_users':int(inc-dec)}
def auc_pairs(pos,neg,mask):
    d=pos[:,None]-neg; v=d[mask];
    return {'pairs':int(len(v)),'mean_margin':float(v.mean()) if len(v) else None,'win_rate_auc':float(((v>0).sum()+.5*(v==0).sum())/len(v)) if len(v) else None}
def direct_evidence_context(model,data,latent,ctx,device,batch=512):
    out=[]; model.eval()
    with torch.no_grad():
      for st in range(0,len(data['users']),batch):
        en=min(st+batch,len(data['users'])); ids=data['items'][st:en,5:30]; z=torch.as_tensor(latent[ids],device=device); c=torch.as_tensor(ctx[st:en],device=device); B,K,D=z.shape
        out.append(model(z.reshape(B*K,D),c[:,None,:].expand(-1,K,-1).reshape(B*K,-1)).reshape(B,K).cpu().numpy())
    return np.concatenate(out).astype(np.float32)

def load_best(run,kind,latent_dim,context_dim,cfg,device,bg_dirs):
    mc=cfg['model']; ck=torch.load(run/'best.pt',map_location=device,weights_only=False)
    if kind=='C':
        model=DirectEvidenceScorer(latent_dim,context_dim,int(mc['hidden_dim']),float(mc['dropout'])).to(device); bg=None; scale=None
    else:
        bg,bgr=T.load_bg(bg_dirs['AE' if kind=='AE_PREF' else 'DM'],latent_dim,int(mc['hidden_dim']),int(mc['time_dim']),float(mc['dropout']),device)
        scale=float(bgr['scale_bg']); model=ConditionalDenoiser(latent_dim,context_dim,int(mc['hidden_dim']),int(mc['time_dim']),float(mc['dropout'])).to(device)
    model.load_state_dict(ck['model']); model.eval(); return model,bg,scale

def condition_evidence(kind,model,bg,data,latent,dims,scale,cfg,manifest,bundle,device,ctx):
    if kind=='C': return direct_evidence_context(model,data,latent,ctx,device)
    return T.evidence_D(model,bg,data,latent,dims,scale,cfg,manifest,bundle,'AE' if kind=='AE_PREF' else 'DM',device,context_override=ctx)[0]

def build_internal_pairs(data,targets,fit,mon,seed):
    fmap={int(u):set(g.itemID.astype(int).tolist()) for u,g in fit.groupby('userID')}; mmap={int(u):set(g.itemID.astype(int).tolist()) for u,g in mon.groupby('userID')}
    n=len(data['users']); negpos=np.full((n,3),-1,np.int16); mask=np.zeros((n,3),bool); sources=np.full((n,3),'',dtype='U16'); pospos=np.full(n,-1,np.int16)
    for i,(u,row,t) in enumerate(zip(data['users'],data['items'],targets)):
        p=np.flatnonzero(row==int(t));
        if not len(p): continue
        pospos[i]=int(p[0]); known=fmap.get(int(u),set())|mmap.get(int(u),set())|{int(t)}; legal=[j for j,it in enumerate(row) if int(it) not in known]
        rng=np.random.default_rng(int(seed)+int(u)*1000003); chosen=[]
        for k,(name,lo,hi) in enumerate([('boundary10',7,13),('boundary20',17,23),('ordinary',0,100)]):
            pool=[j for j in legal if lo<=j<hi and j not in chosen]
            src=name
            if not pool: pool=[j for j in legal if j not in chosen]; src=name+'_fallback'
            if not pool: break
            j=int(rng.choice(np.asarray(pool))); chosen.append(j); negpos[i,k]=j; mask[i,k]=True; sources[i,k]=src
    return pospos,negpos,mask,sources

def pair_evidence_items(kind,model,bg,data,latent,pospos,negpos,mask,dims,scale,cfg,manifest,bundle,device,ctx):
    valid=pospos>=0; rows=np.flatnonzero(valid); items=np.full((len(rows),4),0,np.int32); users=data["users"][rows]; c=ctx[rows]
    for q,i in enumerate(rows):
        items[q,0]=data["items"][i,pospos[i]]
        for k in range(3):
            items[q,k+1]=data["items"][i,negpos[i,k]] if mask[i,k] else items[q,0]
    z=torch.as_tensor(latent[items],device=device); ct=torch.as_tensor(c,device=device)
    if kind=="C":
        B,K,D=z.shape
        with torch.no_grad(): ev=model(z.reshape(B*K,D),ct[:,None,:].expand(-1,K,-1).reshape(B*K,-1)).reshape(B,K).cpu().numpy()
    else:
        mode="AE" if kind=="AE_PREF" else "DM"; specs=probe_specs(mode,manifest["schedule"],manifest["ae_t_index_zero_based"])
        with torch.no_grad(): ev=evidence_advantage(model,bg,z,ct,users,dims,scale,cfg["dataset"],bundle,specs,int(cfg["noise"]["diffusion_steps"]))[0].cpu().numpy()
    pm=mask[rows].copy(); return ev[:,0],ev[:,1:],pm,rows

def selected_metrics(data,evidence,eta,cfg):
    ranked,delta=apply_evidence(data['items'],data['s0'],evidence,float(eta),float(cfg['formal']['clip_c']),T.boundary_gate(data['s0']))
    return T.result_vs(metrics_at(data['items'],data['users'],data['labels']),metrics_at(ranked,data['users'],data['labels'])),ranked,delta

def diagnostic_one(kind,run,cfg,manifest,latent,context,dims,frozen,fit,mon,device,bg_dirs):
    r=load_json(run/'result.json'); eta=float(r['best']['eta']); model,bg,scale=load_best(run,kind,latent.shape[1],context.shape[1],cfg,device,bg_dirs)
    internal=T.make_dataset('internal',frozen,context); n=len(internal['users']); der=np.roll(np.arange(n),1); zero=np.zeros_like(internal['context']); shuf=internal['context'][der]
    bundle=int(cfg['probe_bundle_main']); true=condition_evidence(kind,model,bg,internal,latent,dims,scale,cfg,manifest,bundle,device,internal['context']); null=condition_evidence(kind,model,bg,internal,latent,dims,scale,cfg,manifest,bundle,device,zero); shuffled=condition_evidence(kind,model,bg,internal,latent,dims,scale,cfg,manifest,bundle,device,shuf)
    pospos,negpos,mask,sources=build_internal_pairs(internal,internal['targets'],fit,mon,int(cfg['supervision_seed']))
    diag={}
    pair_cache={}
    for name,ev,ctxv in [('TRUE',true,internal['context']),('NULL',null,zero),('SHUFFLED',shuffled,shuf)]:
        p,nv,pm,rows=pair_evidence_items(kind,model,bg,internal,latent,pospos,negpos,mask,dims,scale,cfg,manifest,bundle,device,ctxv); pair_cache[name]=(p,nv,pm,rows); d=auc_pairs(p,nv,pm); met,ranked,_=selected_metrics(internal,ev,eta,cfg); d['ranking_at_frozen_eta']=met; d['probe_hit_queries']=int(len(rows)); diag[name]=d
    p,nv,pm,rows=pair_cache['TRUE']; strata={}
    src_rows=sources[rows]
    for src in ['boundary10','boundary20','ordinary']:
        sm=(src_rows==src)&pm; strata[src]=auc_pairs(p,nv,sm)
    # S0-gap and semantic/collaborative disagreement diagnostics on the same natural pairs.
    pz=np.load(frozen/'probe_top100.npz'); tz=np.load(frozen/'probe_targets.npz'); imask=tz['internal'].astype(bool); s0all=pz['s0'][imask]; feats=pz['features'][imask]; fn=[str(x) for x in pz['feature_names'].tolist()]; didx=fn.index('cos_disagreement')
    pair_gap=np.full_like(nv,np.nan,dtype=np.float32); pair_dis=np.full_like(nv,np.nan,dtype=np.float32)
    for q,i in enumerate(rows):
        pp=int(pospos[i]);
        for k in range(3):
            if not mask[i,k]: continue
            np0=int(negpos[i,k]); pair_gap[q,k]=abs(float(s0all[i,pp])-float(s0all[i,np0])); pair_dis[q,k]=abs(float(feats[i,pp,didx])-float(feats[i,np0,didx]))
    gapdiag={'le_0p5':auc_pairs(p,nv,pm&(pair_gap<=0.5)),'gt_0p5':auc_pairs(p,nv,pm&(pair_gap>0.5)),'pair_gap_mean':float(np.nanmean(pair_gap[pm])),'pair_gap_median':float(np.nanmedian(pair_gap[pm]))}
    disvals=pair_dis[pm]; med=float(np.nanmedian(disvals)); disagreement={'mean_abs_pair_disagreement':float(np.nanmean(disvals)),'median_abs_pair_disagreement':med,'low_or_equal_median':auc_pairs(p,nv,pm&(pair_dis<=med)),'high_median':auc_pairs(p,nv,pm&(pair_dis>med))}
    # Per-probe readings on both DEV and INTERNAL; fixed eta, no probe selection.
    per_probe={'internal':[],'dev':[]}
    if kind!='C':
        mode='AE' if kind=='AE_PREF' else 'DM'; specs=probe_specs(mode,manifest['schedule'],manifest['ae_t_index_zero_based']); dev=T.make_dataset('dev',frozen,context)
        for spec in specs:
            for split,data in [('internal',internal),('dev',dev)]:
                ev=T.evidence_D(model,bg,data,latent,dims,scale,cfg,manifest,bundle,mode,device,spec_override=[spec])[0]; met,_,_=selected_metrics(data,ev,eta,cfg); per_probe[split].append({'spec':list(spec),'U':met['U']})
    if kind!='C':
        for c in diag.values(): c['raw_mean_margin']=float(c['mean_margin']*scale) if c['mean_margin'] is not None else None; c['scale_bg']=float(scale)
    return {'eta':eta,'condition':diag,'strata':strata,'gap_strata':gapdiag,'semantic_collab_disagreement':disagreement,'per_probe':per_probe}

def load_checkpoint_model(run,kind,epoch,latent_dim,context_dim,cfg,device,bg_dirs):
    mc=cfg['model']; ck=torch.load(run/f'checkpoints/epoch_{epoch:02d}.pt',map_location=device,weights_only=False)
    if kind=='C': model=DirectEvidenceScorer(latent_dim,context_dim,int(mc['hidden_dim']),float(mc['dropout'])).to(device); bg=None; scale=None
    else:
        bg,bgr=T.load_bg(bg_dirs['AE' if kind=='AE_PREF' else 'DM'],latent_dim,int(mc['hidden_dim']),int(mc['time_dim']),float(mc['dropout']),device); scale=float(bgr['scale_bg']); model=ConditionalDenoiser(latent_dim,context_dim,int(mc['hidden_dim']),int(mc['time_dim']),float(mc['dropout'])).to(device)
    model.load_state_dict(ck['model']); model.eval(); return model,bg,scale

def fixed_epoch_pair(seed,runroot,cfg,manifest,latent,context,dims,frozen,device,bg_dirs):
    a=runroot/f'D_GEN_seed{seed}'; b=runroot/f'D_PREF_seed{seed}'; ha=load_json(a/'history.json'); hb=load_json(b/'history.json')
    ea={int(x['epoch']) for x in ha if 'dev' in x}; eb={int(x['epoch']) for x in hb if 'dev' in x}; common=max(ea&eb)
    out={'epoch':common,'eta':0.10}
    for kind,run in [('D_GEN',a),('D_PREF',b)]:
        model,bg,scale=load_checkpoint_model(run,kind,common,latent.shape[1],context.shape[1],cfg,device,bg_dirs)
        rec={}
        for split in ['dev','internal']:
            data=T.make_dataset(split,frozen,context); ev=condition_evidence(kind,model,bg,data,latent,dims,scale,cfg,manifest,int(cfg['probe_bundle_main']),device,data['context']); met,_,_=selected_metrics(data,ev,.10,cfg); rec[split]=met
        out[kind]=rec
    return out

def aggregate_run_result(r):
    return {'kind':r['kind'],'seed':r['seed'],'best_epoch':r['best']['epoch'],'eta':r['best']['eta'],'DEV_U':r['selected_dev_main']['U'],'INTERNAL_U':r['selected_internal_main']['U'],'TRAIN_U':r['selected_train_main']['U'],'TRAIN856_U':r['selected_train_window856']['U'],'INTERNAL189_U':r['selected_internal_window189']['U'],'secondary_DEV_U':r['secondary_bundle_at_frozen_eta']['dev']['U'],'secondary_INTERNAL_U':r['secondary_bundle_at_frozen_eta']['internal']['U'],'g1_DEV_U':r['g1_control_at_frozen_eta']['U'],'parameter_count':r['parameter_count'],'optimizer_steps':r['optimizer_steps'],'epochs_completed':r['epochs_completed'],'train_seconds':r['train_seconds'],'peak_cuda_memory_bytes':r['peak_cuda_memory_bytes'],'main_bundle_dev_energy_seconds':r['main_bundle_dev_energy_seconds'],'background_hash_match':r.get('background_hash_before')==r.get('background_hash_after') if r['kind']!='C' else True}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run-root',default='diffusion_experiments/runs/round3'); ap.add_argument('--out',default='diffusion_experiments/runs/round3/analysis'); a=ap.parse_args()
    runroot=ROOT/a.run_root; out=ROOT/a.out; out.mkdir(parents=True,exist_ok=True); cfg=yaml.safe_load((ROOT/'diffusion_experiments/configs/round3_baby.yaml').read_text()); asset=runroot/'assets_formal'; manifest=load_json(asset/'manifest.json'); frozen=ROOT/cfg['frozen_assets_dir']; pdx=ROOT/cfg['protocol_dir']
    lc=np.load(asset/'latent_context.npz'); latent=lc['item_latent'].astype(np.float32); context=lc['user_context'].astype(np.float32); dims=lc['block_dims'].astype(int).tolist(); fit=pd.read_csv(pdx/'fit_edges.csv'); mon=pd.read_csv(pdx/'monitor_edges.csv')
    bg_dirs={'DM':runroot/'formal_BG_DM_r1','AE':runroot/'formal_BG_AE'}; device=torch.device('cuda:0')
    if not torch.cuda.is_available() or '5090' not in torch.cuda.get_device_name(0): raise RuntimeError('RTX5090 required')
    runs={}
    for kind in ['C','D_GEN','D_PREF','AE_PREF']:
        for seed in cfg['conditional_seeds']:
            name=f'{kind}_seed{seed}'; p=runroot/name
            r=load_json(p/'result.json')
            if r['status']!='COMPLETE' or r['access']['CONFIRM_ACCESSED'] or r['access']['TEST_ACCESSED']: raise RuntimeError(f'invalid run {name}')
            runs[name]=r
    rows=[aggregate_run_result(runs[k]) for k in sorted(runs)]
    aggregate={}
    for kind in ['C','D_GEN','D_PREF','AE_PREF']:
        rr=[x for x in rows if x['kind']==kind]; aggregate[kind]={f'mean_{f}':float(np.mean([x[f] for x in rr])) for f in ['DEV_U','INTERNAL_U','TRAIN_U','TRAIN856_U','INTERNAL189_U','secondary_DEV_U','secondary_INTERNAL_U']}; aggregate[kind]['DEV_U_by_seed']=[x['DEV_U'] for x in rr]; aggregate[kind]['INTERNAL_U_by_seed']=[x['INTERNAL_U'] for x in rr]
    # ranking transition + per-user arrays for bootstrap
    transitions={}; per={}
    for name,r in runs.items():
        p=np.load(runroot/name/'predictions.npz'); rec={}
        for split in ['dev','internal']:
            users=p[f'{split}_users']; base=p[f'{split}_base']; new=p[f'{split}_new']; labels=T.dev_labels(ROOT/'data/baby/baby.inter',users) if split=='dev' else T.probe_labels(users,p['internal_targets'])
            rec[split]={'K10':hit_transitions(base,new,users,labels,10),'K20':hit_transitions(base,new,users,labels,20)}; per[(name,split,'base')]=per_user_metrics(base,users,labels); per[(name,split,'new')]=per_user_metrics(new,users,labels)
        transitions[name]=rec
    bootstrap={}
    bs=int(cfg['bootstrap_resamples']); bseed=int(cfg['bootstrap_seed'])
    for seed in cfg['conditional_seeds']:
        bootstrap[str(seed)]={}
        for split in ['dev','internal']:
            pref=f'D_PREF_seed{seed}'; base=per[(pref,split,'base')]; pa=per[(pref,split,'new')]; bootstrap[str(seed)][split]={'D_PREF_vs_baseline':bootstrap_compare(base,pa,None,bs,bseed)}
            for other in ['C','D_GEN','AE_PREF']:
                ob=per[(f'{other}_seed{seed}',split,'new')]; bootstrap[str(seed)][split][f'D_PREF_vs_{other}']=bootstrap_compare(base,pa,ob,bs,bseed)
    # condition necessity and probe reliability
    diagnostics={}
    for name in sorted(runs): diagnostics[name]=diagnostic_one(runs[name]['kind'],runroot/name,cfg,manifest,latent,context,dims,frozen,fit,mon,device,bg_dirs)
    common={str(seed):fixed_epoch_pair(seed,runroot,cfg,manifest,latent,context,dims,frozen,device,bg_dirs) for seed in cfg['conditional_seeds']}
    summary={'status':'COMPLETE','implementation_commit':runs[next(iter(runs))]['git_sha'],'asset_manifest_sha256':sha(asset/'manifest.json'),'backgrounds':{'BG_DM':load_json(bg_dirs['DM']/'result.json'),'BG_AE':load_json(bg_dirs['AE']/'result.json')},'rows':rows,'aggregate':aggregate,'transitions':transitions,'bootstrap':bootstrap,'condition_diagnostics':diagnostics,'common_epoch_eta010':common,'oracle_dev':manifest['oracle_dev'],'historical_exposure_note':'DEV/INTERNAL and old full Validation/Test are development evidence; not fresh external confirmation.','access':{'CONFIRM_ACCESSED':False,'TEST_ACCESSED':False}}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with open(out/'results.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(json.dumps({'status':'COMPLETE','aggregate':aggregate},sort_keys=True))
if __name__=='__main__': main()
