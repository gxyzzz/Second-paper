from __future__ import annotations
import argparse, hashlib, json, math, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch, yaml

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'src'))
from diffusion_experiments.models.round1_residual import (
    DeterministicResidual, BoundaryResidualDiffusion, cosine_alpha_bars,
    project_zero_mean, deploy_scores, ddim_sample,
)
from diffusion_experiments.models.round2_residual import (
    masked_query_mse, masked_pairwise_rank_loss, ddim_from_noise_differentiable,
)
from modules.ranking import metrics_at

PRIMARY=('R10','N10','R20','N20'); ALL=('R10','N10','R20','N20','R50','N50')

def sha(p:Path):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def dev_labels(path,users):
    keep=set(map(int,users)); out={int(u):set() for u in users}
    for c in pd.read_csv(path,sep='\t',usecols=['userID','itemID','x_label'],chunksize=50000):
        z=c[(c.x_label==1)&c.userID.isin(keep)]
        for u,g in z.groupby('userID'): out[int(u)].update(g.itemID.astype(int).tolist())
    return out

def probe_labels(users,targets): return {int(u):{int(t)} for u,t in zip(users,targets)}

def result_vs(base,new):
    rel={k:(float(new[k])-float(base[k]))/max(float(base[k]),1e-12) for k in PRIMARY}
    return {'metrics':{k:float(new[k]) for k in ALL},'relative_primary':rel,'U':float(np.mean(list(rel.values()))),'delta_R50':float(new['R50']-base['R50']),'delta_N50':float(new['N50']-base['N50'])}

def protected(res,cfg):
    rel=res['relative_primary']; nonneg=sum(v>=0 for v in rel.values())
    return bool(nonneg>=int(cfg['min_nonnegative_primary']) and min(rel.values())>=-float(cfg['max_primary_relative_regression']) and res['delta_R50']>=float(cfg['min_absolute_delta_R50']) and res['delta_N50']>=float(cfg['min_absolute_delta_N50']))

def rerank(items,s0w,x0,sigma,eta,c,tau):
    if float(eta)==0: return items.copy()
    sc=deploy_scores(torch.as_tensor(s0w),torch.as_tensor(x0),sigma,eta,c,tau).numpy()
    order=np.argsort(-sc,axis=1,kind='stable'); out=items.copy(); out[:,5:30]=np.take_along_axis(items[:,5:30],order,axis=1); return out

def batches(idx,batch,rng):
    p=rng.permutation(idx)
    for s in range(0,len(p),batch): yield p[s:s+batch]

def grad_norm(loss,params):
    gs=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
    return float(math.sqrt(sum(float((g.detach()**2).sum()) for g in gs if g is not None)))

def finite_tensor(name,x):
    if not torch.isfinite(x).all(): raise FloatingPointError(f'nonfinite {name}')

def target_ranks(items,targets):
    out=np.zeros(len(items),np.int16)
    for i,(row,t) in enumerate(zip(items,targets)):
        p=np.flatnonzero(row==int(t)); out[i]=int(p[0])+1 if len(p) else 0
    return out

def rank_summary(before,after):
    d=after.astype(np.int32)-before.astype(np.int32)
    valid=(before>0)&(after>0)
    return {'queries':int(len(before)),'valid_both':int(valid.sum()),'improved':int((d[valid]<0).sum()),'worsened':int((d[valid]>0).sum()),'unchanged':int((d[valid]==0).sum()),'mean_delta_after_minus_before':float(d[valid].mean()) if valid.any() else None,'top10_corrected':int(((before>10)&(after>0)&(after<=10)).sum()),'top10_broken':int(((before>0)&(before<=10)&((after>10)|(after==0))).sum()),'top20_corrected':int(((before>20)&(after>0)&(after<=20)).sum()),'top20_broken':int(((before>0)&(before<=20)&((after>20)|(after==0))).sum())}

def predict_paths(model,variant,features,users,sampling_seeds,mc,device,batch_eval=512):
    if variant=='C':
        out=[]
        with torch.no_grad():
            for st in range(0,len(users),batch_eval):
                f=torch.as_tensor(features[st:st+batch_eval],device=device); out.append(model(f).cpu().numpy())
        x=np.concatenate(out)
        return {'five_avg':x,'one_avg':x,'five_first':x,'one_first':x,'five_per_seed':{'deterministic':x},'one_per_seed':{'deterministic':x}}
    five={int(s):[] for s in sampling_seeds}; one={int(s):[] for s in sampling_seeds}
    with torch.no_grad():
        for st in range(0,len(users),batch_eval):
            en=min(st+batch_eval,len(users)); f=torch.as_tensor(features[st:en],device=device); u=torch.as_tensor(users[st:en],device=device)
            for s in sampling_seeds:
                five[int(s)].append(ddim_sample(model,f,u,1.0,int(s),int(mc['ddim_steps']),int(mc['diffusion_steps']),device).cpu().numpy())
                one[int(s)].append(ddim_sample(model,f,u,1.0,int(s),1,int(mc['diffusion_steps']),device).cpu().numpy())
    five={s:np.concatenate(v) for s,v in five.items()}; one={s:np.concatenate(v) for s,v in one.items()}
    # sigma is deliberately not used inside sampler; residual is standardized x0. Keep API payload independent.
    return {'five_avg':np.mean(np.stack(list(five.values())),0),'one_avg':np.mean(np.stack(list(one.values())),0),'five_first':five[int(sampling_seeds[0])],'one_first':one[int(sampling_seeds[0])],'five_per_seed':five,'one_per_seed':one}

def eval_dataset(model,variant,data,labels,sigma,cfg,device):
    mc=cfg['model']; rc=cfg['residual']; pc=cfg['protection']; seeds=cfg['formal']['sampling_seeds']
    pred=predict_paths(model,variant,data['features'],data['users'],seeds,mc,device)
    base=metrics_at(data['items'],data['users'],labels)
    def eval_x(x):
        d={}
        for eta in rc['eta_candidates']:
            ranked=rerank(data['items'],data['s0w'],x,sigma,float(eta),float(rc['clip_c']),float(rc['tau']))
            r=result_vs(base,metrics_at(ranked,data['users'],labels)); r['protected']=protected(r,pc); d[str(eta)]=r
        return d
    five=eval_x(pred['five_avg']); one=eval_x(pred['one_avg']); first5=eval_x(pred['five_first']); first1=eval_x(pred['one_first'])
    samples5={str(s):eval_x(x) for s,x in pred['five_per_seed'].items()}; samples1={str(s):eval_x(x) for s,x in pred['one_per_seed'].items()}
    return {'baseline':{k:float(base[k]) for k in ALL},'five_step_four_sample_etas':five,'one_step_four_sample_etas':one,'five_step_first_sample_etas':first5,'one_step_first_sample_etas':first1,'five_step_per_sample_etas':samples5,'one_step_per_sample_etas':samples1},pred

def choose_dev(dev_eval):
    cand=[(float(k),v) for k,v in dev_eval['five_step_four_sample_etas'].items() if v['protected']]
    return max(cand,key=lambda z:z[1]['U']) if cand else (0.0,dev_eval['five_step_four_sample_etas']['0.0'])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--supervision-dir',required=True); ap.add_argument('--variant',choices=['C','D0','D1'],required=True); ap.add_argument('--seed',type=int,required=True); ap.add_argument('--out',required=True); ap.add_argument('--mode',choices=['smoke','formal'],default='formal')
    a=ap.parse_args(); out=Path(a.out)
    if out.exists() and any((out/x).exists() for x in ['best.pt','history.json','result.json']): raise RuntimeError('refuse overwrite nonempty result directory')
    out.mkdir(parents=True,exist_ok=True)
    cfg=yaml.safe_load((ROOT/'diffusion_experiments/configs/round2_baby.yaml').read_text()); supdir=Path(a.supervision_dir); manifest=json.loads((supdir/'manifest.json').read_text())
    if manifest['access']['CONFIRM_ACCESSED'] or manifest['access']['TEST_ACCESSED']: raise RuntimeError('closed-set invariant')
    if sha(supdir/'supervision.npz')!=manifest['supervision_sha256']: raise RuntimeError('supervision hash mismatch')
    if sha(ROOT/'diffusion_experiments/configs/round2_baby.yaml')!=manifest['config_sha256']: raise RuntimeError('config identity mismatch')
    if not torch.cuda.is_available() or '5090' not in torch.cuda.get_device_name(0): raise RuntimeError('RTX5090 required')
    device=torch.device('cuda:0'); torch.manual_seed(a.seed); np.random.seed(a.seed); torch.cuda.reset_peak_memory_stats(device)
    assets=ROOT/cfg['round1_assets_dir']
    probe=np.load(assets/'probe_top100.npz'); tgt=np.load(assets/'probe_targets.npz'); devz=np.load(assets/'dev_top100.npz'); sz=np.load(supdir/'supervision.npz')
    mean=sz['feature_mean'].astype(np.float32); std=sz['feature_std'].astype(np.float32); sigma=float(sz['sigma_r'])
    sup_rows=sz['sup_rows'].astype(np.int64); clean=sz['x0'].astype(np.float32); active=sz['active_mask'].astype(bool); target_pos=sz['target_pos'].astype(np.int64); affected=sz['affected'].astype(bool)
    feat_all=((probe['features'][:,5:30,:].astype(np.float32)-mean)/std).astype(np.float32)
    feat=feat_all[sup_rows]; s0w=probe['s0'][sup_rows,5:30].astype(np.float32); sup_items=probe['items'][sup_rows].astype(np.int32); sup_users=probe['users'][sup_rows].astype(np.int64); sup_targets=tgt['target_items'][sup_rows].astype(np.int32)
    train_local=np.arange(len(sup_rows)); smoke_seed=2026100604
    if a.mode=='smoke':
        train_local=np.sort(np.random.default_rng(smoke_seed).choice(train_local,size=min(256,len(train_local)),replace=False))
    rr_internal=tgt['internal'].astype(bool); irows=np.flatnonzero(rr_internal)
    iusers=probe['users'][irows].astype(np.int64); iitems=probe['items'][irows].astype(np.int32); itargets=tgt['target_items'][irows].astype(np.int32); ifeat=feat_all[irows]; is0w=probe['s0'][irows,5:30].astype(np.float32)
    dusers=devz['users'].astype(np.int64); ditems=devz['items'].astype(np.int32); dfeat=((devz['features'][:,5:30,:].astype(np.float32)-mean)/std).astype(np.float32); ds0w=devz['s0'][:,5:30].astype(np.float32)
    if a.mode=='smoke':
        rng=np.random.default_rng(smoke_seed)
        di=np.sort(rng.choice(len(dusers),size=min(512,len(dusers)),replace=False)); ii=np.sort(rng.choice(len(iusers),size=min(512,len(iusers)),replace=False))
        dusers,ditems,dfeat,ds0w=dusers[di],ditems[di],dfeat[di],ds0w[di]
        iusers,iitems,ifeat,is0w,itargets=iusers[ii],iitems[ii],ifeat[ii],is0w[ii],itargets[ii]
    devsets=dev_labels(ROOT/'data/baby/baby.inter',dusers); intsets=probe_labels(iusers,itargets); trainsets=probe_labels(sup_users,sup_targets)
    devdata={'users':dusers,'items':ditems,'features':dfeat,'s0w':ds0w}
    intdata={'users':iusers,'items':iitems,'features':ifeat,'s0w':is0w}
    traindata={'users':sup_users,'items':sup_items,'features':feat,'s0w':s0w}
    mc=cfg['model']; cls=DeterministicResidual if a.variant=='C' else BoundaryResidualDiffusion
    model=cls(feat.shape[-1],int(mc['hidden_dim']),int(mc['attention_heads']),int(mc['attention_layers']),float(mc['dropout'])).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=float(mc['learning_rate']),weight_decay=float(mc['weight_decay']))
    nparams=sum(p.numel() for p in model.parameters())
    max_epochs=2 if a.mode=='smoke' else int(cfg['formal']['max_epochs']); min_epoch=1 if a.mode=='smoke' else int(cfg['formal']['min_epochs']); eval_every=1 if a.mode=='smoke' else int(cfg['formal']['eval_every']); patience=int(cfg['formal']['patience_evals']); batch=int(mc['batch_size'])
    alpha=cosine_alpha_bars(int(mc['diffusion_steps']),device=device)
    train_gen=torch.Generator(device=device); train_gen.manual_seed(a.seed*1009+17)
    rank_gen=torch.Generator(device=device); rank_gen.manual_seed(a.seed*1013+29)
    hist=[]; best=None; bad=0; start=time.time(); opt_steps=0; train_forward_calls=0; shared=list(model.cond.parameters())
    for epoch in range(1,max_epochs+1):
        model.train(); rng=np.random.default_rng(a.seed+epoch)
        sums={'loss':0.,'diff':0.,'rank':0.,'keep':0.,'n':0}; first_grad=None; low=[]; mid=[]; high=[]
        for update,rows in enumerate(batches(train_local,batch,rng)):
            f=torch.as_tensor(feat[rows],device=device); cl=torch.as_tensor(clean[rows],device=device); ac=torch.as_tensor(active[rows],device=device); sb=torch.as_tensor(s0w[rows],device=device); pos=torch.as_tensor(target_pos[rows],device=device)
            if a.variant=='C':
                pred=model(f); train_forward_calls+=1; ldiff=masked_query_mse(pred,cl,ac); rank_pred=pred
            else:
                t=torch.randint(0,len(alpha),(len(rows),),device=device,generator=train_gen)
                eps=project_zero_mean(torch.randn(cl.shape,device=device,generator=train_gen)); at=alpha[t][:,None]; xt=at.sqrt()*cl+(1-at).sqrt()*eps
                pred=model(xt,t,f); train_forward_calls+=1; ldiff=masked_query_mse(pred,cl,ac)
                per=(((pred-cl)**2)*ac).sum(1)/ac.sum(1).clamp_min(1); lo=t<len(alpha)//3; hi=t>=2*len(alpha)//3; md=~(lo|hi)
                low.extend(per[lo].detach().cpu().tolist()); mid.extend(per[md].detach().cpu().tolist()); high.extend(per[hi].detach().cpu().tolist())
                if a.variant=='D0': rank_pred=pred
                else:
                    z=torch.randn(cl.shape,device=device,generator=rank_gen)
                    rank_pred=ddim_from_noise_differentiable(model,f,z,int(mc['ddim_steps']),int(mc['diffusion_steps'])); train_forward_calls+=int(mc['ddim_steps'])
            finite_tensor('prediction',pred); finite_tensor('rank_prediction',rank_pred)
            scores=deploy_scores(sb,rank_pred,sigma,float(cfg['residual']['training_eta']),float(cfg['residual']['clip_c']),float(cfg['residual']['tau']))
            lrank=masked_pairwise_rank_loss(scores,pos,ac); lkeep=(pred**2).mean(); wkeep=float(mc['lambda_keep'])*lkeep
            loss=ldiff+float(mc['lambda_rank'])*lrank+wkeep; finite_tensor('loss',loss)
            if first_grad is None:
                first_grad={'diff_raw':grad_norm(ldiff,shared),'rank_weighted':grad_norm(float(mc['lambda_rank'])*lrank,shared),'keep_weighted':grad_norm(wkeep,shared)}
                if a.variant=='D1' and (not np.isfinite(first_grad['rank_weighted']) or first_grad['rank_weighted']<=0): raise RuntimeError('D1 terminal rank gradient is zero/nonfinite')
            opt.zero_grad(set_to_none=True); loss.backward()
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all(): raise FloatingPointError('nonfinite gradient')
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step(); opt_steps+=1
            n=len(rows); sums['loss']+=float(loss.detach())*n; sums['diff']+=float(ldiff.detach())*n; sums['rank']+=float(lrank.detach())*n; sums['keep']+=float(lkeep.detach())*n; sums['n']+=n
        rec={'epoch':epoch,'optimizer_steps_total':opt_steps,'train_forward_calls_total':train_forward_calls,'train':{k:(v/max(sums['n'],1) if k!='n' else int(v)) for k,v in sums.items()},'weighted_grad_norm_shared_first_batch':first_grad}
        if a.variant!='C': rec['masked_timestep_mse']={'low':float(np.mean(low)) if low else None,'mid':float(np.mean(mid)) if mid else None,'high':float(np.mean(high)) if high else None}
        if epoch%eval_every==0:
            model.eval(); dev_eval,_=eval_dataset(model,a.variant,devdata,devsets,sigma,cfg,device); int_eval,_=eval_dataset(model,a.variant,intdata,intsets,sigma,cfg,device); chosen=choose_dev(dev_eval); eta=chosen[0]
            rec['dev']=dev_eval; rec['dev']['chosen_eta']=eta; rec['dev']['chosen_U']=chosen[1]['U']; rec['internal']=int_eval; rec['internal']['at_dev_chosen_eta']=int_eval['five_step_four_sample_etas'][str(eta)]
            if epoch>=min_epoch:
                score=float(chosen[1]['U'])
                if best is None or score>best['U']+1e-12:
                    best={'U':score,'epoch':epoch,'eta':eta,'dev_result':chosen[1],'internal_result':int_eval['five_step_four_sample_etas'][str(eta)]}
                    torch.save({'model':model.state_dict(),'epoch':epoch,'eta':eta,'sigma_r':sigma,'variant':a.variant,'seed':a.seed},out/'best.pt'); bad=0
                else: bad+=1
        hist.append(rec); (out/'history.json').write_text(json.dumps(hist,indent=2)+'\n')
        if epoch>=min_epoch and bad>=patience: break
    if best is None: raise RuntimeError('no checkpoint selected')
    ck=torch.load(out/'best.pt',map_location=device,weights_only=False); model.load_state_dict(ck['model']); model.eval(); eta=float(best['eta'])
    t0=time.perf_counter(); dev_eval,dp=eval_dataset(model,a.variant,devdata,devsets,sigma,cfg,device); infer_dev_seconds=time.perf_counter()-t0
    int_eval,ip=eval_dataset(model,a.variant,intdata,intsets,sigma,cfg,device); train_eval,tp=eval_dataset(model,a.variant,traindata,trainsets,sigma,cfg,device)
    dev_new=rerank(ditems,ds0w,dp['five_avg'],sigma,eta,float(cfg['residual']['clip_c']),float(cfg['residual']['tau']))
    int_new=rerank(iitems,is0w,ip['five_avg'],sigma,eta,float(cfg['residual']['clip_c']),float(cfg['residual']['tau']))
    train_new=rerank(sup_items,s0w,tp['five_avg'],sigma,eta,float(cfg['residual']['clip_c']),float(cfg['residual']['tau']))
    if not np.array_equal(rerank(ditems,ds0w,dp['five_avg'],sigma,0.0,float(cfg['residual']['clip_c']),1.0),ditems): raise RuntimeError('eta0 identity failed')
    if not np.array_equal(dev_new[:,:5],ditems[:,:5]) or not np.array_equal(dev_new[:,30:],ditems[:,30:]): raise RuntimeError('fixed slots changed')
    for old,new in zip(ditems,dev_new):
        if set(old.tolist())!=set(new.tolist()) or len(set(new.tolist()))!=100: raise RuntimeError('candidate set changed')
    br=target_ranks(sup_items,sup_targets); ar=target_ranks(train_new,sup_targets)
    train_rank={'affected94':rank_summary(br[affected],ar[affected]),'unaffected762':rank_summary(br[~affected],ar[~affected]),'all856':rank_summary(br,ar)}
    np.savez_compressed(out/'predictions.npz',dev_users=dusers,dev_base=ditems,dev_new=dev_new,internal_users=iusers,internal_targets=itargets,internal_base=iitems,internal_new=int_new,train_users=sup_users,train_targets=sup_targets,train_base=sup_items,train_new=train_new,affected94=affected)
    git_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(); tracked_dirty=subprocess.run(['git','diff','--quiet'],cwd=ROOT).returncode!=0
    result={'status':'COMPLETE','variant':a.variant,'seed':a.seed,'mode':a.mode,'git_sha':git_sha,'tracked_dirty':tracked_dirty,'config_sha256':sha(ROOT/'diffusion_experiments/configs/round2_baby.yaml'),'supervision_sha256':sha(supdir/'supervision.npz'),'mask_version':manifest['mask_version'],'parameter_count':nparams,'supervised_queries_total':int(len(sup_rows)),'supervised_queries_used':int(len(train_local)),'affected_monitor_queries':int(affected.sum()),'sigma_r':sigma,'best':best,'final_dev':dev_eval,'final_internal':int_eval,'final_train_probe':train_eval,'train_rank_diagnostic':train_rank,'epochs_completed':len(hist),'optimizer_steps':opt_steps,'train_forward_calls':train_forward_calls,'elapsed_train_seconds':time.time()-start,'final_dev_inference_seconds_all_paths':infer_dev_seconds,'peak_cuda_memory_bytes':int(torch.cuda.max_memory_allocated(device)),'gpu_name':torch.cuda.get_device_name(0),'access':{'CONFIRM_ACCESSED':False,'TEST_ACCESSED':False,'TEST_USED_FOR_SELECTION':False},'training_sampling':'D1 uses one independent terminal Gaussian/query batch draw; train/eval RNG streams separate','dropout_semantics':'D1 differentiable terminal chain runs in model.train(); evaluation sampling runs in model.eval()'}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':'COMPLETE','variant':a.variant,'seed':a.seed,'best':best,'optimizer_steps':opt_steps,'peak_cuda_memory_bytes':result['peak_cuda_memory_bytes']},sort_keys=True))
if __name__=='__main__': main()
