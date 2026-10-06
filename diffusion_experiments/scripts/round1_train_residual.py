from __future__ import annotations
import argparse,hashlib,json,math,subprocess,sys,time
from pathlib import Path
import numpy as np,pandas as pd,torch,yaml
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'src'))
from diffusion_experiments.models.round1_residual import DeterministicResidual,BoundaryResidualDiffusion,cosine_alpha_bars,project_zero_mean,make_clean_target,deploy_scores,ddim_sample
from modules.ranking import metrics_at

PRIMARY=('R10','N10','R20','N20'); ALL=('R10','N10','R20','N20','R50','N50')
def dev_labels(inter,users):
    keep=set(map(int,users)); out={int(u):set() for u in users}
    for c in pd.read_csv(inter,sep='\t',usecols=['userID','itemID','x_label'],chunksize=50000):
        z=c[(c.x_label==1)&c.userID.isin(keep)]
        for u,g in z.groupby('userID'): out[int(u)].update(g.itemID.astype(int).tolist())
    return out
def calc_metrics(items,users,sets): return metrics_at(items,users,sets)
def result_vs(base,new):
    rel={k:(float(new[k])-float(base[k]))/max(float(base[k]),1e-12) for k in PRIMARY}
    return {'metrics':{k:float(new[k]) for k in ALL},'relative_primary':rel,'U':float(np.mean(list(rel.values()))),'delta_R50':float(new['R50']-base['R50']),'delta_N50':float(new['N50']-base['N50'])}
def protected(res,cfg):
    rel=res['relative_primary']; nonneg=sum(v>=0 for v in rel.values())
    return bool(nonneg>=int(cfg['min_nonnegative_primary']) and min(rel.values())>=-float(cfg['max_primary_relative_regression']) and res['delta_R50']>=float(cfg['min_absolute_delta_R50']) and res['delta_N50']>=float(cfg['min_absolute_delta_N50']))
def rerank(items,s0,x0,sigma,eta,c,tau):
    if float(eta)==0: return items.copy()
    sc=deploy_scores(torch.as_tensor(s0),torch.as_tensor(x0),sigma,eta,c,tau).numpy()
    order=np.argsort(-sc,axis=1,kind='stable'); out=items.copy(); out[:,5:30]=np.take_along_axis(items[:,5:30],order,axis=1); return out
def batches(idx,batch,rng):
    p=rng.permutation(idx)
    for s in range(0,len(p),batch): yield p[s:s+batch]
def grad_norm(loss,params):
    gs=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
    return float(math.sqrt(sum(float((g.detach()**2).sum()) for g in gs if g is not None)))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--assets',required=True); ap.add_argument('--model',choices=['deterministic','diffusion'],required=True); ap.add_argument('--seed',type=int,required=True); ap.add_argument('--out',required=True); ap.add_argument('--mode',choices=['smoke','formal'],default='formal')
    a=ap.parse_args(); assets=Path(a.assets); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    cfg=yaml.safe_load((ROOT/'diffusion_experiments/configs/round1_baby.yaml').read_text()); mc=cfg['model']; rc=cfg['residual']; formal=cfg['formal']; pcfg=cfg['protection']
    audit=json.loads((assets/'audit.json').read_text());
    if audit['access']['CONFIRM_ACCESSED'] or audit['access']['TEST_ACCESSED']: raise RuntimeError('closed-set invariant')
    if not torch.cuda.is_available() or '5090' not in torch.cuda.get_device_name(0): raise RuntimeError('RTX5090 required')
    device=torch.device('cuda:0'); torch.manual_seed(a.seed); np.random.seed(a.seed)
    probe=np.load(assets/'probe_top100.npz'); tgt=np.load(assets/'probe_targets.npz'); dev=np.load(assets/'dev_top100.npz')
    users=probe['users'].astype(np.int64); items=probe['items'].astype(np.int32); s0=probe['s0'].astype(np.float32); feats=probe['features'].astype(np.float32)
    ranks=tgt['target_rank']; rr=tgt['reranker_train'].astype(bool); internal=tgt['internal'].astype(bool)
    win=feats[:,5:30,:]; s0w=s0[:,5:30]; sup=rr&(ranks>=6)&(ranks<=30); insup=internal&(ranks>=6)&(ranks<=30)
    if sup.sum()<10: raise RuntimeError(f'too few supervised queries: {sup.sum()}')
    mean=win[rr].reshape(-1,win.shape[-1]).mean(0); std=win[rr].reshape(-1,win.shape[-1]).std(0); std=np.maximum(std,1e-6)
    xfeat=((win-mean)/std).astype(np.float32); target_pos=(ranks[sup]-6).astype(np.int64); rstar=make_clean_target(s0w[sup],target_pos,float(rc['epsilon']),float(rc['tau'])); sigma=float(np.sqrt(np.mean(rstar*rstar))); x0=(rstar/sigma).astype(np.float32)
    sup_idx=np.flatnonzero(sup); pos_by_row=np.full(len(users),-1,dtype=np.int64); pos_by_row[sup_idx]=target_pos; x0_by_row=np.zeros((len(users),25),np.float32); x0_by_row[sup_idx]=x0
    dusers=dev['users'].astype(np.int64); ditems=dev['items'].astype(np.int32); ds0=dev['s0'].astype(np.float32); dfeat=((dev['features'][:,5:30,:].astype(np.float32)-mean)/std).astype(np.float32)
    eval_sets=dev_labels(ROOT/'data/baby/baby.inter',dusers); base=calc_metrics(ditems,dusers,eval_sets)
    cls=DeterministicResidual if a.model=='deterministic' else BoundaryResidualDiffusion
    model=cls(win.shape[-1],int(mc['hidden_dim']),int(mc['attention_heads']),int(mc['attention_layers']),float(mc['dropout'])).to(device); opt=torch.optim.AdamW(model.parameters(),lr=float(mc['learning_rate']),weight_decay=float(mc['weight_decay'])); nparams=sum(p.numel() for p in model.parameters())
    max_epochs=2 if a.mode=='smoke' else int(formal['max_epochs']); eval_every=1 if a.mode=='smoke' else int(formal['eval_every']); min_epoch=1 if a.mode=='smoke' else int(formal['min_epochs']); patience=int(formal['patience_evals']); batch=int(mc['batch_size']); alpha=cosine_alpha_bars(int(mc['diffusion_steps']),device=device)
    hist=[]; best=None; bad=0; start=time.time(); shared=list(model.cond.parameters())
    for epoch in range(1,max_epochs+1):
        model.train(); rng=np.random.default_rng(a.seed+epoch); sums={'loss':0.,'diff':0.,'rank':0.,'keep':0.,'n':0}; gb=None; low=[]; high=[]; snr_low=[]; snr_mid=[]; snr_high=[]
        for rows in batches(sup_idx,batch,rng):
            f=torch.as_tensor(xfeat[rows],device=device); clean=torch.as_tensor(x0_by_row[rows],device=device); sb=torch.as_tensor(s0w[rows],device=device); pos=torch.as_tensor(pos_by_row[rows],device=device)
            if a.model=='deterministic': pred=model(f); ldiff=F.mse_loss(pred,clean); t=None
            else:
                t=torch.randint(0,len(alpha),(len(rows),),device=device); eps=project_zero_mean(torch.randn_like(clean)); at=alpha[t][:,None]; xt=at.sqrt()*clean+(1-at).sqrt()*eps; pred=model(xt,t,f); per=((pred-clean)**2).mean(1); ldiff=per.mean(); low_mask=t<len(alpha)//3; high_mask=t>=2*len(alpha)//3; mid_mask=~(low_mask|high_mask); low.extend(per[low_mask].detach().cpu().tolist()); high.extend(per[high_mask].detach().cpu().tolist()); sig=(at*clean.pow(2)).mean(1); noi=((1-at)*eps.pow(2)).mean(1).clamp_min(1e-12); snr=(sig/noi).detach(); snr_low.extend(snr[low_mask].cpu().tolist()); snr_mid.extend(snr[mid_mask].cpu().tolist()); snr_high.extend(snr[high_mask].cpu().tolist())
            deployed=deploy_scores(sb,pred,sigma,float(rc['training_eta']),float(rc['clip_c']),float(rc['tau'])); ps=deployed.gather(1,pos[:,None]); mask=torch.ones_like(deployed,dtype=torch.bool); mask.scatter_(1,pos[:,None],False); lrank=F.softplus(-(ps.expand_as(deployed)[mask]-deployed[mask])).mean(); lkeep=(pred**2).mean(); loss=ldiff+float(mc['lambda_rank'])*lrank+float(mc['lambda_keep'])*lkeep
            if gb is None: gb={'diff':grad_norm(ldiff,shared),'rank':grad_norm(lrank,shared),'keep':grad_norm(lkeep,shared)}
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
            n=len(rows); sums['loss']+=float(loss)*n; sums['diff']+=float(ldiff)*n; sums['rank']+=float(lrank)*n; sums['keep']+=float(lkeep)*n; sums['n']+=n
        rec={'epoch':epoch,'train':{k:(v/max(sums['n'],1) if k!='n' else int(v)) for k,v in sums.items()},'grad_norm_shared':gb}
        if a.model=='diffusion': rec['timestep_mse']={'low':float(np.mean(low)) if low else None,'high':float(np.mean(high)) if high else None}; rec['empirical_snr']={'low_t':float(np.mean(snr_low)) if snr_low else None,'mid_t':float(np.mean(snr_mid)) if snr_mid else None,'high_t':float(np.mean(snr_high)) if snr_high else None}
        if epoch%eval_every==0:
            model.eval(); pred_parts=[]; seed_parts={int(s):[] for s in formal['sampling_seeds']}
            with torch.no_grad():
                for st in range(0,len(dusers),512):
                    en=min(st+512,len(dusers)); f=torch.as_tensor(dfeat[st:en],device=device); uu=torch.as_tensor(dusers[st:en],device=device)
                    if a.model=='deterministic': pred_parts.append(model(f).cpu().numpy())
                    else:
                        for ss in formal['sampling_seeds']: seed_parts[int(ss)].append(ddim_sample(model,f,uu,sigma,int(ss),int(mc['ddim_steps']),int(mc['diffusion_steps']),device).cpu().numpy())
            if a.model=='deterministic': xpred=np.concatenate(pred_parts); first=xpred
            else:
                per_seed={s:np.concatenate(v) for s,v in seed_parts.items()}; first=per_seed[int(formal['sampling_seeds'][0])]; xpred=np.mean(np.stack(list(per_seed.values()),axis=0),axis=0)
            eta_results={}; first_results={}
            for eta in rc['eta_candidates']:
                ranked=rerank(ditems,ds0[:,5:30],xpred,sigma,float(eta),float(rc['clip_c']),float(rc['tau'])); m=calc_metrics(ranked,dusers,eval_sets); r=result_vs(base,m); r['protected']=protected(r,pcfg); eta_results[str(eta)]=r
                if a.model=='diffusion':
                    r1=result_vs(base,calc_metrics(rerank(ditems,ds0[:,5:30],first,sigma,float(eta),float(rc['clip_c']),float(rc['tau'])),dusers,eval_sets)); r1['protected']=protected(r1,pcfg); first_results[str(eta)]=r1
            if not np.array_equal(rerank(ditems,ds0[:,5:30],xpred,sigma,0.0,float(rc['clip_c']),float(rc['tau'])),ditems): raise RuntimeError('eta=0 identity failed')
            candidates=[(float(k),v) for k,v in eta_results.items() if v['protected']]; chosen=max(candidates,key=lambda kv:kv[1]['U']) if candidates else (0.0,eta_results['0.0'])
            rec['dev']={'baseline':{k:float(base[k]) for k in ALL},'etas':eta_results,'chosen_eta':chosen[0],'chosen_U':chosen[1]['U'],'first_sample_etas':first_results if a.model=='diffusion' else None}
            if insup.sum():
                ii=np.flatnonzero(insup); ipos=(ranks[ii]-6).astype(np.int64); ir=make_clean_target(s0w[ii],ipos,float(rc['epsilon']),float(rc['tau']))/sigma; outs=[]
                with torch.no_grad():
                    for st in range(0,len(ii),512):
                        jj=ii[st:st+512]; f=torch.as_tensor(xfeat[jj],device=device); uu=torch.as_tensor(users[jj],device=device)
                        if a.model=='deterministic': z=model(f)
                        else: z=ddim_sample(model,f,uu,sigma,int(formal['sampling_seeds'][0]),int(mc['ddim_steps']),int(mc['diffusion_steps']),device)
                        outs.append(z.cpu().numpy())
                rec['internal']={'supervised_queries':int(len(ii)),'terminal_path_mse':float(np.mean((np.concatenate(outs)-ir)**2))}
            if epoch>=min_epoch:
                score=float(chosen[1]['U'])
                if best is None or score>best['U']+1e-12:
                    best={'U':score,'epoch':epoch,'eta':chosen[0],'dev_result':chosen[1]}; torch.save({'model':model.state_dict(),'epoch':epoch,'eta':chosen[0],'sigma_r':sigma,'feature_mean':mean,'feature_std':std,'model_type':a.model,'seed':a.seed},out/'best.pt'); bad=0
                else: bad+=1
        hist.append(rec); (out/'history.json').write_text(json.dumps(hist,indent=2)+'\n')
        if epoch>=min_epoch and bad>=patience: break
    if best is None: raise RuntimeError('no checkpoint selected')
    git_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    git_dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip())
    asset_hashes={name:hashlib.sha256((assets/name).read_bytes()).hexdigest() for name in ['audit.json','probe_top100.npz','dev_top100.npz','probe_targets.npz']}
    result={'status':'COMPLETE','git_sha':git_sha,'git_dirty':git_dirty,'asset_hashes':asset_hashes,'model':a.model,'seed':a.seed,'mode':a.mode,'parameter_count':nparams,'supervised_train_queries':int(sup.sum()),'no_positive_window_train_queries':int(rr.sum()-sup.sum()),'no_positive_loss_policy':'excluded from L_diff/L_rank; no future-negative assumption; evaluated at deployment','supervised_internal_queries':int(insup.sum()),'sigma_r':sigma,'residual_rms_before_scale':sigma,'best':best,'epochs_completed':len(hist),'elapsed_seconds':time.time()-start,'gpu_name':torch.cuda.get_device_name(0),'access':{'CONFIRM_ACCESSED':False,'TEST_ACCESSED':False},'assets_audit':str((assets/'audit.json').resolve())}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,sort_keys=True))
if __name__=='__main__': main()
