from __future__ import annotations
import gc,hashlib,json
from pathlib import Path
import numpy as np
import torch
from modules.ranking import metric_arrays
from diffusion_experiments.round33_seed1000_cross_domain import round33_core as c
from diffusion_experiments.round33_seed1000_cross_domain import round33_diffusion as d

def _opt(model,config):
    wd=float(config['weight_decay']) if config['weight_decay'] is not None else 0.
    opt=torch.optim.Adam(model.parameters(),lr=float(config['learning_rate']),weight_decay=wd)
    fac=config['learning_rate_scheduler']
    sch=torch.optim.lr_scheduler.LambdaLR(opt,lr_lambda=lambda e:fac[0]**(e/fac[1]))
    return opt,sch,wd

def _save_train(dataset,variant,out):
    c.json_write(c.EVID/f'ROUND33_{dataset.upper()}_{variant}_TRAIN.json',out)

def train_b0(dataset,evaluator,formal=True,smoke_batches=2):
    model,config,td,vd,pack=c.instantiate_shared(dataset);opt,sch,wd=_opt(model,config)
    epochs=list(range(c.FORMAL_EPOCHS)) if formal else [0]
    best=-1.;best_epoch=-1;best_eval=None;best_path=None;traj=[];plans={};lrs=[];steps=0
    for ep in epochs:
        hh=hashlib.sha256();losses=[];lrs.append(float(opt.param_groups[0]['lr']));model.train()
        for bi,interaction in enumerate(td):
            if not formal and bi>=smoke_batches:break
            for x in interaction[:3]:hh.update(x.detach().cpu().numpy().astype(np.int64).tobytes())
            opt.zero_grad(set_to_none=True);b=c.forward_bundle(model);loss=c.msca_loss_from_bundle(model,interaction,b)
            if not torch.isfinite(loss):raise RuntimeError(f'nonfinite B0 {dataset} ep{ep} bi{bi}')
            loss.backward()
            if config['clip_grad_norm']:torch.nn.utils.clip_grad_norm_(model.parameters(),**config['clip_grad_norm'])
            opt.step();losses.append(float(loss.detach()));steps+=1
        sch.step();plans[str(ep)]=hh.hexdigest();ev=evaluator.evaluate(model);score=float(ev['colift']['R20'])
        rec={'epoch':ep,'loss':float(np.mean(losses)),'evaluation':ev,'full_colift_R20':score};traj.append(rec)
        if formal and score>best:
            best=score;best_epoch=ep;best_eval=ev;best_path=c.save_best(model,None,dataset,'B0',ep,score)
        print(json.dumps({'dataset':dataset,'variant':'B0','epoch':ep,'R20':score,'R10':ev['colift']['R10'],'N10':ev['colift']['N10'],'N20':ev['colift']['N20']}),flush=True)
    out={'protocol':c.PROTOCOL,'protocol_hash':c.protocol_hash(),'dataset':dataset,'seed':c.SEED,'variant':'B0',
         'initial_state_hash':pack['state_hash'],'epochs_trained':len(epochs),'normal_plan_hashes':plans,'lr_trajectory':lrs,
         'optimizer_steps':steps,'weight_decay':wd,'best_epoch':best_epoch if formal else None,'best_evaluation':best_eval,
         'best_checkpoint':str(best_path) if formal else None,'checkpoint_sha256':c.sha256_file(best_path) if formal else None,
         'trajectory':traj,'checkpoint_selection':'best Full CoLiftRec Validation R20','TEST_ACCESSED':False}
    if formal:_save_train(dataset,'B0',out)
    del model,opt;torch.cuda.empty_cache();gc.collect();return out

def _calibrate_bands(b,interaction,ids,dirs):
    user=b['fu'][interaction[0]].detach();pos=b['fi'][interaction[1]].detach();deltas=[];diags=[]
    for j,x in enumerate(ids):
        delta,diag=d.calibrate(user,pos,b['fi'][x].detach(),dirs[j]);diag['band']=c.BAND_NAMES[j];deltas.append(delta);diags.append(diag)
    return deltas,diags

def train_m25(dataset,evaluator,formal=True,smoke_batches=2):
    model,config,td,vd,pack=c.instantiate_shared(dataset)
    events=c.TrainEvents(dataset,model)
    opt,sch,wd=_opt(model,config)
    net,dopt,dsched=c.make_diff(c.SEED,model.device)
    mask=c.valid_mask(td,model.n_items,model.device)
    epochs=list(range(c.FORMAL_EPOCHS)) if formal else [0,10]
    best=-1.;best_epoch=-1;best_eval=None;best_path=None
    traj=[];plans={};pair={};offs_hash={};lrs=[];steps=0;dsteps=0;mech_epochs={}
    for ep in epochs:
        active=c.curriculum_count(ep);s=c.latent_stats(model)
        hh=hashlib.sha256();ph=hashlib.sha256();oh=hashlib.sha256()
        losses=[];aux=[];dl=[];db=[];cb=[]
        lrs.append(float(opt.param_groups[0]['lr']));model.train()
        for bi,interaction in enumerate(td):
            if not formal and bi>=smoke_batches:break
            u=interaction[0].detach().cpu().numpy();p=interaction[1].detach().cpu().numpy();n=interaction[2].detach().cpu().numpy()
            hh.update(u.tobytes());hh.update(p.tobytes());hh.update(n.tobytes())
            ph.update(u.tobytes());ph.update(p.tobytes())
            offsets=c.rank_offsets(c.SEED,ep,bi,len(interaction[0]),active)
            for q in offsets:oh.update(np.asarray(q,dtype=np.int64).tobytes())
            opt.zero_grad(set_to_none=True);b=c.forward_bundle(model)
            base=c.msca_loss_from_bundle(model,interaction,b);ids=[]
            la=torch.zeros((),device=model.device)
            if active>0:
                ids,used,_=c.mine_current_ids(b,interaction,events,mask,c.SEED,ep,bi,active)
                for a,zv in zip(offsets,used):
                    assert np.array_equal(a,zv)
            if active==0:
                q=d.warmup_step(net,dopt,dsched,b,interaction,s,202860000+c.SEED*100000+ep*1000+bi)
                dl.append(q);dsteps+=1
            else:
                for inner in range(c.DIFF_STEPS_ACTIVE):
                    q=d.direction_inner_step(net,dopt,dsched,b,interaction,s,ids,202861000+c.SEED*100000+ep*10000+bi*10+inner)
                    dl.append(q);dsteps+=1
                dirs,ddiag,_=d.predict_directions(net,dsched,b,interaction,s,ids,202862000+c.SEED*100000+ep*1000+bi)
                db.append(ddiag)
                deltas,cdiag=_calibrate_bands(b,interaction,ids,dirs);cb.append(cdiag)
                refined=[b['fi'][ids[j]]+deltas[j] for j in range(active)]
                la=c.aux_bpr(b['fu'][interaction[0]],b['fi'][interaction[1]],refined)
            total=base+(c.LAMBDA_HN*la if active>0 else 0.)
            assert torch.isfinite(total)
            total.backward()
            if config['clip_grad_norm']:torch.nn.utils.clip_grad_norm_(model.parameters(),**config['clip_grad_norm'])
            opt.step();losses.append(float(base.detach()));aux.append(float(la.detach()) if active>0 else 0.);steps+=1
        sch.step();plans[str(ep)]=hh.hexdigest();pair[str(ep)]=ph.hexdigest();offs_hash[str(ep)]=oh.hexdigest()
        ev=evaluator.evaluate(model);score=float(ev["colift"]["R20"])
        rec={'epoch':ep,'active':active,'normal_loss':float(np.mean(losses)),'aux_loss':float(np.mean(aux)),'evaluation':ev,'full_colift_R20':score}
        if active>0:
            mech=d.summarize(db,cb,dl);rec["mechanism"]=mech;mech_epochs[str(ep)]=mech
            assert mech["angle_bound_pass"] and mech["target_never_exceeds_positive"]
        traj.append(rec)
        if formal and score>best:
            best=score;best_epoch=ep;best_eval=ev;best_path=c.save_best(model,net,dataset,'M25',ep,score)
        print(json.dumps({"dataset":dataset,"variant":"M25","epoch":ep,"active":active,"R20":score,"R10":ev["colift"]["R10"],"N10":ev["colift"]["N10"],"N20":ev["colift"]["N20"]}),flush=True)
    out={'protocol':c.PROTOCOL,'protocol_hash':c.protocol_hash(),'dataset':dataset,'seed':c.SEED,'variant':'M25',
         'config':{'lambda_HN':c.LAMBDA_HN,'half_margin':c.HALF_MARGIN,'theta_max_deg':c.THETA_MAX_DEG},
         'initial_state_hash':pack['state_hash'],'epochs_trained':len(epochs),'normal_plan_hashes':plans,'aux_pair_hashes':pair,
         'rank_offset_hashes':offs_hash,'lr_trajectory':lrs,'optimizer_steps':steps,'diffusion_steps':dsteps,'weight_decay':wd}
    out.update({'best_epoch':best_epoch if formal else None,'best_evaluation':best_eval,'best_checkpoint':str(best_path) if formal else None,
                'checkpoint_sha256':c.sha256_file(best_path) if formal else None,'mechanism_at_best':mech_epochs.get(str(best_epoch)) if formal else mech_epochs.get('10'),
                'trajectory':traj,'checkpoint_selection':'best Full CoLiftRec Validation R20','RANKING_SCORE_GRAD_TO_DIFFUSION':0,'TEST_ACCESSED':False})
    if formal:_save_train(dataset,'M25',out)
    del model,opt,net,dopt;torch.cuda.empty_cache();gc.collect();return out

def _load(dataset,variant):
    tr=json.loads((c.EVID/f'ROUND33_{dataset.upper()}_{variant}_TRAIN.json').read_text())
    p=Path(tr['best_checkpoint']);assert c.sha256_file(p)==tr['checkpoint_sha256']
    cp=torch.load(p,map_location='cpu',weights_only=False)
    model,_,_,_,_=c.instantiate_shared(dataset)
    model.load_state_dict(cp['model_state_dict'],strict=True);model.eval();return model,tr

def _per_user(ranked,users,sets):
    _,_,hit,pos_len=metric_arrays(ranked,users,sets,max_k=20)
    rr=np.arange(1,21,dtype=np.float64)[None,:]
    recall=np.cumsum(hit,axis=1)/pos_len[:,None]
    dcg=np.cumsum(hit/np.log2(rr+1),axis=1)
    idcg=np.cumsum(np.ones((len(users),20))/np.log2(rr+1),axis=1)
    for i,n in enumerate(pos_len):
        cut=min(int(n),20)
        if cut<20:idcg[i,cut:]=idcg[i,cut-1]
    nd=dcg/idcg
    return np.stack([recall[:,9],nd[:,9],recall[:,19],nd[:,19]],1)

def _bootstrap(base,new,resamples=1000,seed=20263300):
    def uu(a,b):
        am=a.mean(0);bm=b.mean(0);return float(np.mean((bm-am)/np.maximum(am,1e-12)))
    rng=np.random.default_rng(seed);vals=np.empty(resamples,np.float64)
    for r in range(resamples):
        ix=rng.integers(0,len(base),len(base));vals[r]=uu(base[ix],new[ix])
    return {'point_U':uu(base,new),'ci95':[float(np.quantile(vals,.025)),float(np.quantile(vals,.975))],
            'positive_replicate_fraction':float((vals>0).mean()),'resamples':resamples}

def evaluate_test(dataset):
    m0,t0=_load(dataset,'B0');ev=c.DomainEvaluator(dataset,m0,'test')
    a0=ev.evaluate(m0,True);arr0=_per_user(a0.pop("ranked_items"),ev.users,ev.sets)
    m1,t1=_load(dataset,'M25');a1=ev.evaluate(m1,True);arr1=_per_user(a1.pop('ranked_items'),ev.users,ev.sets)
    out={'protocol':c.PROTOCOL,'dataset':dataset,'seed':c.SEED}
    out['parameter_source']='Baby seed1000 M25 frozen before cross-domain evaluation'
    out['M25']={'lambda_HN':c.LAMBDA_HN,'half_margin':c.HALF_MARGIN,'theta_max_deg':c.THETA_MAX_DEG}
    out['B0']=a0;out['M25_result']=a1
    out['MSCA_to_CoLiftRec']=c.delta_pack(a0['colift'],a0['backbone'])
    out['Diffusion_vs_CoLiftRec']=c.delta_pack(a1['colift'],a0['colift'])
    out['Diffusion_backbone_vs_B0_backbone']=c.delta_pack(a1['backbone'],a0['backbone'])
    out['bootstrap_Diffusion_vs_CoLiftRec']=_bootstrap(arr0,arr1,1000,20263300+(1 if dataset=='sports' else 2))
    out['B0_best_epoch']=t0['best_epoch'];out['M25_best_epoch']=t1['best_epoch']
    out['TEST_ACCESSED']=True;out['TEST_USED_FOR_TUNING']=False;out['diffusion_inference_calls']=0
    c.json_write(c.EVID/f'ROUND33_{dataset.upper()}_TEST.json',out)
    del m0,m1,ev;torch.cuda.empty_cache();gc.collect();return out

def run_domain(dataset,smoke=False):
    assert dataset in c.DATASETS
    init_path=c.ASSETS/f"{dataset}_seed{c.SEED}_init.pt"
    if not init_path.exists():c.create_shared_init(dataset)
    probe,_,_,_,_=c.instantiate_shared(dataset)
    ev=c.DomainEvaluator(dataset,probe,'validation')
    del probe;torch.cuda.empty_cache();gc.collect()
    b0=train_b0(dataset,ev,formal=not smoke)
    m25=train_m25(dataset,ev,formal=not smoke)
    if smoke:return {"dataset":dataset,"smoke":True,"B0":b0["trajectory"][-1],"M25":m25["trajectory"][-1]}
    fair={'initial_state_exact':b0['initial_state_hash']==m25['initial_state_hash']}
    fair['normal_plan_exact']=b0['normal_plan_hashes']==m25['normal_plan_hashes']
    fair['lr_trajectory_exact']=b0['lr_trajectory']==m25['lr_trajectory']
    fair['PASS']=all(fair.values());assert fair['PASS']
    c.json_write(c.EVID/f'ROUND33_{dataset.upper()}_FAIRNESS.json',fair)
    val={'dataset':dataset,'seed':c.SEED,'B0':b0['best_evaluation'],'M25':m25['best_evaluation']}
    val['MSCA_to_CoLiftRec']=c.delta_pack(b0['best_evaluation']['colift'],b0['best_evaluation']['backbone'])
    val['Diffusion_vs_CoLiftRec']=c.delta_pack(m25['best_evaluation']['colift'],b0['best_evaluation']['colift'])
    val['B0_best_epoch']=b0['best_epoch'];val['M25_best_epoch']=m25['best_epoch']
    c.json_write(c.EVID/f'ROUND33_{dataset.upper()}_VALIDATION.json',val)
    test=evaluate_test(dataset)
    return {'dataset':dataset,'validation':val,'test':test,'fairness':fair}
