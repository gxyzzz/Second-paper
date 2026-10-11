from __future__ import annotations
import gc,hashlib,json
import numpy as np
import torch
from diffusion_experiments.round32_round26_parameter_sweep import round32_core as c
from diffusion_experiments.round32_round26_parameter_sweep import round32_diffusion as d


def _calibrate_bands(b,interaction,ids,dirs):
    user=b['fu'][interaction[0]].detach(); pos=b['fi'][interaction[1]].detach(); deltas=[]; diags=[]
    for j,x in enumerate(ids):
        delta,diag=d.calibrate(user,pos,b['fi'][x].detach(),dirs[j]); diag['band']=c.BAND_NAMES[j]; deltas.append(delta); diags.append(diag)
    return deltas,diags


def train_variant(seed,variant,formal=True,smoke_batches=2):
    cfg=c.apply_variant(variant)
    if variant not in c.TRAIN_VARIANTS: raise ValueError(variant)
    model,config,td,vd,pack=c.instantiate_shared(seed); events=c.TrainEvents(model); evaluator=c.CachedEvaluator(seed,model)
    wd=float(config['weight_decay']) if config['weight_decay'] is not None else 0.
    ropt=torch.optim.Adam(model.parameters(),lr=float(config['learning_rate']),weight_decay=wd)
    fac=config['learning_rate_scheduler']; rsched=torch.optim.lr_scheduler.LambdaLR(ropt,lr_lambda=lambda e:fac[0]**(e/fac[1]))
    net,dopt,dsched=c.make_diff(seed,model.device); mask=c.valid_mask(td,model.n_items,model.device)
    logical_epochs=list(range(c.FORMAL_EPOCHS)) if formal else [0,10]
    normal_hashes={}; pair_hashes={}; offset_hashes={}; id_hashes={}; lr_traj=[]; trajectory=[]; mechanism_epochs={}; total_steps=0; diff_steps=0
    best=-1.; best_epoch=-1; best_eval=None; best_path=None
    for run_ep,ep in enumerate(logical_epochs):
        active=c.curriculum_count(ep); s=c.latent_stats(model); hh=hashlib.sha256(); ph=hashlib.sha256(); oh=hashlib.sha256(); ih=hashlib.sha256(); nl=[]; al=[]; dl=[]; dir_batches=[]; cal_batches=[]
        lr_traj.append(float(ropt.param_groups[0]['lr'])); model.train()
        for bi,interaction in enumerate(td):
            if not formal and bi>=int(smoke_batches): break
            u_np=interaction[0].detach().cpu().numpy(); p_np=interaction[1].detach().cpu().numpy(); n_np=interaction[2].detach().cpu().numpy()
            hh.update(u_np.tobytes()); hh.update(p_np.tobytes()); hh.update(n_np.tobytes()); ph.update(u_np.tobytes()); ph.update(p_np.tobytes())
            offs=c.rank_offsets(seed,ep,bi,len(interaction[0]),active)
            for x in offs: oh.update(np.asarray(x,dtype=np.int64).tobytes())
            ropt.zero_grad(set_to_none=True); b=c.forward_bundle(model); lbase=c.msca_loss_from_bundle(model,interaction,b); ids=[]; la=torch.zeros((),device=model.device)
            if active>0:
                ids,used,_=c.mine_current_ids(b,interaction,events,mask,seed,ep,bi,active)
                for x in ids: ih.update(x.detach().cpu().numpy().astype(np.int64).tobytes())
                for a,z in zip(offs,used):
                    if not np.array_equal(a,z): raise RuntimeError('rank-offset policy mismatch')
            if active==0:
                q=d.warmup_step(net,dopt,dsched,b,interaction,s,202860000+seed*100000+ep*1000+bi); dl.append(q); diff_steps+=1
            else:
                for inner in range(c.DIFF_STEPS_ACTIVE):
                    q=d.direction_inner_step(net,dopt,dsched,b,interaction,s,ids,202861000+seed*100000+ep*10000+bi*10+inner); dl.append(q); diff_steps+=1
                dirs,ddiag,_=d.predict_directions(net,dsched,b,interaction,s,ids,202862000+seed*100000+ep*1000+bi); dir_batches.append(ddiag)
                deltas,cdiag=_calibrate_bands(b,interaction,ids,dirs); cal_batches.append(cdiag)
                if any(x.requires_grad for x in deltas): raise RuntimeError('Diffusion delta must be detached')
                negs=[b['fi'][ids[j]]+deltas[j] for j in range(active)]; la=c.aux_bpr(b['fu'][interaction[0]],b['fi'][interaction[1]],negs)
            total=lbase+(c.LAMBDA_HN*la if active>0 else 0.)
            if not torch.isfinite(total): raise RuntimeError(f'nonfinite rec loss seed{seed} {variant} ep{ep} batch{bi}')
            total.backward()
            if config['clip_grad_norm']: torch.nn.utils.clip_grad_norm_(model.parameters(),**config['clip_grad_norm'])
            ropt.step(); nl.append(float(lbase.detach())); al.append(float(la.detach()) if active>0 else 0.); total_steps+=1
        rsched.step(); normal_hashes[str(ep)]=hh.hexdigest(); pair_hashes[str(ep)]=ph.hexdigest(); offset_hashes[str(ep)]=oh.hexdigest(); id_hashes[str(ep)]=ih.hexdigest()
        ev=evaluator.evaluate(model); score=float(ev['colift']['R20'])
        rec={'run_epoch':run_ep,'epoch':ep,'active_negative_count':active,'normal_loss_mean':float(np.mean(nl)),'aux_loss_mean':float(np.mean(al)),'diff_loss_mean':float(np.mean([x['total'] for x in dl])),'lr':lr_traj[-1],'evaluation':ev,'full_colift_R20':score}
        if active>0:
            mech=d.summarize(dir_batches,cal_batches,dl); rec['mechanism']=mech; mechanism_epochs[str(ep)]=mech
            if not mech['angle_bound_pass']: raise RuntimeError(f'theta cap broken {variant} seed{seed} ep{ep}: max={mech["angle_deg"]["max"]}')
            if not mech['target_never_exceeds_positive']: raise RuntimeError('finite target overshoots positive')
        else:
            rec['warmup_diffusion']={k:c.stats(np.asarray([x[k] for x in dl])) for k in ('total','rec')}
        trajectory.append(rec)
        if formal and score>best:
            best=score; best_epoch=ep; best_eval=ev; best_path=c.save_best(model,net,seed,variant,ep,score)
        print(json.dumps({'phase':'formal' if formal else 'smoke','seed':seed,'variant':variant,'epoch':ep,'active':active,'R20':score,'R10':ev['colift']['R10'],'N10':ev['colift']['N10'],'N20':ev['colift']['N20'],'lambda_HN':c.LAMBDA_HN,'half_margin':c.HALF_MARGIN,'theta':c.THETA_MAX_DEG,'cap_hit':None if active==0 else rec['mechanism']['cap_hit_fraction'],'angle_max':None if active==0 else rec['mechanism']['angle_deg']['max']}),flush=True)
    out={'protocol':c.PROTOCOL,'protocol_hash':c.protocol_hash(),'seed':seed,'variant':variant,'config':cfg,'initial_state_hash':pack['state_hash'],'current_boundary_mining':True,'rank_offset_policy':'Round26 identical deterministic rank offsets; item ids may diverge with model state','optimizer':'Adam','learning_rate':float(config['learning_rate']),'weight_decay':wd,'scheduler':list(config['learning_rate_scheduler']),'epochs_trained':len(logical_epochs),'formal_epochs_expected':c.FORMAL_EPOCHS,'warmup_epochs':c.WARMUP_EPOCHS,'lambda_HN':c.LAMBDA_HN,'half_margin':c.HALF_MARGIN,'theta_max_deg':c.THETA_MAX_DEG,'normal_optimizer_steps_total':int(total_steps),'diffusion_optimizer_steps_total':int(diff_steps),'normal_plan_hashes':normal_hashes,'aux_pair_hashes':pair_hashes,'rank_offset_hashes':offset_hashes,'selected_real_item_id_hashes':id_hashes,'lr_trajectory':lr_traj,'checkpoint_selection':'Full CoLiftRec Validation R20','best_epoch':int(best_epoch) if formal else None,'best_full_colift_R20':float(best) if formal else None,'best_evaluation':best_eval,'best_checkpoint':str(best_path) if formal else None,'checkpoint_sha256':c.sha256_file(best_path) if formal else None,'trajectory':trajectory,'mechanism_at_best':mechanism_epochs.get(str(best_epoch)) if formal else mechanism_epochs.get('10'),'RANKING_SCORE_GRAD_TO_DIFFUSION':0,'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
    if formal:
        c.json_write(c.EVID/f'ROUND32_{variant}_SEED{seed}_TRAIN.json',out)
    else:
        c.json_write(c.EVID/f'ROUND32_SMOKE_{variant}_SEED{seed}.json',out)
    del model,ropt,net,dopt; torch.cuda.empty_cache(); gc.collect(); return out
