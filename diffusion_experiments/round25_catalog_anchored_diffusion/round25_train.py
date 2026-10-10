from __future__ import annotations
import gc,hashlib,json
import numpy as np
import torch
from diffusion_experiments.round25_catalog_anchored_diffusion import round25_core as c


def train_variant(seed,variant):
    if variant not in c.VARIANTS: raise ValueError(variant)
    model,config,td,vd,pack=c.instantiate_shared(seed); events=c.TrainEvents(model); evaluator=c.CachedEvaluator(seed,model)
    wd=float(config['weight_decay']) if config['weight_decay'] is not None else 0.
    ropt=torch.optim.Adam(model.parameters(),lr=float(config['learning_rate']),weight_decay=wd)
    fac=config['learning_rate_scheduler']; rsched=torch.optim.lr_scheduler.LambdaLR(ropt,lr_lambda=lambda e:fac[0]**(e/fac[1]))
    net=dopt=dsched=None
    if variant=='D2': net,dopt,dsched=c.make_diff(seed,model.device)
    mask=c.valid_mask(td,model.n_items,model.device)
    normal_hashes={}; pair_hashes={}; offset_hashes={}; id_hashes={}; lr_traj=[]; trajectory=[]; mechanism_epochs={}; total_steps=0; diff_steps=0
    best=-1.; best_epoch=-1; best_eval=None; best_path=None
    for ep in range(c.FORMAL_EPOCHS):
        active=c.curriculum_count(ep); s=c.latent_stats(model); hh=hashlib.sha256(); ph=hashlib.sha256(); oh=hashlib.sha256(); ih=hashlib.sha256(); nl=[]; al=[]; dl=[]; mechanism_batches=[]
        lr_traj.append(float(ropt.param_groups[0]['lr'])); model.train()
        for bi,interaction in enumerate(td):
            u_np=interaction[0].detach().cpu().numpy(); p_np=interaction[1].detach().cpu().numpy(); n_np=interaction[2].detach().cpu().numpy()
            hh.update(u_np.tobytes()); hh.update(p_np.tobytes()); hh.update(n_np.tobytes()); ph.update(u_np.tobytes()); ph.update(p_np.tobytes())
            offs=c.rank_offsets(seed,ep,bi,len(interaction[0]),active)
            for x in offs: oh.update(np.asarray(x,dtype=np.int64).tobytes())
            ropt.zero_grad(set_to_none=True); b=c.forward_bundle(model); lmsca=c.msca_loss_from_bundle(model,interaction,b); ids=[]; la=torch.zeros((),device=model.device)
            if active>0 and variant!='B0':
                ids,used,_=c.mine_current_ids(b,interaction,events,mask,seed,ep,bi,active)
                for x in ids: ih.update(x.detach().cpu().numpy().astype(np.int64).tobytes())
                for a,z in zip(offs,used):
                    if not np.array_equal(a,z): raise RuntimeError('rank-offset policy mismatch')
            if variant=='C1FULL' and active>0:
                negs=[b['fi'][x] for x in ids]; la=c.aux_bpr(b['fu'][interaction[0]],b['fi'][interaction[1]],negs)
            elif variant=='D2':
                if active==0:
                    q=c.base_denoise_step(net,dopt,dsched,b,interaction,s,202850000+seed*100000+ep*1000+bi); dl.append(q); diff_steps+=1
                else:
                    for inner in range(c.DIFF_STEPS_ACTIVE):
                        q=c.diffusion_inner_step(net,dopt,dsched,b,interaction,s,ids,202851000+seed*100000+ep*10000+bi*10+inner); dl.append(q); diff_steps+=1
                    deltas,diag=c.generate_refinement(net,dsched,b,interaction,s,ids,202852000+seed*100000+ep*1000+bi); mechanism_batches.append(diag)
                    negs=[b['fi'][ids[j]]+deltas[j] for j in range(active)]
                    if any(x.requires_grad for x in deltas): raise RuntimeError('Delta_diff must be detached for recommender')
                    la=c.aux_bpr(b['fu'][interaction[0]],b['fi'][interaction[1]],negs)
            total=lmsca+(c.LAMBDA_HN*la if active>0 and variant!='B0' else 0.)
            if not torch.isfinite(total): raise RuntimeError(f'nonfinite rec loss seed{seed} {variant} ep{ep} batch{bi}')
            total.backward()
            if config['clip_grad_norm']: torch.nn.utils.clip_grad_norm_(model.parameters(),**config['clip_grad_norm'])
            ropt.step(); nl.append(float(lmsca.detach())); al.append(float(la.detach()) if active>0 and variant!='B0' else 0.); total_steps+=1
        rsched.step(); normal_hashes[str(ep)]=hh.hexdigest(); pair_hashes[str(ep)]=ph.hexdigest(); offset_hashes[str(ep)]=oh.hexdigest(); id_hashes[str(ep)]=ih.hexdigest()
        ev=evaluator.evaluate(model); score=float(ev['colift']['R20'])
        rec={'epoch':ep,'active_negative_count':active,'normal_loss_mean':float(np.mean(nl)),'aux_loss_mean':float(np.mean(al)),'diff_loss_mean':None if not dl else float(np.mean([x['total'] for x in dl])),'lr':lr_traj[-1],'evaluation':ev,'full_colift_R20':score}
        if variant=='D2':
            if active>0:
                mech=c.summarize_mechanism(mechanism_batches,dl); rec['mechanism']=mech; mechanism_epochs[str(ep)]=mech
                if not mech['angle_bound_pass']: raise RuntimeError(f'5-degree trust region broken seed{seed} ep{ep}')
            else:
                rec['warmup_diffusion']={k:c.stats(np.asarray([x[k] for x in dl])) for k in ('total','rec')}
        trajectory.append(rec)
        if score>best:
            best=score; best_epoch=ep; best_eval=ev; best_path=c.save_best(model,net,seed,variant,ep,score)
        print(json.dumps({'phase':'formal','seed':seed,'variant':variant,'epoch':ep,'g':active,'R20':score,'R10':ev['colift']['R10'],'N10':ev['colift']['N10'],'N20':ev['colift']['N20'],'G_hard':None if 'mechanism' not in rec else rec['mechanism']['G_hard']['median'],'G_user':None if 'mechanism' not in rec else rec['mechanism']['G_user']['median']}),flush=True)
    out={'protocol':c.PROTOCOL,'protocol_hash':c.protocol_hash(),'seed':seed,'variant':variant,'initial_state_hash':pack['state_hash'],'current_boundary_mining':variant!='B0','rank_offset_policy':'same deterministic rank offsets; item ids may diverge with model state','optimizer':'Adam','learning_rate':float(config['learning_rate']),'weight_decay':wd,'scheduler':list(config['learning_rate_scheduler']),'epochs_trained':c.FORMAL_EPOCHS,'warmup_epochs':c.WARMUP_EPOCHS,'aux_onset_epoch':10,'lambda_HN':0. if variant=='B0' else c.LAMBDA_HN,'normal_optimizer_steps_total':int(total_steps),'diffusion_optimizer_steps_total':int(diff_steps),'normal_plan_hashes':normal_hashes,'aux_pair_hashes':pair_hashes,'rank_offset_hashes':offset_hashes,'selected_real_item_id_hashes':id_hashes,'lr_trajectory':lr_traj,'checkpoint_selection':'Full CoLiftRec Validation R20','best_epoch':int(best_epoch),'best_full_colift_R20':float(best),'best_evaluation':best_eval,'best_checkpoint':str(best_path),'checkpoint_sha256':c.sha256_file(best_path),'trajectory':trajectory,'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
    c.json_write(c.EVID/f'ROUND25_{variant}_SEED{seed}.json',out)
    if variant=='D2':
        c.json_write(c.EVID/f'ROUND25_MECHANISM_SEED{seed}.json',{'protocol':c.PROTOCOL,'seed':seed,'best_epoch':best_epoch,'epochs':mechanism_epochs,'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False})
    del model,ropt,net,dopt; torch.cuda.empty_cache(); gc.collect(); return out
