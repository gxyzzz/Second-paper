from __future__ import annotations
import gc, hashlib, json
import numpy as np
import torch
from diffusion_experiments.round29_positive_preference_purification import round29_core as c
from diffusion_experiments.round29_positive_preference_purification.round29_colift_reference import FrozenCoLiftReference
from diffusion_experiments.round29_positive_preference_purification.round29_positive_field import PositiveFieldBank, dynamic_field
from diffusion_experiments.round29_positive_preference_purification import round29_diffusion as d

def train_d6(seed,epochs=None,formal=True):
    epochs=c.FORMAL_EPOCHS if epochs is None else int(epochs)
    model,config,td,vd,pack=c.instantiate_shared(seed); events=c.TrainEvents(model); evaluator=c.CachedEvaluator(seed,model)
    ref=FrozenCoLiftReference(seed,model,events); bank=PositiveFieldBank(seed,model,events,ref)
    net,dopt,dsched=c.make_diff(seed,model.device)
    wd=float(config['weight_decay']) if config['weight_decay'] is not None else 0.; ropt=torch.optim.Adam(model.parameters(),lr=float(config['learning_rate']),weight_decay=wd)
    fac=config['learning_rate_scheduler']; rsched=torch.optim.lr_scheduler.LambdaLR(ropt,lr_lambda=lambda e:fac[0]**(e/fac[1]))
    normal_hashes={}; pair_hashes={}; lr_traj=[]; trajectory=[]; total_steps=0; diff_steps=0
    best=-1.; best_epoch=-1; best_eval=None; best_path=None
    for ep in range(epochs):
        bank.reset_epoch(); s=c.latent_stats(model); hh=hashlib.sha256(); ph=hashlib.sha256(); nl=[]; pl=[]; dl=[]; frows=[]; drows=[]; grows=[]; resolved=0
        lr_traj.append(float(ropt.param_groups[0]['lr'])); model.train()
        for bi,interaction in enumerate(td):
            u_np=interaction[0].detach().cpu().numpy(); p_np=interaction[1].detach().cpu().numpy(); n_np=interaction[2].detach().cpu().numpy()
            hh.update(u_np.tobytes()); hh.update(p_np.tobytes()); hh.update(n_np.tobytes()); ph.update(u_np.tobytes()); ph.update(p_np.tobytes())
            eids=bank.resolve_event_ids(interaction[0],interaction[1]); resolved+=len(eids); static=bank.batch_static(eids,model.device)
            ropt.zero_grad(set_to_none=True); b=c.forward_bundle(model); lmsca=c.msca_loss_from_bundle(model,interaction,b); lpur=torch.zeros((),device=model.device)
            if ep<c.WARMUP_EPOCHS:
                q=d.base_warmup_step(net,dopt,dsched,b,interaction,s,202900000+int(seed)*100000+ep*1000+bi); dl.append(q); diff_steps+=1
            else:
                field=dynamic_field(b,interaction[1],static); packs=[field]; target=[interaction[1]]
                frows.append(field['diag'])
                for inner in range(c.DIFF_STEPS_ACTIVE):
                    q=d.field_inner_step(net,dopt,dsched,b,interaction,s,target,packs,202901000+int(seed)*100000+ep*10000+bi*10+inner); dl.append(q); diff_steps+=1
                dirs,ddiag=d.predict_ensemble_directions(net,dsched,b,interaction,s,target,packs); drows.extend(ddiag)
                teacher,gdiag=d.purify_positive(b['fu'][interaction[0]],b['fi'][interaction[1]],dirs[0],field['confidence']); gdiag['g_field']=(b['fu'][interaction[0]].detach()*field['direction']).sum(1).detach().cpu().numpy(); gdiag['g_diff']=(b['fu'][interaction[0]].detach()*dirs[0]).sum(1).detach().cpu().numpy(); grows.append(gdiag)
                if teacher.requires_grad: raise RuntimeError('purified positive teacher must be detached')
                lpur=d.purification_loss(b['fu'][interaction[0]],b['fi'][interaction[2]],teacher,field['confidence'])
            total=lmsca+(c.LAMBDA_PUR*lpur if ep>=c.WARMUP_EPOCHS else 0.)
            if not torch.isfinite(total): raise RuntimeError(f'nonfinite D6 loss seed{seed} ep{ep} batch{bi}')
            total.backward()
            if config['clip_grad_norm']: torch.nn.utils.clip_grad_norm_(model.parameters(),**config['clip_grad_norm'])
            ropt.step(); nl.append(float(lmsca.detach())); pl.append(float(lpur.detach())); total_steps+=1
        rsched.step(); normal_hashes[str(ep)]=hh.hexdigest(); pair_hashes[str(ep)]=ph.hexdigest()
        ev=evaluator.evaluate(model); score=float(ev['colift']['R20'])
        rec={'epoch':ep,'normal_loss_mean':float(np.mean(nl)),'L_pur_mean':float(np.mean(pl)),'diff_loss_mean':float(np.mean([x['total'] for x in dl])),
             'lr':lr_traj[-1],'evaluation':ev,'full_colift_R20':score,'resolved_event_count':int(resolved)}
        if ep>=c.WARMUP_EPOCHS:
            rec['mechanism']=d.summarize(frows,drows,grows,dl)
            if not rec['mechanism']['angle_bound_pass']: raise RuntimeError('positive purification angle bound broken')
            if not rec['mechanism']['norm_preservation_pass']: raise RuntimeError('positive teacher norm preservation broken')
        trajectory.append(rec)
        if score>best:
            best=score; best_epoch=ep; best_eval=ev
            if formal: best_path=c.save_best(model,net,seed,ep,score)
        msg={'phase':'formal' if formal else 'smoke','seed':seed,'variant':'D6','epoch':ep,'R20':score,'R10':ev['colift']['R10'],'N10':ev['colift']['N10'],'N20':ev['colift']['N20'],'L_pur':rec['L_pur_mean']}
        if ep>=c.WARMUP_EPOCHS:
            m=rec['mechanism']; msg.update({'A_true':m['A_true']['median'],'DeltaA':m['DeltaA']['median'],'P_DeltaA_gt0':m['DeltaA']['p_gt_zero'],'g_diff_score_delta':m['score_delta']['median'],'angle':m['angle_deg']['median'],'conf':m['field']['confidence']['median']})
        print(json.dumps(msg),flush=True)
    out={'protocol':c.PROTOCOL,'protocol_hash':c.protocol_hash(),'seed':int(seed),'variant':'D6','initial_state_hash':pack['state_hash'],
         'optimizer':'Adam','learning_rate':float(config['learning_rate']),'weight_decay':wd,'scheduler':list(config['learning_rate_scheduler']),
         'epochs_trained':int(epochs),'warmup_epochs':c.WARMUP_EPOCHS,'lambda_PUR':c.LAMBDA_PUR,'normal_optimizer_steps_total':int(total_steps),
         'diffusion_optimizer_steps_total':int(diff_steps),'normal_plan_hashes':normal_hashes,'aux_pair_hashes':pair_hashes,'lr_trajectory':lr_traj,
         'checkpoint_selection':'Full CoLiftRec Validation R20','best_epoch':int(best_epoch),'best_full_colift_R20':float(best),'best_evaluation':best_eval,
         'best_checkpoint':None if not formal else str(best_path),'checkpoint_sha256':None if not formal else c.sha256_file(best_path),'trajectory':trajectory,
         'baseline':c.baseline_record(seed),'field_bank_hash':bank.bank_hash(),'colift_reference':ref.provenance_audit(),
         'NORMAL_NEGATIVE_SOURCE':'ORIGINAL_TRAIN_DATALOADER','EXTRA_NEGATIVE_ADDED':False,'BOUNDARY_NEGATIVE_USED':False,'FROZEN_HARD_NEGATIVE_USED':False,'DIFFUSION_NEGATIVE_USED':False,
         'diffusion_inference_at_recommendation_eval':0,'ranking_gradient_to_diffusion':0,
         'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
    if formal: c.json_write(c.EVID/f'ROUND29_D6_SEED{seed}.json',out)
    del model,net,dopt,ropt,evaluator,ref,bank; torch.cuda.empty_cache(); gc.collect(); return out
