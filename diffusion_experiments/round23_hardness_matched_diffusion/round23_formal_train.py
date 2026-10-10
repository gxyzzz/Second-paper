from __future__ import annotations
import gc,hashlib,json
import numpy as np
import torch
from diffusion_experiments.round23_hardness_matched_diffusion import round23_core as c
from diffusion_experiments.round23_hardness_matched_diffusion import round23_formal as f


def train_variant(seed,variant):
    if variant not in c.VARIANTS: raise ValueError(variant)
    model,config,td,vd,pack=c.instantiate_shared(seed);events=c.TrainEvents(model);evaluator=c.CachedEvaluator(seed,model)
    wd=float(config['weight_decay']) if config['weight_decay'] is not None else 0.
    ropt=torch.optim.Adam(model.parameters(),lr=float(config['learning_rate']),weight_decay=wd)
    fac=config['learning_rate_scheduler'];rsched=torch.optim.lr_scheduler.LambdaLR(ropt,lr_lambda=lambda e:fac[0]**(e/fac[1]))
    net=dopt=dsched=None
    if variant=='D1HM': net,dopt,dsched=c.make_diff(seed,model.device)
    mask=c.valid_mask(td,model.n_items,model.device)
    shared_plan=None;building_plan={}
    if variant!='B0': shared_plan=c.load_real_plan(seed)
    normal_hashes={};pair_hashes={};lr_traj=[];trajectory=[];mechanism_logs={};total_steps=0;generation_count=0;stale_reuse=0
    best=-1.;best_epoch=-1;best_eval=None;best_path=None
    for ep in range(c.FORMAL_EPOCHS):
        active=c.curriculum_count(ep);s=c.latent_stats(model);hh=hashlib.sha256();ph=hashlib.sha256();nl=[];al=[];dl=[];match_batches=[];real_margin=[[] for _ in range(active)];building_plan[ep]=[]
        lr_traj.append(float(ropt.param_groups[0]['lr']));model.train()
        for bi,interaction in enumerate(td):
            u_np=interaction[0].detach().cpu().numpy();p_np=interaction[1].detach().cpu().numpy();n_np=interaction[2].detach().cpu().numpy();hh.update(u_np.tobytes());hh.update(p_np.tobytes());hh.update(n_np.tobytes());ph.update(u_np.tobytes());ph.update(p_np.tobytes())
            ropt.zero_grad(set_to_none=True);b=c.forward_bundle(model);lmsca=c.msca_loss_from_bundle(model,interaction,b)
            if variant=='B0':
                ids=c.mine_real_ids(b,interaction,events,mask,seed,ep,bi,active=max(active,0)) if active>0 else []
                building_plan[ep].append([x.detach().cpu().numpy().astype(np.int32) for x in ids])
            else:
                ids=c.plan_ids(shared_plan,ep,bi,model.device,active) if active>0 else []
                building_plan[ep].append([x.detach().cpu().numpy().astype(np.int32) for x in ids])
            negs=[];la=torch.zeros((),device=model.device)
            if variant=='C1FULL' and active>0:
                negs=c.real_negs(b,ids,detach=False);la=c.aux_bpr(b['fu'][interaction[0]],b['fi'][interaction[1]],negs)
            elif variant=='C1DETACH' and active>0:
                negs=c.real_negs(b,ids,detach=True);la=c.aux_bpr(b['fu'][interaction[0]],b['fi'][interaction[1]],negs)
            elif variant=='D1HM':
                x0,hu,tc,vc=c.batch_diff_inputs(b,interaction,s);ld=c.isolated_diff_step(net,dopt,dsched,x0,hu,tc,vc,20273000+int(seed)*100000+ep*1000+bi);dl.append(ld)
                if active>0:
                    negs,diag=c.hardness_match(model,net,dsched,b,interaction,s,ids,seed,ep,bi);match_batches.append(diag);generation_count+=1
                    la=c.aux_bpr(b['fu'][interaction[0]],b['fi'][interaction[1]],negs)
            if active>0 and variant in ('C1FULL','C1DETACH'):
                with torch.no_grad():
                    ps=(b['fu'][interaction[0]]*b['fi'][interaction[1]]).sum(1)
                    for j,x in enumerate(negs): real_margin[j].append((ps-(b['fu'][interaction[0]]*x).sum(1)).detach().cpu().numpy())
            total=lmsca+(c.LAMBDA_HN*la if active>0 and variant!='B0' else 0.)
            if not torch.isfinite(total): raise RuntimeError(f'nonfinite rec loss seed{seed} {variant} ep{ep} batch{bi}')
            total.backward()
            if config['clip_grad_norm']: torch.nn.utils.clip_grad_norm_(model.parameters(),**config['clip_grad_norm'])
            ropt.step();nl.append(float(lmsca.detach()));al.append(float(la.detach()) if active>0 and variant!='B0' else 0.);total_steps+=1
        rsched.step();normal_hashes[str(ep)]=hh.hexdigest();pair_hashes[str(ep)]=ph.hexdigest()
        ev=evaluator.evaluate(model);score=float(ev['colift']['R20'])
        rec={'epoch':ep,'active_negative_count':active,'normal_loss_mean':float(np.mean(nl)),'aux_loss_mean':float(np.mean(al)),'diff_loss_mean':None if not dl else float(np.mean(dl)),'lr':lr_traj[-1],'evaluation':ev,'full_colift_R20':score}
        if any(real_margin): rec['real_negative_margin']={('Easy','Medium','Hard')[j]:c.stats(np.concatenate(v)) for j,v in enumerate(real_margin) if v}
        if match_batches: rec['hardness_match']=c.summarize_match(match_batches)
        trajectory.append(rec)
        if score>best:
            best=score;best_epoch=ep;best_eval=ev;best_path=f.save_best(model,net,seed,variant,ep,score)
        if ep%5==4 or ep in f.LOG_EPOCHS:
            log={'epoch':ep,'normal_loss':rec['normal_loss_mean'],'aux_loss':rec['aux_loss_mean'],'diff_epsilon_loss':rec['diff_loss_mean'],'active_negative_count':active,'item_norm':c.latent_stats(model)['raw_item_norm']}
            if 'real_negative_margin' in rec: log['real_negative_margin']=rec['real_negative_margin']
            if 'hardness_match' in rec: log['hardness_match']=rec['hardness_match']
            if variant=='D1HM':
                send=c.latent_stats(model);ca=c.condition_audit(model,net,dsched,send,events,seed,n=512);log['epsilon_mse']=ca['mse'];log['user_epsilon_advantage']=ca['user_relative_advantage'];log['modality_epsilon_advantage']=ca['modality_relative_advantage']
            mechanism_logs[str(ep)]=log
        print(json.dumps({'phase':'formal','seed':seed,'variant':variant,'epoch':ep,'g':active,'R20':score,'R10':ev['colift']['R10'],'N10':ev['colift']['N10'],'N20':ev['colift']['N20']}),flush=True)
    if variant=='B0': plan_obj=c.save_real_plan(seed,building_plan)
    else:
        dig=c.plan_digest(building_plan)
        if dig!=shared_plan['digest']: raise RuntimeError(f'real-negative ID plan mismatch seed{seed} {variant}')
        plan_obj={'digest':dig}
    out={'protocol':c.PROTOCOL,'shared_miner_resolution':'B0 online boundary miner fixes exact negative IDs across C1FULL/C1DETACH/D1HM; target margin/norm are recomputed in each current variant model','seed':seed,'variant':variant,'initial_state_hash':pack['state_hash'],'optimizer':'Adam','learning_rate':float(config['learning_rate']),'weight_decay':wd,'scheduler':list(config['learning_rate_scheduler']),'epochs_trained':c.FORMAL_EPOCHS,'warmup_epochs':c.WARMUP_EPOCHS,'aux_onset_epoch':10,'lambda_HN':0. if variant=='B0' else c.LAMBDA_HN,'normal_optimizer_steps_total':int(total_steps),'normal_plan_hashes':normal_hashes,'aux_pair_hashes':pair_hashes,'lr_trajectory':lr_traj,'real_negative_plan_digest':plan_obj['digest'],'checkpoint_selection':'Full CoLiftRec Validation R20','best_epoch':int(best_epoch),'best_full_colift_R20':float(best),'best_evaluation':best_eval,'best_checkpoint':str(best_path),'checkpoint_sha256':c.sha256_file(best_path),'trajectory':trajectory,'mechanism_logs':mechanism_logs,'stale_synthetic_reuse_count':int(stale_reuse),'online_generation_count':int(generation_count),'diffusion_inference_at_recommendation_eval':0,'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
    c.json_write(c.EVID/f'ROUND23_{variant}_SEED{seed}.json',out)
    if variant=='D1HM':
        hm={'protocol':c.PROTOCOL,'seed':seed,'best_epoch':best_epoch,'epochs':{str(x['epoch']):x['hardness_match'] for x in trajectory if 'hardness_match' in x},'TEST_ACCESSED':False};c.json_write(c.EVID/f'ROUND23_HARDNESS_MATCH_SEED{seed}.json',hm)
    del model,ropt,net,dopt;torch.cuda.empty_cache();gc.collect();return out
