from __future__ import annotations
import gc,hashlib,json
import numpy as np
import torch
from diffusion_experiments.round22_corrected_online_gdnsm import round22_core as c
from diffusion_experiments.round22_corrected_online_gdnsm import round22_formal as f


def train_variant(seed,variant):
    if variant not in ('B0','C1','D1'):raise ValueError(variant)
    t0=f.selected_t0()
    model,config,td,vd,pack=c.instantiate_shared(seed);events=c.TrainEvents(model);evaluator=c.CachedEvaluator(seed,model)
    wd=float(config['weight_decay']) if config['weight_decay'] is not None else 0.
    ropt=torch.optim.Adam(model.parameters(),lr=float(config['learning_rate']),weight_decay=wd)
    fac=config['learning_rate_scheduler'];rsched=torch.optim.lr_scheduler.LambdaLR(ropt,lr_lambda=lambda e:fac[0]**(e/fac[1]))
    net=dopt=dsched=None
    if variant=='D1':
        net,dopt,dsched=c.make_diff(seed,model.device);f.online_diffusion_negatives.diff_opt=dopt
    valid_mask=f._valid_mask(td,model.n_items,model.device)
    normal_hashes={};pair_hashes={};lr_traj=[];trajectory=[];mechanism_logs={};state_hash_epoch9=None;stale_reuse=0;generation_step=0
    best=-1.;best_epoch=-1;best_eval=None;best_path=None;total_steps=0
    for ep in range(c.FORMAL_EPOCHS):
        active=c.curriculum_count(ep);s=c.latent_stats(model);hh=hashlib.sha256();ph=hashlib.sha256();nl=[];al=[];dl=[];margin_acc=[[] for _ in range(active)];norm_acc=[[] for _ in range(active)];reverse_steps=[]
        lr_traj.append(float(ropt.param_groups[0]['lr']));model.train()
        for bi,interaction in enumerate(td):
            u_np=interaction[0].detach().cpu().numpy();p_np=interaction[1].detach().cpu().numpy();n_np=interaction[2].detach().cpu().numpy();hh.update(u_np.tobytes());hh.update(p_np.tobytes());hh.update(n_np.tobytes());ph.update(u_np.tobytes());ph.update(p_np.tobytes())
            ropt.zero_grad(set_to_none=True);b=c.forward_bundle(model);lmsca=c.msca_loss_from_bundle(model,interaction,b);negs=[]
            if variant=='D1':
                negs,ld,steps=f.online_diffusion_negatives(model,net,dsched,b,interaction,s,active,t0,seed,ep,bi);dl.append(float(ld))
                if active>0:generation_step+=1;reverse_steps.append(int(steps))
            elif variant=='C1' and active>0:
                negs,_=f.online_real_negatives(model,b,interaction,events,valid_mask,active,seed,ep,bi)
            la=c.aux_bpr(b['fu'][interaction[0]],b['fi'][interaction[1]],negs) if active>0 and variant!='B0' else torch.zeros((),device=model.device)
            total=lmsca+(c.LAMBDA_HN*la if active>0 and variant!='B0' else 0.)
            if not torch.isfinite(total):raise RuntimeError(f'nonfinite rec loss seed{seed} {variant} ep{ep} batch{bi}')
            if active>0 and variant!='B0':
                ms=f._margin_arrays(b['fu'][interaction[0]],b['fi'][interaction[1]],negs)
                for j,m in enumerate(ms):margin_acc[j].append(m)
                if variant=='D1':
                    for j,x in enumerate(negs):norm_acc[j].append(x.norm(dim=1).detach().cpu().numpy())
            total.backward()
            if config['clip_grad_norm']:torch.nn.utils.clip_grad_norm_(model.parameters(),**config['clip_grad_norm'])
            ropt.step();nl.append(float(lmsca.detach()));al.append(float(la.detach()) if active>0 and variant!='B0' else 0.);total_steps+=1
        rsched.step();normal_hashes[str(ep)]=hh.hexdigest();pair_hashes[str(ep)]=ph.hexdigest()
        if ep==9:state_hash_epoch9=c.state_hash(model.state_dict())
        ev=evaluator.evaluate(model);score=float(ev['colift']['R20'])
        epoch_rec={'epoch':ep,'active_negative_count':active,'normal_loss_mean':float(np.mean(nl)),'aux_loss_mean':float(np.mean(al)),'diff_loss_mean':None if not dl else float(np.mean(dl)),'lr':lr_traj[-1],'evaluation':ev,'full_colift_R20':score}
        if any(margin_acc):
            epoch_rec['actual_margin']={(f.MODES[j] if variant=='D1' else ('Easy','Medium','Hard')[j]):dict(c.stats(np.concatenate(v)),p_negative_ge_positive=float((np.concatenate(v)<=0).mean())) for j,v in enumerate(margin_acc) if v}
        if variant=='D1' and any(norm_acc):
            epoch_rec['synthetic_norm']={f.MODES[j]:c.stats(np.concatenate(v)) for j,v in enumerate(norm_acc) if v}
            epoch_rec['synthetic_p99_real_p99_ratio']=float(max(v['p99'] for v in epoch_rec['synthetic_norm'].values())/s['raw_item_norm']['p99'])
        if reverse_steps:epoch_rec['partial_reverse_steps_unique']=sorted(set(reverse_steps))
        trajectory.append(epoch_rec)
        if score>best:
            best=score;best_epoch=ep;best_eval=ev;best_path=f._save_best(model,net,seed,variant,ep,score,t0)
        if ep%5==4 or ep in f.LOG_EPOCHS:
            log={'epoch':ep,'item_norm':c.latent_stats(model)['raw_item_norm'],'standardization_RMS':{k:float(s[k]) for k in ('rI','rU','rT','rV')},'normal_loss':epoch_rec['normal_loss_mean'],'aux_loss':epoch_rec['aux_loss_mean'],'diff_epsilon_loss':epoch_rec['diff_loss_mean'],'active_negative_count':active}
            if 'actual_margin' in epoch_rec:log['actual_margin']=epoch_rec['actual_margin']
            if 'synthetic_p99_real_p99_ratio' in epoch_rec:log['synthetic_p99_real_p99_ratio']=epoch_rec['synthetic_p99_real_p99_ratio']
            if variant=='D1':
                send=c.latent_stats(model);ca=c.condition_audit(model,net,dsched,send,events,seed,n=512);log['epsilon_mse']=ca['mse'];log['true_vs_shuffled_user_advantage']=ca['user_relative_advantage'];log['true_vs_no_modality_advantage']=ca['modality_relative_advantage']
            mechanism_logs[str(ep)]=log
        print(json.dumps({'phase':'formal','seed':seed,'variant':variant,'epoch':ep,'g':active,'R20':score,'R10':ev['colift']['R10'],'N10':ev['colift']['N10'],'N20':ev['colift']['N20']}),flush=True)
    out={'protocol':c.PROTOCOL,'seed':seed,'variant':variant,'selected_t0':t0,'initial_state_hash':pack['state_hash'],'warmup_state_hash_epoch9':state_hash_epoch9,'optimizer':'Adam','learning_rate':float(config['learning_rate']),'weight_decay':wd,'scheduler':list(config['learning_rate_scheduler']),'epochs_trained':c.FORMAL_EPOCHS,'warmup_epochs':c.WARMUP_EPOCHS,'aux_onset_epoch':10,'lambda_HN':0. if variant=='B0' else c.LAMBDA_HN,'normal_optimizer_steps_total':int(total_steps),'normal_plan_hashes':normal_hashes,'aux_pair_hashes':pair_hashes,'lr_trajectory':lr_traj,'checkpoint_selection':'Full CoLiftRec Validation R20','best_epoch':int(best_epoch),'best_full_colift_R20':float(best),'best_evaluation':best_eval,'best_checkpoint':str(best_path),'checkpoint_sha256':c.sha256_file(best_path),'trajectory':trajectory,'mechanism_logs':mechanism_logs,'stale_synthetic_reuse_count':int(stale_reuse),'online_generation_count':int(generation_step),'diffusion_inference_at_recommendation_eval':0,'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
    f._json_write(c.RDIR/'evidence'/f'ROUND22_{variant}_SEED{seed}.json',out);f.online_diffusion_negatives.diff_opt=None
    del model,ropt,net,dopt;torch.cuda.empty_cache();gc.collect();return out
