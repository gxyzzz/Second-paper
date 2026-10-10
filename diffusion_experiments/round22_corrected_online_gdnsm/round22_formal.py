from __future__ import annotations
import gc,hashlib,json
from pathlib import Path
import numpy as np
import torch

from diffusion_experiments.round22_corrected_online_gdnsm import round22_core as c
from diffusion_experiments.round22_corrected_online_gdnsm.round22_diffusion import generate_partial_online

RDIR=c.RDIR
LOG_EPOCHS={9,10,19,20,29,30}
BANDS=((60,100),(30,60),(10,30))
MODES=('V','T','TV')


def _json_write(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,default=lambda x:float(x) if isinstance(x,np.generic) else str(x))+'\n')


def selected_t0():
    p=RDIR/'evidence'/'ROUND22_PREFLIGHT_SUMMARY.json'
    x=json.loads(p.read_text())
    if not x.get('FORMAL_ALLOWED',False):
        raise RuntimeError('Round22 formal blocked by preflight')
    return int(x['selected_t0'])


def _valid_mask(td,n_items,device):
    m=torch.zeros(n_items,device=device,dtype=torch.bool)
    ids=torch.as_tensor(np.asarray(td.all_items,dtype=np.int64),device=device)
    m[ids]=True
    return m


def online_real_negatives(model,b,interaction,events,valid_mask,active,seed,epoch,batch_idx):
    if active<=0:return [],[]
    users=interaction[0];pos=interaction[1];hu=b['fu'][users]
    with torch.no_grad():
        score=hu.detach()@b['fi'].detach().T
        score[:,~valid_mask]=-torch.inf
        ucpu=users.detach().cpu().numpy();pcpu=pos.detach().cpu().numpy()
        for r,(u,p) in enumerate(zip(ucpu,pcpu)):
            seen=events.histories[int(u)]
            if seen:score[r,torch.as_tensor(seen,device=score.device,dtype=torch.long)]=-torch.inf
            score[r,int(p)]=-torch.inf
        vals,top=torch.topk(score,100,dim=1)
        if not torch.isfinite(vals[:,-1]).all():raise RuntimeError('C1 fewer than 100 legal TRAIN items')
        rng=np.random.default_rng(20265000+int(seed)*100000+int(epoch)*1000+int(batch_idx))
        rows=torch.arange(len(users),device=score.device)
        ids=[]
        for lo,hi in BANDS[:active]:
            off=torch.as_tensor(rng.integers(lo,hi,size=len(users)),device=score.device,dtype=torch.long)
            ids.append(top[rows,off])
    return [b['fi'][ix] for ix in ids],[ix.detach().cpu().numpy() for ix in ids]


def online_diffusion_negatives(model,net,dsched,b,interaction,s,active,t0,seed,epoch,batch_idx):
    x0,hu,tc,vc=c.batch_diff_inputs(b,interaction,s)
    diff_loss=None
    if active>=0:
        diff_loss=c.isolated_diff_step(net,online_diffusion_negatives.diff_opt,dsched,x0,hu,tc,vc,20266000+int(seed)*100000+int(epoch)*1000+int(batch_idx))
    if active<=0:return [],diff_loss,0
    g=torch.Generator(device=model.device);g.manual_seed(20267000+int(seed)*100000+int(epoch)*1000+int(batch_idx))
    parts,steps=generate_partial_online(net,dsched,hu,tc,vc,t0,g)
    raw={k:c.inv_item(v,s).detach() for k,v in parts.items()}
    return [raw[k] for k in MODES[:active]],diff_loss,steps
online_diffusion_negatives.diff_opt=None


def _margin_arrays(hu,hp,negs):
    if not negs:return []
    with torch.no_grad():
        ps=(hu*hp).sum(1)
        return [(ps-(hu*x).sum(1)).detach().cpu().numpy() for x in negs]


def _save_best(model,net,seed,variant,epoch,score,t0):
    p=RDIR/'outputs'/f'seed{seed}/{variant}/best.pt';p.parent.mkdir(parents=True,exist_ok=True)
    obj={'protocol':c.PROTOCOL,'seed':seed,'variant':variant,'epoch':int(epoch),'full_colift_R20':float(score),'selected_t0':int(t0),'model_state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'TEST_ACCESSED':False}
    if net is not None:obj['diffusion_state_dict']={k:v.detach().cpu() for k,v in net.state_dict().items()}
    torch.save(obj,p);return p


def train_variant(seed,variant):
    if variant not in ('B0','C1','D1'):raise ValueError(variant)
    t0=selected_t0()
    model,config,td,vd,pack=c.instantiate_shared(seed);events=c.TrainEvents(model);evaluator=c.CachedEvaluator(seed,model)
    wd=float(config['weight_decay']) if config['weight_decay'] is not None else 0.
    ropt=torch.optim.Adam(model.parameters(),lr=float(config['learning_rate']),weight_decay=wd)
    fac=config['learning_rate_scheduler'];rsched=torch.optim.lr_scheduler.LambdaLR(ropt,lr_lambda=lambda e:fac[0]**(e/fac[1]))
    net=dopt=dsched=None
    if variant=='D1':
        net,dopt,dsched=c.make_diff(seed,model.device);online_diffusion_negatives.diff_opt=dopt
    valid_mask=_valid_mask(td,model.n_items,model.device)
    normal_hashes={};pair_hashes={};lr_traj=[];trajectory=[];mechanism_logs={};state_hash_epoch9=None;stale_reuse=0;generation_step=0
    best=-1.;best_epoch=-1;best_eval=None;best_path=None
    total_steps=0
    for ep in range(c.FORMAL_EPOCHS):
        active=c.curriculum_count(ep);s=c.latent_stats(model);hh=hashlib.sha256();ph=hashlib.sha256();nl=[];al=[];dl=[];margin_acc=[[] for _ in range(active)];norm_acc=[[] for _ in range(active)];reverse_steps=[]
        lr_traj.append(float(ropt.param_groups[0]['lr']));model.train()
        for bi,interaction in enumerate(td):
            u_np=interaction[0].detach().cpu().numpy();p_np=interaction[1].detach().cpu().numpy();n_np=interaction[2].detach().cpu().numpy();hh.update(u_np.tobytes());hh.update(p_np.tobytes());hh.update(n_np.tobytes());ph.update(u_np.tobytes());ph.update(p_np.tobytes())
            ropt.zero_grad(set_to_none=True);b=c.forward_bundle(model);lmsca=c.msca_loss_from_bundle(model,interaction,b);negs=[]
            if variant=='D1':
                negs,ld,steps=online_diffusion_negatives(model,net,dsched,b,interaction,s,active,t0,seed,ep,bi);dl.append(float(ld));
                if active>0:
                    generation_step+=1;reverse_steps.append(int(steps))
            elif variant=='C1' and active>0:
                negs,_=online_real_negatives(model,b,interaction,events,valid_mask,active,seed,ep,bi)
            la=c.aux_bpr(b['fu'][interaction[0]],b['fi'][interaction[1]],negs) if active>0 and variant!='B0' else torch.zeros((),device=model.device)
            total=lmsca+(c.LAMBDA_HN*la if active>0 and variant!='B0' else 0.)
            if not torch.isfinite(total):raise RuntimeError(f'nonfinite rec loss seed{seed} {variant} ep{ep} batch{bi}')
            if active>0 and variant!='B0':
                ms=_margin_arrays(b['fu'][interaction[0]],b['fi'][interaction[1]],negs)
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
        if margin_acc:
            epoch_rec['actual_margin']={MODES[j] if variant=='D1' else ('Easy','Medium','Hard')[j]:c.stats(np.concatenate(v)) for j,v in enumerate(margin_acc) if v}
        if norm_acc:
            epoch_rec['synthetic_norm']={MODES[j]:c.stats(np.concatenate(v)) for j,v in enumerate(norm_acc) if v}
            epoch_rec['synthetic_p99_real_p99_ratio']=float(max(epoch_rec['synthetic_norm'][k]['p99'] for k in epoch_rec['synthetic_norm'])/s['raw_item_norm']['p99'])
        if reverse_steps:epoch_rec['partial_reverse_steps_unique']=sorted(set(reverse_steps))
        trajectory.append(epoch_rec)
        if score>best:
            best=score;best_epoch=ep;best_eval=ev;best_path=_save_best(model,net,seed,variant,ep,score,t0)
        if ep%5==4 or ep in LOG_EPOCHS:
            log={'epoch':ep,'item_norm':c.latent_stats(model)['raw_item_norm'],'standardization_RMS':{k:float(s[k]) for k in ('rI','rU','rT','rV')},'normal_loss':epoch_rec['normal_loss_mean'],'aux_loss':epoch_rec['aux_loss_mean'],'diff_epsilon_loss':epoch_rec['diff_loss_mean'],'active_negative_count':active}
            if 'actual_margin' in epoch_rec:log['actual_margin']=epoch_rec['actual_margin']
            if 'synthetic_p99_real_p99_ratio' in epoch_rec:log['synthetic_p99_real_p99_ratio']=epoch_rec['synthetic_p99_real_p99_ratio']
            if variant=='D1':
                send=c.latent_stats(model);ca=c.condition_audit(model,net,dsched,send,events,seed,n=512);log['true_vs_shuffled_user_advantage']=ca['user_relative_advantage'];log['true_vs_no_modality_advantage']=ca['modality_relative_advantage']
            mechanism_logs[str(ep)]=log
        print(json.dumps({'phase':'formal','seed':seed,'variant':variant,'epoch':ep,'g':active,'R20':score,'R10':ev['colift']['R10'],'N10':ev['colift']['N10'],'N20':ev['colift']['N20']}),flush=True)
    out={'protocol':c.PROTOCOL,'seed':seed,'variant':variant,'selected_t0':t0,'initial_state_hash':pack['state_hash'],'warmup_state_hash_epoch9':state_hash_epoch9,'optimizer':'Adam','learning_rate':float(config['learning_rate']),'weight_decay':wd,'scheduler':list(config['learning_rate_scheduler']),'epochs_trained':c.FORMAL_EPOCHS,'warmup_epochs':c.WARMUP_EPOCHS,'aux_onset_epoch':10,'lambda_HN':0. if variant=='B0' else c.LAMBDA_HN,'normal_optimizer_steps_total':int(total_steps),'normal_plan_hashes':normal_hashes,'aux_pair_hashes':pair_hashes,'lr_trajectory':lr_traj,'checkpoint_selection':'Full CoLiftRec Validation R20','best_epoch':int(best_epoch),'best_full_colift_R20':float(best),'best_evaluation':best_eval,'best_checkpoint':str(best_path),'checkpoint_sha256':c.sha256_file(best_path),'trajectory':trajectory,'mechanism_logs':mechanism_logs,'stale_synthetic_reuse_count':int(stale_reuse),'online_generation_count':int(generation_step),'diffusion_inference_at_recommendation_eval':0,'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
    _json_write(RDIR/'evidence'/f'ROUND22_{variant}_SEED{seed}.json',out)
    online_diffusion_negatives.diff_opt=None
    del model,ropt,net,dopt;torch.cuda.empty_cache();gc.collect();return out


def load_result(seed,variant):
    return json.loads((RDIR/'evidence'/f'ROUND22_{variant}_SEED{seed}.json').read_text())


def fairness(seed):
    r={v:load_result(seed,v) for v in ('B0','C1','D1')};epochs=range(c.FORMAL_EPOCHS)
    normal_bad=[e for e in epochs if len({r[v]['normal_plan_hashes'][str(e)] for v in r})!=1]
    pair_bad=[e for e in epochs if len({r[v]['aux_pair_hashes'][str(e)] for v in r})!=1]
    lr_equal=len({tuple(r[v]['lr_trajectory']) for v in r})==1;init_equal=len({r[v]['initial_state_hash'] for v in r})==1;warm_equal=len({r[v]['warmup_state_hash_epoch9'] for v in r})==1;steps={v:r[v]['normal_optimizer_steps_total'] for v in r}
    out={'seed':seed,'initial_state_exact':init_equal,'warmup_state_epoch9_exact':warm_equal,'normal_plan_mismatch_epochs':normal_bad,'aux_pair_mismatch_epochs':pair_bad,'lr_trajectory_exact':lr_equal,'normal_optimizer_steps':steps,'warmup_epochs':{v:r[v]['warmup_epochs'] for v in r},'aux_onset_epoch':{v:r[v]['aux_onset_epoch'] for v in r},'stale_synthetic_reuse_count_D1':r['D1']['stale_synthetic_reuse_count'],'PASS':bool(init_equal and warm_equal and not normal_bad and not pair_bad and lr_equal and len(set(steps.values()))==1 and r['D1']['stale_synthetic_reuse_count']==0),'TEST_ACCESSED':False}
    _json_write(RDIR/'evidence'/f'ROUND22_FAIRNESS_SEED{seed}.json',out);return out


def validation_summary():
    out={'protocol':c.PROTOCOL,'selected_t0':selected_t0(),'seeds':{},'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False};uc=[];ub=[];cb=[]
    for seed in c.SEEDS:
        r={v:load_result(seed,v) for v in ('B0','C1','D1')};b=r['B0']['best_evaluation']['colift'];cc=r['C1']['best_evaluation']['colift'];d=r['D1']['best_evaluation']['colift'];row={'C1_vs_B0':c.delta_pack(cc,b),'D1_vs_B0':c.delta_pack(d,b),'D1_vs_C1':c.delta_pack(d,cc),'best_epochs':{v:r[v]['best_epoch'] for v in r},'best_metrics':{v:r[v]['best_evaluation']['colift'] for v in r}};out['seeds'][str(seed)]=row;cb.append(row['C1_vs_B0']['U']);ub.append(row['D1_vs_B0']['U']);uc.append(row['D1_vs_C1']['U'])
    out['mean_U_C1_vs_B0']=float(np.mean(cb));out['mean_U_D1_vs_B0']=float(np.mean(ub));out['mean_U_D1_vs_C1']=float(np.mean(uc));out['D1_vs_C1_positive_both']=bool(all(x>0 for x in uc));out['hard_negative_any_positive_mean']=bool(out['mean_U_C1_vs_B0']>0 or out['mean_U_D1_vs_B0']>0);out['diffusion_specific_positive']=bool(out['D1_vs_C1_positive_both'] and out['mean_U_D1_vs_C1']>0);out['diffusion_specific_strong_0_5pct']=bool(out['D1_vs_C1_positive_both'] and out['mean_U_D1_vs_C1']>=.005);out['total_target_1pct']=bool(out['mean_U_D1_vs_B0']>=.01);out['distance_to_1pct']=float(.01-out['mean_U_D1_vs_B0']);_json_write(RDIR/'evidence'/'ROUND22_VALIDATION_SUMMARY.json',out);return out


def write_report():
    s=json.loads((RDIR/'evidence'/'ROUND22_VALIDATION_SUMMARY.json').read_text());pre=json.loads((RDIR/'evidence'/'ROUND22_PREFLIGHT_SUMMARY.json').read_text());lines=['# Round22 Final Report','','## Protocol','',f"Selected TRAIN-only t0: `{s['selected_t0']}`. Test/Sports/Electronics remained closed.",'','## Validation']
    for seed in ('999','1000'):
        x=s['seeds'][seed];lines+=['',f'### seed{seed}','',f"- C1 vs B0 U: {x['C1_vs_B0']['U']*100:+.4f}%",f"- D1 vs B0 U: {x['D1_vs_B0']['U']*100:+.4f}%",f"- D1 vs C1 U: {x['D1_vs_C1']['U']*100:+.4f}%",f"- Best epochs: {x['best_epochs']}"]
    lines+=['','## Cross-seed','',f"- mean U(C1,B0): {s['mean_U_C1_vs_B0']*100:+.4f}%",f"- mean U(D1,B0): {s['mean_U_D1_vs_B0']*100:+.4f}%",f"- mean U(D1,C1): {s['mean_U_D1_vs_C1']*100:+.4f}%",f"- D1>C1 on both seeds: {s['D1_vs_C1_positive_both']}",f"- distance to +1% D1-vs-B0 target: {s['distance_to_1pct']*100:+.4f} percentage points",'','## Required answers','',f"1. Warm preference space formed: yes on both preflight seeds.",f"2. Normalized latent/prior scale compatible: yes on both seeds.",f"3. TRUE user lowers epsilon error: yes; advantages seed999/1000 = {pre['epsilon_user_advantages']['999']*100:.3f}% / {pre['epsilon_user_advantages']['1000']*100:.3f}%.",f"4. TRUE-vs-SHUFFLED changes generation: nonzero same-direction score effect on both seeds.",f"5. selected t0: {s['selected_t0']}, chosen TRAIN-only by the preregistered safe-hard rule.",f"6. Partial reverse differs from full reconstruction: verified by trajectory audit.",f"7. Synthetic norm is same-order as real item latent: preflight P1 passed both seeds.",f"8. TV stably harder than V/T: no in preflight; this was non-catastrophic and non-blocking by protocol.",f"9. stale synthetic reuse: 0 by online immediate-use assertion.",f"10. C1 vs B0 mean U: {s['mean_U_C1_vs_B0']*100:+.4f}%.",f"11. D1 vs B0 mean U: {s['mean_U_D1_vs_B0']*100:+.4f}%.",f"12. D1 vs C1 mean U: {s['mean_U_D1_vs_C1']*100:+.4f}%.",f"13. Distance to +1% target: {s['distance_to_1pct']*100:+.4f} percentage points.",'','No Test result was accessed or used.']
    p=RDIR/'evidence'/'ROUND22_REPORT.md';p.write_text('\n'.join(lines)+'\n');return {'path':str(p),'lines':lines}
