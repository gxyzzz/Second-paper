from __future__ import annotations
import gc,json
import numpy as np
import torch
from diffusion_experiments.round26_preference_directed_diffusion import round26_core as c
from diffusion_experiments.round26_preference_directed_diffusion import round26_diffusion as d


def _cal_summary(batches):
    rows=[r for batch in batches for r in batch]; cat=lambda k:np.concatenate([r[k] for r in rows])
    return {'positive_margin_fraction':float(cat('positive_margin').mean()),'already_hard_no_refine_fraction':float(cat('already_hard_no_refine').mean()),'direction_positive_fraction':float(cat('direction_positive').mean()),'direction_miss_fraction':float(cat('direction_miss').mean()),'applied_refinement_fraction':float(cat('applied_refinement').mean()),'target_reachable_fraction':float(cat('target_reachable').mean()),'target_hit_abs_error':c.stats(cat('target_hit_abs_error')),'cap_hit_fraction':float(cat('cap_hit').mean()),'angle_deg':c.stats(cat('angle_deg')),'requested_score_shift':c.stats(cat('requested_score_shift')),'actual_score_shift':c.stats(cat('actual_score_shift')),'target_never_exceeds_positive':bool(cat('target_never_exceeds_positive').all())}


def _oracle_dirs(b,interaction,ids):
    pos=b['fi'][interaction[1]].detach(); return [d.preference_direction(pos,b['fi'][x].detach())[0] for x in ids]


def _calibrate(b,interaction,ids,dirs):
    u=b['fu'][interaction[0]].detach(); p=b['fi'][interaction[1]].detach(); out=[]; deltas=[]
    for j,x in enumerate(ids):
        de,di=d.calibrate(u,p,b['fi'][x].detach(),dirs[j]); di['band']=c.BAND_NAMES[j]; out.append(di); deltas.append(de)
    return deltas,out


def run():
    obj=json.loads((c.EVID/'ROUND26_OBJECTIVE_SANITY_AUDIT.json').read_text()); smoke=json.loads((c.EVID/'ROUND26_SMOKE.json').read_text()); grad=json.loads((c.EVID/'ROUND26_GRADIENT_AUDIT.json').read_text())
    model,config,td,vd,pack=c.instantiate_shared(999); events=c.TrainEvents(model); mask=c.valid_mask(td,model.n_items,model.device); net,dopt,dsched=c.make_diff(999,model.device)
    wd=float(config['weight_decay']) if config['weight_decay'] is not None else 0.; ropt=torch.optim.Adam(model.parameters(),lr=float(config['learning_rate']),weight_decay=wd); fac=config['learning_rate_scheduler']; rsched=torch.optim.lr_scheduler.LambdaLR(ropt,lr_lambda=lambda e:fac[0]**(e/fac[1]))
    warm=[]
    for ep in range(10):
        s=c.latent_stats(model); nl=[]; dl=[]; model.train()
        for bi,interaction in enumerate(td):
            ropt.zero_grad(set_to_none=True); b=c.forward_bundle(model); loss=c.msca_loss_from_bundle(model,interaction,b); q=d.warmup_step(net,dopt,dsched,b,interaction,s,202872000+ep*1000+bi); loss.backward()
            if config['clip_grad_norm']: torch.nn.utils.clip_grad_norm_(model.parameters(),**config['clip_grad_norm'])
            ropt.step(); nl.append(float(loss.detach())); dl.append(q['rec'])
        rsched.step(); warm.append({'epoch':ep,'normal_loss':float(np.mean(nl)),'L_rec':float(np.mean(dl))}); print(json.dumps({'phase':'round26_preflight_warmup',**warm[-1]}),flush=True)
    u,p=events.sample(1024,20287300); n=events.random_unobserved(u,p,20287301); chunks=[]
    for st in range(0,len(u),128): chunks.append([torch.as_tensor(x[st:st+128],device=model.device) for x in (u,p,n)])
    s=c.latent_stats(model); model.eval()
    updates=[]
    for step in range(250):
        interaction=chunks[step%len(chunks)]; b=c.forward_bundle(model); ids,_,_=c.mine_current_ids(b,interaction,events,mask,999,10,step,1); q=d.direction_inner_step(net,dopt,dsched,b,interaction,s,ids,202874000+step); updates.append(q)
    dbatches=[]; cbatches=[]; obatches=[]
    for bi,interaction in enumerate(chunks):
        b=c.forward_bundle(model); ids,_,_=c.mine_current_ids(b,interaction,events,mask,999,10,1000+bi,1)
        dirs,ddiag,_=d.predict_directions(net,dsched,b,interaction,s,ids,202875000+bi); _,cdiag=_calibrate(b,interaction,ids,dirs); dbatches.append(ddiag); cbatches.append(cdiag)
        odirs=_oracle_dirs(b,interaction,ids); _,odiag=_calibrate(b,interaction,ids,odirs); obatches.append(odiag)
    d3=d.summarize(dbatches,cbatches,updates); oracle=_cal_summary(obatches)
    finite=all(np.isfinite(x) for x in (d3['A_true']['median'],d3['DeltaA']['median'],d3['raw_residual_norm']['median'],d3['angle_deg']['max'],oracle['angle_deg']['max']))
    reasons=[]
    if not obj['PASS']: reasons.append('OBJECTIVE_SANITY_FAIL')
    if not smoke['PASS']: reasons.append('SMOKE_FAIL')
    if not grad['PASS'] or grad['ranking_gradient_to_diffusion_abs_sum']!=0: reasons.append('GRADIENT_TOPOLOGY_OR_LEAK_FAIL')
    if not finite: reasons.append('NAN_INF')
    if not d3['angle_bound_pass'] or oracle['angle_deg']['max']>5.01: reasons.append('ANGLE_BOUND_FAIL')
    if d3['near_zero_residual_fraction']>=.99: reasons.append('DIFFUSION_RESIDUAL_NUMERICALLY_ZERO')
    if not oracle['target_never_exceeds_positive'] or not np.isfinite(oracle['target_hit_abs_error']['median']): reasons.append('ORACLE_INTERFACE_NUMERIC_FAIL')
    out={'protocol':c.PROTOCOL,'seed':999,'TRAIN_ONLY':True,'events':1024,'online_diffusion_updates':250,'warmup':warm,'D3_direction_and_calibration':d3,'O1_oracle_calibration':oracle,'objective_sanity_pass':obj['PASS'],'smoke_pass':smoke['PASS'],'gradient_audit_pass':grad['PASS'],'RANKING_SCORE_GRAD_TO_DIFFUSION':grad['RANKING_SCORE_GRAD_TO_DIFFUSION'],'PREFLIGHT_PASS':not reasons,'BLOCK_REASONS':reasons,'FORMAL_ALLOWED':not reasons,'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
    c.json_write(c.EVID/'ROUND26_PREFLIGHT.json',out); del model,net,dopt,ropt; torch.cuda.empty_cache(); gc.collect(); return out
