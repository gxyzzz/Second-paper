from __future__ import annotations
import gc,json
import numpy as np
import torch
from diffusion_experiments.round27_local_preference_field_diffusion import round27_core as c
from diffusion_experiments.round27_local_preference_field_diffusion import round27_lpf as lpf


def run():
    repro=json.loads((c.EVID/'ROUND27_BOUNDARY_REPRO_AUDIT.json').read_text()); target=json.loads((c.EVID/'ROUND27_TARGET_STABILITY_AUDIT.json').read_text()); smoke=json.loads((c.EVID/'ROUND27_SMOKE.json').read_text()); grad=json.loads((c.EVID/'ROUND27_GRADIENT_AUDIT.json').read_text())
    model,config,td,vd,pack=c.instantiate_shared(999); events=c.TrainEvents(model); mask=c.valid_mask(td,model.n_items,model.device); ctx=lpf.LPFContext(999,model); net,dopt,dsched=c.make_diff(999,model.device)
    wd=float(config['weight_decay']) if config['weight_decay'] is not None else 0.; ropt=torch.optim.Adam(model.parameters(),lr=float(config['learning_rate']),weight_decay=wd); fac=config['learning_rate_scheduler']; rsched=torch.optim.lr_scheduler.LambdaLR(ropt,lr_lambda=lambda e:fac[0]**(e/fac[1]))
    warm=[]
    for ep in range(c.WARMUP_EPOCHS):
        s=c.latent_stats(model); nl=[]; dl=[]; model.train()
        for bi,interaction in enumerate(td):
            ropt.zero_grad(set_to_none=True); b=c.forward_bundle(model); loss=c.msca_loss_from_bundle(model,interaction,b); q=lpf.warmup_step(net,dopt,dsched,b,interaction,s,202883000+ep*1000+bi); loss.backward()
            if config['clip_grad_norm']: torch.nn.utils.clip_grad_norm_(model.parameters(),**config['clip_grad_norm'])
            ropt.step(); nl.append(float(loss.detach())); dl.append(q['rec'])
        rsched.step(); warm.append({'epoch':ep,'normal_loss':float(np.mean(nl)),'L_rec':float(np.mean(dl))}); print(json.dumps({'phase':'round27_preflight_warmup',**warm[-1]}),flush=True)
    u,p=events.sample(1024,20288400); n=events.random_unobserved(u,p,20288401); chunks=[]
    for st in range(0,len(u),128): chunks.append([torch.as_tensor(x[st:st+128],device=model.device,dtype=torch.long) for x in (u,p,n)])
    s=c.latent_stats(model); updates=[]
    for step in range(250):
        interaction=chunks[step%len(chunks)]; b=c.forward_bundle(model); ids,_,_=c.mine_current_ids_stable(b,interaction,events,mask,999,10,step,1); packs=ctx.targets(b,interaction[0],interaction[1],ids); q=lpf.field_inner_step(net,dopt,dsched,b,interaction,s,ids,packs,202885000+step); updates.append(q)
    pb=[]; db=[]; cb=[]
    for bi,interaction in enumerate(chunks):
        b=c.forward_bundle(model); ids,_,_=c.mine_current_ids_stable(b,interaction,events,mask,999,10,1000+bi,1); packs=ctx.targets(b,interaction[0],interaction[1],ids); dirs,ddi=lpf.predict_directions(net,dsched,b,interaction,s,ids,packs,202886000+bi)
        user=b['fu'][interaction[0]].detach(); pos=b['fi'][interaction[1]].detach(); cdiag=[]
        for j,x in enumerate(ids):
            _,di=lpf.calibrate(user,pos,b['fi'][x].detach(),dirs[j],packs[j]['confidence']); di['band']=c.BAND_NAMES[j]; cdiag.append(di)
        pb.append(packs); db.append(ddi); cb.append(cdiag)
    mech=lpf.summarize_mechanism(pb,db,cb,updates)
    vals=[mech['LPF']['rho']['median'],mech['LPF']['confidence']['median'],mech['A_true']['median'],mech['DeltaA']['median'],mech['angle_deg']['max'],mech['L_rec']['median'],mech['L_field']['median']]; finite=bool(np.isfinite(vals).all())
    reasons=[]
    if repro.get('C1_REPRODUCIBILITY')!='PASS': reasons.append('C1_REPRODUCIBILITY_UNRESOLVED')
    if not target.get('PASS'): reasons.append('TARGET_AUDIT_IMPLEMENTATION_FAIL')
    if target.get('VALIDATION_HISTORY_ACCESSED') is not False or target.get('TEST_ACCESSED') is not False: reasons.append('HISTORY_LEAKAGE')
    if not smoke.get('PASS'): reasons.append('SMOKE_FAIL')
    if not grad.get('PASS') or grad.get('RANKING_SCORE_GRAD_TO_DIFFUSION')!=0: reasons.append('GRADIENT_TOPOLOGY_OR_LEAK_FAIL')
    if not finite: reasons.append('NAN_INF')
    if not mech['angle_bound_pass']: reasons.append('ANGLE_BOUND_BROKEN')
    if mech['near_zero_residual_fraction']>=.99: reasons.append('DIFFUSION_RESIDUAL_NUMERICALLY_ZERO')
    if mech['LPF']['confidence']['min'] < -1e-7 or mech['LPF']['confidence']['max'] > 1+1e-7: reasons.append('CONFIDENCE_OUT_OF_RANGE')
    out={'protocol':c.PROTOCOL,'seed':999,'TRAIN_ONLY':True,'events':1024,'online_diffusion_updates':250,'warmup':warm,'mechanism':mech,'C1_reproducibility':repro.get('C1_REPRODUCIBILITY'),'target_stability':target.get('bootstrap_stability'),'smoke_pass':smoke.get('PASS'),'gradient_audit_pass':grad.get('PASS'),'RANKING_SCORE_GRAD_TO_DIFFUSION':grad.get('RANKING_SCORE_GRAD_TO_DIFFUSION'),'PREFLIGHT_PASS':not reasons,'BLOCK_REASONS':reasons,'FORMAL_ALLOWED':not reasons,'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
    c.json_write(c.EVID/'ROUND27_PREFLIGHT.json',out); del model,net,dopt,ropt,ctx; torch.cuda.empty_cache(); gc.collect(); return out
