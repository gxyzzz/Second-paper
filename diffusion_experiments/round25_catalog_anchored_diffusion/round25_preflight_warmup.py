from __future__ import annotations
import json
import numpy as np
import torch
from diffusion_experiments.round25_catalog_anchored_diffusion import round25_core as c

def run():
    model,config,td,vd,pack=c.instantiate_shared(999); net,dopt,sched=c.make_diff(999,model.device)
    wd=float(config['weight_decay']) if config['weight_decay'] is not None else 0.
    ropt=torch.optim.Adam(model.parameters(),lr=float(config['learning_rate']),weight_decay=wd)
    fac=config['learning_rate_scheduler']; rsched=torch.optim.lr_scheduler.LambdaLR(ropt,lr_lambda=lambda e:fac[0]**(e/fac[1]))
    rec=[]
    for ep in range(10):
        s=c.latent_stats(model); nl=[]; dl=[]; model.train()
        for bi,interaction in enumerate(td):
            ropt.zero_grad(set_to_none=True); b=c.forward_bundle(model); loss=c.msca_loss_from_bundle(model,interaction,b)
            q=c.base_denoise_step(net,dopt,sched,b,interaction,s,202862000+ep*1000+bi); dl.append(q['rec'])
            loss.backward()
            if config['clip_grad_norm']: torch.nn.utils.clip_grad_norm_(model.parameters(),**config['clip_grad_norm'])
            ropt.step(); nl.append(float(loss.detach()))
        rsched.step(); rec.append({'epoch':ep,'normal_loss':float(np.mean(nl)),'base_rec_mse':float(np.mean(dl))}); print(json.dumps({'phase':'preflight_warmup',**rec[-1]}),flush=True)
    del ropt; torch.cuda.empty_cache(); return model,net,dopt,sched,c.TrainEvents(model),td,rec
