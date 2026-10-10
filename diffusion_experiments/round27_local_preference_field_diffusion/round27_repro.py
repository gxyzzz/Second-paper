from __future__ import annotations
import argparse, gc, hashlib, json
from pathlib import Path
import numpy as np
import torch
from diffusion_experiments.round26_preference_directed_diffusion import round26_core as c26

RDIR=c26.ROOT/'diffusion_experiments/round27_local_preference_field_diffusion'
EVID=RDIR/'evidence'; LOGS=RDIR/'logs'; EVID.mkdir(parents=True,exist_ok=True); LOGS.mkdir(parents=True,exist_ok=True)
PROTOCOL='ROUND27_C1_REPRO_V1'; EPOCHS=16; SEED=999

def write(p,o): Path(p).write_text(json.dumps(o,indent=2)+'\n')

def run(tag):
    model,config,td,vd,pack=c26.instantiate_shared(SEED); events=c26.TrainEvents(model); evaluator=c26.CachedEvaluator(SEED,model)
    wd=float(config['weight_decay']) if config['weight_decay'] is not None else 0.
    opt=torch.optim.Adam(model.parameters(),lr=float(config['learning_rate']),weight_decay=wd)
    fac=config['learning_rate_scheduler']; sched=torch.optim.lr_scheduler.LambdaLR(opt,lr_lambda=lambda e:fac[0]**(e/fac[1]))
    mask=c26.valid_mask(td,model.n_items,model.device)
    nh={}; oh={}; ih={}; traj=[]
    for ep in range(EPOCHS):
        active=c26.curriculum_count(ep); hh=hashlib.sha256(); ho=hashlib.sha256(); hi=hashlib.sha256(); model.train()
        for bi,interaction in enumerate(td):
            for x in interaction:
                hh.update(x.detach().cpu().numpy().tobytes())
            offs=c26.rank_offsets(SEED,ep,bi,len(interaction[0]),active)
            for x in offs: ho.update(np.asarray(x,dtype=np.int64).tobytes())
            opt.zero_grad(set_to_none=True); b=c26.forward_bundle(model); loss=c26.msca_loss_from_bundle(model,interaction,b)
            if active>0:
                ids,used,_=c26.mine_current_ids(b,interaction,events,mask,SEED,ep,bi,active)
                for x in ids: hi.update(x.detach().cpu().numpy().astype(np.int64).tobytes())
                for a,z in zip(offs,used):
                    if not np.array_equal(a,z): raise RuntimeError('rank-offset mismatch')
                loss=loss+c26.LAMBDA_HN*c26.aux_bpr(b['fu'][interaction[0]],b['fi'][interaction[1]],[b['fi'][x] for x in ids])
            loss.backward()
            if config['clip_grad_norm']: torch.nn.utils.clip_grad_norm_(model.parameters(),**config['clip_grad_norm'])
            opt.step()
        sched.step(); nh[str(ep)]=hh.hexdigest(); oh[str(ep)]=ho.hexdigest(); ih[str(ep)]=hi.hexdigest()
        ev=evaluator.evaluate(model); r20=float(ev['colift']['R20']); traj.append({'epoch':ep,'active_negative_count':active,'R20':r20})
        print(json.dumps({'tag':tag,'epoch':ep,'active':active,'R20':r20}),flush=True)
    out={'protocol':PROTOCOL,'tag':tag,'seed':SEED,'epochs':EPOCHS,'initial_state_hash':pack['state_hash'],'normal_plan_hashes':nh,'rank_offset_hashes':oh,'selected_real_item_hashes':ih,'trajectory':traj,'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
    write(EVID/f'ROUND27_C1_REPRO_{tag}.json',out)
    del model,opt; torch.cuda.empty_cache(); gc.collect(); return out

def audit():
    a=json.load(open(EVID/'ROUND27_C1_REPRO_A.json')); b=json.load(open(EVID/'ROUND27_C1_REPRO_B.json'))
    r20a=[x['R20'] for x in a['trajectory']]; r20b=[x['R20'] for x in b['trajectory']]
    out={'protocol':PROTOCOL,'seed':SEED,'initial_state_exact':a['initial_state_hash']==b['initial_state_hash'],'normal_plan_exact':a['normal_plan_hashes']==b['normal_plan_hashes'],'rank_offset_exact':a['rank_offset_hashes']==b['rank_offset_hashes'],'selected_real_item_hashes_exact':a['selected_real_item_hashes']==b['selected_real_item_hashes'],'r20_exact':r20a==r20b,'r20_max_abs_diff':float(max(abs(x-y) for x,y in zip(r20a,r20b))),'mismatch_epochs':{'selected_real_item':[e for e in range(EPOCHS) if a['selected_real_item_hashes'][str(e)]!=b['selected_real_item_hashes'][str(e)]],'R20':[e for e in range(EPOCHS) if r20a[e]!=r20b[e]]},'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
    out['C1_REPRODUCIBILITY']='PASS' if all(out[k] for k in ('initial_state_exact','normal_plan_exact','rank_offset_exact','selected_real_item_hashes_exact','r20_exact')) else 'FAIL'
    write(EVID/'ROUND27_BOUNDARY_REPRO_AUDIT.json',out); print(json.dumps(out,indent=2)); return out

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('cmd',choices=['A','B','audit']); z=ap.parse_args()
    audit() if z.cmd=='audit' else run(z.cmd)
