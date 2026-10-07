from __future__ import annotations
import argparse,json,subprocess,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from diffusion_experiments.modules.round5_common import *
from diffusion_experiments.modules.round5_diffusion_sampler import build_blacklists,make_base_plan,build_selector_cache,choose_from_cache

def schedule(cfg,epoch):
    for x in cfg['continuation']['replacement_schedule']:
        if int(x['start_epoch'])<=epoch<=int(x['end_epoch']): return float(x['probability']),{'low':0,'mid':1,'high':2}[str(x['hardness_bin'])],str(x['hardness_bin'])
    return 0.0,0,'none'
def save_ck(out,epoch,model,opt,meta):
    p=out/'checkpoints'/f'epoch_{epoch:02d}.pt'; torch.save({'epoch':epoch,'state_dict':model.state_dict(),'optimizer':opt.state_dict(),'model_hash':state_hash(model),'meta':meta},p); return p

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--assets',required=True); ap.add_argument('--generator',required=True); ap.add_argument('--out',required=True); ap.add_argument('--seed',type=int,required=True); ap.add_argument('--branch',choices=['B_CONT','D_CURR'],required=True); ap.add_argument('--mode',choices=['smoke','formal'],default='formal'); a=ap.parse_args(); out=Path(a.out)
    if out.exists() and any(out.iterdir()): raise RuntimeError('refuse overwrite nonempty output')
    (out/'checkpoints').mkdir(parents=True,exist_ok=True); (out/'refresh').mkdir(parents=True,exist_ok=True); cfg=cfg_round5(); pdx=ROOT/cfg['protocol_dir']; fit=load_edges(pdx/'fit_edges.csv'); mon=load_edges(pdx/'monitor_edges.csv'); n_users=int(fit.userID.max())+1; n_items=int(np.load(ROOT/'data/baby/text_feat.npy',mmap_mode='r').shape[0]); histories=histories_from_fit(fit,n_users); black=build_blacklists(fit,mon,n_users)
    if a.seed not in [int(x) for x in cfg['formal_seeds']]: raise RuntimeError('unregistered seed')
    torch.manual_seed(a.seed); np.random.seed(a.seed & 0xffffffff)
    teacher_model,teacher_state,mcfg,dl=load_teacher_model(ROOT/cfg['teacher_training_json'],fit); teacher_model.cpu(); del teacher_model; torch.cuda.empty_cache(); model,mcfg,dl=load_student_from_state(teacher_state,fit); initial_hash=state_hash(model); opt=torch.optim.Adam(model.parameters(),lr=float(cfg['continuation']['lr']),weight_decay=float(cfg['continuation']['weight_decay']))
    ev=np.load(Path(a.assets)/'events.npz'); users=ev['users'].astype(np.int64); pos=ev['pos_items'].astype(np.int64); beh=np.load(Path(a.assets)/'teacher_behavior.npz'); observed=beh['observed_items'].astype(np.int32); teacher_cf=beh['standardized'].astype(np.float32); fit_degree=np.bincount(fit.itemID.to_numpy(np.int64),minlength=n_items).astype(np.int64); observed_set=set(map(int,observed.tolist()))
    epochs=1 if a.mode=='smoke' else int(cfg['continuation']['epochs']); N=min(len(users),4096) if a.mode=='smoke' else len(users); active=np.arange(N,dtype=np.int32); users=users[:N]; pos=pos[:N]
    plan=make_base_plan(users,observed,black,a.seed,epochs); np.savez_compressed(out/'base_plan.npz',**plan); plan_hash=sha(out/'base_plan.npz'); side=prepare_side(histories,n_items) if a.branch=='D_CURR' else None
    traj=None
    if a.branch=='D_CURR':
        z=np.load(Path(a.generator)/'trajectory_states.npz');
        if not np.array_equal(z['users'][:N],users.astype(np.int32)) or not np.array_equal(z['pos_items'][:N],pos.astype(np.int32)): raise RuntimeError('trajectory/event identity mismatch')
        traj={s:z[f'x_{s}'][:N].astype(np.float32) for s in [35,25,15]}
    hist=[]; refresh_stats={}; cache=None; replaced_users=set(); replaced_items=set(); total_replaced=0; total_planned=0; total_fallback=0; t0=time.time(); save_ck(out,0,model,opt,{'initial':True,'branch':a.branch,'seed':a.seed})
    for ep in range(1,epochs+1):
        prob,bin_id,bin_name=schedule(cfg,ep); refresh_at=ep-1
        if a.branch=='D_CURR' and refresh_at in [int(x) for x in cfg['continuation']['refresh_epochs']]:
            rs=time.time(); model.eval(); cand=build_candidates(model,histories,side,int(cfg['L_train'])); A=boundary_mask(cand['items'],cand['s0'],cfg); cache,st=build_selector_cache(cand,A,traj,teacher_cf,side['text_unit'],side['visual_unit'],cand['emb']['final_user'],cand['emb']['final_item'],users,pos,black,observed,fit_degree,cfg); st['refresh_epoch']=refresh_at; st['seconds']=time.time()-rs; refresh_stats[str(refresh_at)]=st; np.savez_compressed(out/'refresh'/f'cache_epoch{refresh_at:02d}.npz',**cache); (out/'refresh'/f'stats_epoch{refresh_at:02d}.json').write_text(json.dumps(st,indent=2)+'\n')
        neg=plan['base_neg'][ep-1].copy(); planned=np.zeros(N,bool); actual=np.zeros(N,bool)
        if a.branch=='D_CURR':
            planned=plan['route'][ep-1] < prob; ids=np.flatnonzero(planned); total_planned+=int(len(ids)); sel=choose_from_cache(cache,ids,bin_id,a.seed,ep); ok=sel>=0; chosen_ids=ids[ok]; neg[chosen_ids]=sel[ok]; actual[chosen_ids]=True; total_replaced+=int(ok.sum()); total_fallback+=int((~ok).sum())
            for e,it in zip(chosen_ids,sel[ok]):
                if int(it) in black[int(users[e])] or int(it) not in observed_set: raise RuntimeError('illegal replacement ID')
                replaced_users.add(int(users[e])); replaced_items.add(int(it))
        model.train(); losses=[]; order=plan['order'][ep-1]
        for st in range(0,N,int(cfg['continuation']['batch_size'])):
            ix=order[st:st+int(cfg['continuation']['batch_size'])]; inter=torch.stack([torch.as_tensor(users[ix],device='cuda:0',dtype=torch.long),torch.as_tensor(pos[ix],device='cuda:0',dtype=torch.long),torch.as_tensor(neg[ix],device='cuda:0',dtype=torch.long)]); loss=model.calculate_loss(inter); opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); losses.append(float(loss.detach().cpu()))
        margin=None
        if a.branch=='D_CURR' and actual.any():
            em=export_embeddings(model); ix=np.flatnonzero(actual); ps=np.einsum('bd,bd->b',em['final_user'][users[ix]],em['final_item'][pos[ix]]); ns=np.einsum('bd,bd->b',em['final_user'][users[ix]],em['final_item'][neg[ix]]); margin=float(np.mean(ps-ns))
        rec={'epoch':ep,'loss_mean':float(np.mean(losses)),'requested_probability':prob,'hardness_bin':bin_name,'planned_routes':int(planned.sum()),'actual_replacements':int(actual.sum()),'fallbacks':int(planned.sum()-actual.sum()),'actual_replacement_rate':float(actual.mean()),'actual_bpr_margin_mean':margin,'model_hash':state_hash(model)}; hist.append(rec)
        if ep in [int(x) for x in cfg['continuation']['checkpoints']]: save_ck(out,ep,model,opt,rec)
    status='COMPLETE'; selector_active=None
    if a.branch=='D_CURR':
        selector_active=float(total_replaced/max(total_planned,1)); status='SELECTOR_INACTIVE' if selector_active<0.05 else 'COMPLETE'
    result={'status':status,'mode':a.mode,'branch':a.branch,'seed':a.seed,'initial_model_hash':initial_hash,'final_model_hash':state_hash(model),'epochs':epochs,'events_per_epoch':N,'base_plan_sha256':plan_hash,'history':hist,'refresh_stats':refresh_stats,'total_planned_routes':int(total_planned),'total_actual_replacements':int(total_replaced),'total_fallbacks':int(total_fallback),'selector_nonfallback_fraction':selector_active,'unique_replaced_users':int(len(replaced_users)),'unique_replaced_items':int(len(replaced_items)),'teacher_checkpoint_sha256':cfg['expected_teacher_sha256'],'generator_result_sha256':sha(Path(a.generator)/'result.json') if (Path(a.generator)/'result.json').exists() else None,'elapsed_seconds':time.time()-t0,'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'gpu_name':torch.cuda.get_device_name(0),'access':{'DEV_ACCESSED':False,'INTERNAL_ACCESSED':False,'CONFIRM_ACCESSED':False,'TEST_ACCESSED':False}}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({'status':status,'branch':a.branch,'seed':a.seed,'final_hash':result['final_model_hash'],'replaced':total_replaced,'planned':total_planned,'elapsed':result['elapsed_seconds']},sort_keys=True))
if __name__=='__main__': main()
