from __future__ import annotations
import argparse,json,subprocess,sys,time
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'src'))
from diffusion_experiments.modules.round6r1_train_common import *

def install_capture(model):
    cap={}; ob=model.cal_bpr_loss; oc=model.cal_cl_loss; orr=model.cal_reg_loss
    def bpr(*x):
        v=ob(*x); cap['bpr']=float(v.detach().cpu()); return v
    def cl(*x):
        v=oc(*x); cap.setdefault('cl',[]).append(float(v.detach().cpu())); return v
    def reg(*x):
        v=orr(*x); cap['reg']=float(v.detach().cpu()); return v
    model.cal_bpr_loss=bpr; model.cal_cl_loss=cl; model.cal_reg_loss=reg
    return cap

def save_best(path,epoch,model,opt,sched,best,cur,meta):
    torch.save({'epoch':int(epoch),'state_dict':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'best_valid_score':float(best),'cur_step':int(cur),'model_hash':state_hash(model),'meta':meta},path)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--branch',choices=['B','D'],required=True)
    ap.add_argument('--plan',required=True)
    ap.add_argument('--risk')
    ap.add_argument('--out',required=True)
    ap.add_argument('--mode',choices=['smoke','formal'],default='formal')
    a=ap.parse_args(); out=Path(a.out)
    if out.exists() and any(out.iterdir()): raise RuntimeError('refuse overwrite nonempty output')
    (out/'checkpoints').mkdir(parents=True,exist_ok=True)
    c=cfg_r1(); tcfg=c['training']; pdx=ROOT/c['protocol_dir']
    fit=load_edges(pdx/'fit_edges.csv'); mon=load_edges(pdx/'monitor_edges.csv')
    histories=histories_from_fit(fit,int(fit.userID.max())+1)
    planp=Path(a.plan); z=np.load(planp)
    U=z['users'].astype(np.int32); P=z['pos'].astype(np.int32); Neg=z['neg'].astype(np.int32)
    maxep=min(U.shape[0],2 if a.mode=='smoke' else int(tcfg['max_epochs']))
    N=min(U.shape[1],4096) if a.mode=='smoke' else U.shape[1]
    if a.seed not in [int(x) for x in c['student_seeds']]: raise RuntimeError('unregistered student seed')
    model,mcfg,_=fresh_student(a.seed,fit,out)
    initial_hash=state_hash(model); cap=install_capture(model)
    opt=torch.optim.Adam(model.parameters(),lr=float(tcfg['lr']),weight_decay=float(tcfg['weight_decay']))
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lr_lambda=lambda ep: float(tcfg['scheduler'][0])**(ep/float(tcfg['scheduler'][1])))
    items=A=W=valid=None; risk_sha=None
    if a.branch=='D':
        if not a.risk: raise RuntimeError('D requires risk cache')
        rp=Path(a.risk); rz=np.load(rp)
        items=rz['items'].astype(np.int32); A=rz['A_mask'].astype(bool)
        W=rz['weights'].astype(np.float32); valid=rz['valid_query'].astype(bool); risk_sha=sha(rp)
        ref=np.load(ROOT/c['round6_assets']/'reference_L100.npz')
        if not np.array_equal(items,ref['items']) or not np.array_equal(A,ref['A_mask']):
            raise RuntimeError('risk/reference identity mismatch')
        obs=set(map(int,fit.itemID.unique().tolist()))
        for u,h in enumerate(histories):
            hs=set(map(int,h)); ids=items[u,np.flatnonzero(A[u])]
            if any(int(x) in hs or int(x) not in obs for x in ids):
                raise RuntimeError(f'illegal A item user={u}')
    histsets=[set(map(int,h)) for h in histories]
    for ep in range(maxep):
        for u,n in zip(U[ep,:N],Neg[ep,:N]):
            if int(n) in histsets[int(u)]: raise RuntimeError('base negative intersects FIT history')

    best=-1.0; cur=0; history=[]; best_path=out/'checkpoints'/'best.pt'
    start=time.time(); batch_size=int(tcfg['batch_size']); aux_seen=0; aux_nonzero=0
    for ep0 in range(maxep):
        epoch=ep0+1; model.train()
        sums={k:0.0 for k in ['loss','bpr','cl','weighted_cl','reg','weighted_reg','visual_reg','l0','lA','coeff_event']}
        batches=0; event_aux_w=[]; event_coeff=[]
        users=U[ep0,:N]; pos=P[ep0,:N]; neg=Neg[ep0,:N]
        aux_ids=aux_cols=None
        if a.branch=='D':
            aux_ids,aux_cols=keyed_choice_positions(A,items,users,a.seed,epoch,np.arange(N,dtype=np.int64))
        for st in range(0,N,batch_size):
            en=min(st+batch_size,N); ub=users[st:en]; pb=pos[st:en]; nb=neg[st:en]
            inter=torch.stack([torch.as_tensor(ub,device='cuda:0',dtype=torch.long),torch.as_tensor(pb,device='cuda:0',dtype=torch.long),torch.as_tensor(nb,device='cuda:0',dtype=torch.long)])
            opt.zero_grad(set_to_none=True); cap.clear()
            beta=0.0 if epoch<=int(tcfg['beta_warmup_epochs']) or a.branch=='B' else float(tcfg['beta'])
            if beta==0.0:
                loss=model.calculate_loss(inter)
                parts={'bpr':torch.tensor(cap['bpr']),'cl':torch.tensor(sum(cap['cl'])),'reg':torch.tensor(cap['reg']),'weighted_cl':torch.tensor(model.cl_weight*sum(cap['cl'])),'weighted_reg':torch.tensor(model.reg_weight*cap['reg']),'l0':torch.tensor(cap['bpr']),'lA':torch.tensor(float('nan')),'coeff_event':torch.tensor(0.0)}
            else:
                jb=aux_ids[st:en].copy(); cb=aux_cols[st:en].copy(); ok=(jb>=0)&valid[ub]
                ww=np.zeros(en-st,np.float32)
                rows=np.flatnonzero(ok)
                if len(rows):
                    ww[rows]=W[ub[rows],cb[rows]]
                    finite=np.isfinite(ww); ww[~finite]=0.0; ok &= finite
                jb[~ok]=nb[~ok]; ww[~ok]=0.0
                jt=torch.as_tensor(jb,device='cuda:0',dtype=torch.long); wt=torch.as_tensor(ww,device='cuda:0')
                loss,parts=auxiliary_loss(model,inter,jt,wt,beta)
                aux_seen += len(ww); aux_nonzero += int((ww>0).sum())
                event_aux_w.extend(ww.tolist()); event_coeff.extend((beta*ww/(1+beta*ww)).tolist())
            vr=float(visual_reg(model).detach().cpu())
            loss.backward(); opt.step(); batches+=1
            sums['loss']+=float(loss.detach().cpu())
            for k in ['bpr','cl','weighted_cl','reg','weighted_reg','l0','coeff_event']:
                sums[k]+=float(parts[k].detach().cpu())
            sums['lA']+=0.0 if not torch.isfinite(parts['lA']) else float(parts['lA'].detach().cpu())
            sums['visual_reg']+=vr
        sched.step()
        monitor,_=high_precision_monitor(model,fit,mon,histories); score=float(monitor['R20'])
        improved=score>best
        if improved: best=score; cur=0
        else: cur+=1
        rec={'epoch':epoch,'monitor':{k:float(v) for k,v in monitor.items()},'monitor_R20':score,'improved':improved,'cur_step':cur,'lr':float(opt.param_groups[0]['lr']),'components':{k:float(v/batches) for k,v in sums.items()},'beta':0.0 if epoch<=int(tcfg['beta_warmup_epochs']) or a.branch=='B' else float(tcfg['beta'])}
        if event_aux_w:
            rec['aux']={'nonzero_fraction':float(np.mean(np.asarray(event_aux_w)>0)),'mean_w':float(np.mean(event_aux_w)),'mean_effective_coefficient':float(np.mean(event_coeff)),'max_effective_coefficient':float(np.max(event_coeff))}
        history.append(rec)
        if improved: save_best(best_path,epoch,model,opt,sched,best,cur,rec)
        if cur>int(tcfg['early_stopping']): break
    best_state=torch.load(best_path,map_location='cpu',weights_only=False)
    result={'status':'COMPLETE','mode':a.mode,'branch':a.branch,'seed':a.seed,'initial_model_hash':initial_hash,'epochs_ran':len(history),'best_epoch':int(best_state['epoch']),'best_monitor_R20':float(best_state['best_valid_score']),'best_checkpoint':str(best_path.resolve()),'best_checkpoint_sha256':sha(best_path),'best_model_hash':best_state['model_hash'],'base_plan_sha256':sha(planp),'risk_cache_sha256':risk_sha,'history':history,'aux_total_events_after_warmup':int(aux_seen),'aux_nonzero_events':int(aux_nonzero),'elapsed_seconds':time.time()-start,'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'gpu_name':torch.cuda.get_device_name(0),'access':{'MONITOR_USED_FOR_SELECTION':True,'DEV':False,'INTERNAL':False,'CONFIRM':False,'TEST':False}}
    if a.branch=='B' and a.seed==999 and a.mode=='formal':
        ref=float(tcfg['teacher_monitor_R20'])
        result['baseline_reproduction']={'teacher_R20':ref,'relative_change':float((result['best_monitor_R20']-ref)/ref),'pass':result['best_monitor_R20']>=ref*(1-float(tcfg['baseline_reproduction_max_relative_drop']))}
        if not result['baseline_reproduction']['pass']: result['status']='BASE_REPRODUCTION_MISMATCH'
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'branch':a.branch,'seed':a.seed,'epochs':result['epochs_ran'],'best_epoch':result['best_epoch'],'best_R20':result['best_monitor_R20'],'aux_nonzero':result['aux_nonzero_events'],'elapsed':result['elapsed_seconds']},sort_keys=True))
if __name__=='__main__': main()
