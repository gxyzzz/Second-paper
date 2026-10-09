from __future__ import annotations
import copy,gc,hashlib,json,math,random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[2]
RDIR=ROOT/'diffusion_experiments/round20_positive_anchored_hardneg'
PROTOCOL='ROUND20_PA_HCDHN_V1'
SEEDS=(999,1000)
LAMBDA_HARD=.20
Q_STEP=24
TOPK=5
NOISE_SEEDS=(20261801,20261802)
AUX_SELECT_BASE=20262001
AUX_ORDER_BASE=20262101
ALL=('R10','N10','R20','N20','R50','N50')
PRIMARY=('R10','N10','R20','N20')

from models.msca import MSCA
from utils.dataset import RecDataset
from utils.dataloader import TrainDataLoader,EvalDataLoader
from common.trainer import Trainer
from utils.utils import init_seed
from pipelines.dataset_config import load_dataset_config
from pipelines.msca_assets import build_train_histories_and_validation
from pipelines.coliftrec import _params,fit_backgrounds
from modules.ranking import candidate_dot_scores,sha256_file
from diffusion_experiments.round18_diffusion_hard_negative.round18_core import HCDDiffuser,DiffusionSchedule,load_start
from diffusion_experiments.round19_safe_aux_hardneg.round19_core import round19_loss,CachedEvaluator,delta_pack,stats,load_generator
from diffusion_experiments.round16r_cleantrain.clean_context import pseudo_asset_path
from diffusion_experiments.round16_fcbrd import colift_context as cctx

def seed_all(s):
    init_seed(int(s))

def state_hash(sd):
    h=hashlib.sha256()
    for k in sorted(sd):
        v=sd[k].detach().cpu().contiguous()
        h.update(k.encode());h.update(str(v.dtype).encode())
        h.update(np.asarray(v.shape,dtype=np.int64).tobytes());h.update(v.numpy().tobytes())
    return h.hexdigest()

def capture_rng():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}

def restore_rng(x):
    random.setstate(x['python']);np.random.set_state(x['numpy']);torch.set_rng_state(x['torch'])
    if torch.cuda.is_available() and x['cuda']: torch.cuda.set_rng_state_all(x['cuda'])

def utility(m,b):
    return float(np.mean([(m[k]-b[k])/b[k] for k in PRIMARY]))

def tensor3(model,u,p,n):
    d=model.device
    return (torch.as_tensor(u,device=d,dtype=torch.long),torch.as_tensor(p,device=d,dtype=torch.long),torch.as_tensor(n,device=d,dtype=torch.long))

def build_prefix_safe_pool(seed):
    cfg=load_dataset_config('baby');paths=cfg['resolved_paths']
    audit,c=cctx._load_model_components(seed);n_users=int(audit['n_users']);n_items=int(audit['n_items'])
    histories,prefixes,pusers,_,_=build_train_histories_and_validation(paths['interaction'],n_users)
    users=np.asarray([u for u,h in enumerate(histories) if len(h)>=3],dtype=np.int64)
    targets=np.asarray([histories[int(u)][-1] for u in users],dtype=np.int32)
    prefix=[prefixes[int(u)] for u in users]
    z=np.load(pseudo_asset_path(seed));zu=z['users'].astype(np.int64)
    if not np.array_equal(zu,pusers): raise RuntimeError('canonical pseudo users mismatch')
    row={int(u):i for i,u in enumerate(zu)};rr=np.asarray([row[int(u)] for u in users],np.int64)
    cand=z['items'][rr].astype(np.int32);msca_c=z['scores'][rr].astype(np.float32)
    target2=targets[:,None]
    msca_e=candidate_dot_scores(c['final_user'],c['final_item'],users,target2)
    ztr,zte_raw=cctx._semantic_raw(paths['text_feature'],prefix,users,cand,target2,256)
    zvr,zve_raw=cctx._semantic_raw(paths['visual_feature'],prefix,users,cand,target2,64)
    zt=cctx._z_ref(ztr,ztr);zte=cctx._z_ref(zte_raw,ztr)
    zv=cctx._z_ref(zvr,zvr);zve=cctx._z_ref(zve_raw,zvr)
    ar,are=cctx._attribute_raw(cfg,prefix,users,cand,target2)
    za,zae=cctx._attribute_z_from_raw(ar,are,cfg['coliftrec']['attribute'].get('weights'))
    bg_raw=fit_backgrounds(cand,zt,za,zv,n_items)
    bg={k:v['shrunk_mean'] for k,v in bg_raw.items()}
    _,full,order,_,full_e=cctx._contexts(msca_c,cand,zt,za,zv,bg,_params(cfg['coliftrec']),extra=(msca_e,target2,zte,zae,zve))
    ranked=np.take_along_axis(cand,order,axis=1).astype(np.int32)
    rfull=np.take_along_axis(full,order,axis=1).astype(np.float32)
    rmsca=np.take_along_axis(msca_c,order,axis=1).astype(np.float32)
    safe=ranked[:,20:100].copy();safe_full=rfull[:,20:100].copy();safe_bb=rmsca[:,20:100].copy();mask=np.ones(safe.shape,bool)
    prefix_collision=0;target_collision=0
    for r,u in enumerate(users):
        seen=set(map(int,prefix[r]));t=int(targets[r])
        bad=np.asarray([(int(x) in seen) or (int(x)==t) or int(x)<=0 for x in safe[r]],bool)
        prefix_collision+=sum(int(x) in seen for x in safe[r]);target_collision+=sum(int(x)==t for x in safe[r]);mask[r,bad]=False
    counts=mask.sum(1)
    if int(counts.min())<TOPK: raise RuntimeError('safe pool has fewer than Top5 legal items')
    out=RDIR/'assets'/f'seed{seed}_prefix_safe_pool.npz'
    np.savez_compressed(out,users=users,target=targets,ranked_items=ranked,ranked_full_scores=rfull,ranked_backbone_scores=rmsca,
        safe_items=safe,safe_mask=mask,safe_full_scores=safe_full,safe_backbone_scores=safe_bb,target_backbone_score=msca_e[:,0].astype(np.float32),target_full_score=full_e[:,0].astype(np.float32))
    rng=np.random.default_rng(20262062+seed);sample=rng.choice(len(users),size=min(1000,len(users)),replace=False)
    prefix_mismatch=0;target_in_prefix=0
    for r in sample:
        u=int(users[r]);p=list(histories[u][:-1]);target_in_prefix+=int(int(targets[r]) in set(p));prefix_mismatch+=int(p!=prefix[r])
    audit_out={'seed':seed,'users':int(len(users)),'retrieval_source':str(pseudo_asset_path(seed)),'retrieval_prefix_mask':'h[:-1]',
      'colift_profile_history':'h[:-1]','background_fit_history':'all clean h[:-1]','safe_pool_rank':[21,100],
      'protected_rank':[1,20],'legal_count':stats(counts),'prefix_observed_hits_before_filter':int(prefix_collision),'positive_target_hits_before_filter':int(target_collision),
      'sampled_prefix_alignment_users':int(len(sample)),'prefix_alignment_mismatch':int(prefix_mismatch),'target_in_prefix':int(target_in_prefix),
      'future_train_event_count':0,'validation_used':False,'test_used':False,'sha256':sha256_file(out),'TEST_ACCESSED':False}
    return audit_out

def _load_repr(seed,device):
    z=np.load(ROOT/f'diffusion_experiments/round18_diffusion_hard_negative/assets/seed{seed}_frozen_repr.npz')
    return {k:torch.as_tensor(z[k],device=device) for k in z.files}

def _pack_prefixes(histories,users,source_users,device,max_len=20):
    arr=np.zeros((len(users),max_len),np.int64);mask=np.zeros((len(users),max_len),bool)
    for r,su in enumerate(source_users):
        p=list(histories[int(su)][:-1][-max_len:]);arr[r,:len(p)]=p;mask[r,:len(p)]=True
    return torch.as_tensor(arr,device=device),torch.as_tensor(mask,device=device)

def _reverse_q75(net,sched,reps,x31,hids,hmask,reverse_seed):
    g=torch.Generator(device=x31.device);g.manual_seed(int(reverse_seed));x=x31.clone()
    for step,ti in enumerate(range(31,7,-1),start=1):
        t=torch.full((len(x),),ti,device=x.device,dtype=torch.long)
        pred=net(x,t,hids,hmask,reps['id_item'],reps['visual_item'],reps['text_item'])
        mean=sched.extract(sched.pm1,t,x)*pred+sched.extract(sched.pm2,t,x)*x
        if ti==0: x=mean
        else: x=mean+torch.sqrt(sched.extract(sched.pvar,t,x))*torch.randn(x.shape,generator=g,device=x.device)
    if step!=Q_STEP: raise RuntimeError('Q75 reverse step mismatch')
    return x

@torch.no_grad()
def build_positive_anchored_queries(seed,batch=256,limit=None):
    device='cuda:0';pool=np.load(RDIR/'assets'/f'seed{seed}_prefix_safe_pool.npz');users=pool['users'].astype(np.int64);targets=pool['target'].astype(np.int64)
    if limit is not None: users=users[:limit];targets=targets[:limit]
    _,ck,ds,tr,a=load_start(seed,True);cfg=load_dataset_config('baby');histories,_,_,_,_=build_train_histories_and_validation(cfg['resolved_paths']['interaction'],int(a['n_users']))
    reps=_load_repr(seed,device);net,ga=load_generator(seed,device);sched=DiffusionSchedule(device)
    qtrue=[];qshuf=[];x31cos=[];x31norm=[];x0norm=[];manual=[];directdiff=[]
    for st in range(0,len(users),batch):
        u=users[st:st+batch];tgt=targets[st:st+batch];su=((u+1)%int(a['n_users'])).astype(np.int64)
        ht,mt=_pack_prefixes(histories,u,u,device);hs,ms=_pack_prefixes(histories,u,su,device)
        x0=reps['final_item'][torch.as_tensor(tgt,device=device)];x0norm.append(x0.norm(dim=1).cpu().numpy())
        per_t=[];per_s=[]
        for ns in NOISE_SEEDS:
            fg=torch.Generator(device=device);fg.manual_seed(int(ns)*100003+st);eps=torch.randn(x0.shape,generator=fg,device=device)
            tt=torch.full((len(u),),31,device=device,dtype=torch.long);x31=sched.q(x0,tt,eps)
            manual_x=sched.sqrt_ac[31]*x0+sched.sqrt_om[31]*eps
            manual.append((x31-manual_x).abs().max().item());directdiff.append((x31-eps).abs().max().item())
            x31cos.append(F.cosine_similarity(x31,x0).cpu().numpy());x31norm.append(x31.norm(dim=1).cpu().numpy())
            rs=int(ns)*200003+st
            per_t.append(_reverse_q75(net,sched,reps,x31,ht,mt,rs))
            per_s.append(_reverse_q75(net,sched,reps,x31,hs,ms,rs))
        qtrue.append(F.normalize(per_t[0]+per_t[1],p=2,dim=1).cpu().numpy().astype(np.float32))
        qshuf.append(F.normalize(per_s[0]+per_s[1],p=2,dim=1).cpu().numpy().astype(np.float32))
    qt=np.concatenate(qtrue);qs=np.concatenate(qshuf);x0np=reps['final_item'][torch.as_tensor(targets,device=device)].cpu().numpy();x0u=x0np/np.maximum(np.linalg.norm(x0np,axis=1,keepdims=True),1e-12)
    ctrue=np.sum(qt*x0u,axis=1);cshuf=np.sum(qs*x0u,axis=1);dh=ctrue-cshuf;cx31=np.concatenate(x31cos)
    out=RDIR/'assets'/f'seed{seed}_positive_anchored_queries.npz';np.savez_compressed(out,users=users,target=targets,q_a3=x0u.astype(np.float32),q_a4=qt,q_a4_shuffled=qs)
    audit={'seed':seed,'events':int(len(users)),'generator':ga,'deployment':'x0 positive -> q_sample(t31) -> 24 reverse steps -> Q75','pure_noise_start':False,
      'q_sample_manual_max_abs_diff':float(max(manual)),'q_sample_parity_pass':bool(max(manual)<1e-6),'x31_diff_from_direct_randn_max_abs':float(max(directdiff)),'x31_is_not_direct_randn':bool(min(directdiff)>1e-6),
      'x0_norm':stats(np.concatenate(x0norm)),'x31_norm':stats(np.concatenate(x31norm)),'q_true_norm':stats(np.linalg.norm(qt,axis=1)),'q_shuffled_norm':stats(np.linalg.norm(qs,axis=1)),
      'cos_x31_x0':stats(cx31),'cos_q_true_x0':stats(ctrue),'cos_q_shuffled_x0':stats(cshuf),'history_delta':stats(dh),'history_positive_fraction':float((dh>0).mean()),
      'positive_anchor_pass':bool(ctrue.mean()>cx31.mean()),'history_specificity_pass':bool(dh.mean()>0 and (dh>0).mean()>.5),'near_identity_fraction_cos_gt_0_995':float((ctrue>.995).mean()),
      'near_identity_warning':bool((ctrue>.995).mean()>.9),'sha256':sha256_file(out),'TEST_ACCESSED':False}
    del net,reps;torch.cuda.empty_cache();gc.collect();return audit

def _item_norm(seed):
    z=np.load(ROOT/f'diffusion_experiments/round18_diffusion_hard_negative/assets/seed{seed}_frozen_repr.npz');x=z['final_item'].astype(np.float32)
    return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)

def _top5(pool,mask,q,item_norm,batch=512):
    out_i=np.empty((len(q),TOPK),np.int32);out_c=np.empty((len(q),TOPK),np.int16);out_s=np.empty((len(q),TOPK),np.float32)
    for st in range(0,len(q),batch):
        en=min(st+batch,len(q));p=pool[st:en];m=mask[st:en];s=np.einsum('bkd,bd->bk',item_norm[p],q[st:en],optimize=True).astype(np.float32);s[~m]=-np.inf
        ix=np.argsort(-s,axis=1,kind='stable')[:,:TOPK];out_i[st:en]=np.take_along_axis(p,ix,axis=1);out_c[st:en]=ix;out_s[st:en]=np.take_along_axis(s,ix,axis=1)
    return out_i,out_c,out_s

def build_negative_maps(seed):
    p=np.load(RDIR/'assets'/f'seed{seed}_prefix_safe_pool.npz');q=np.load(RDIR/'assets'/f'seed{seed}_positive_anchored_queries.npz');users=p['users'].astype(np.int64);target=p['target'].astype(np.int64)
    if not np.array_equal(users,q['users']) or not np.array_equal(target,q['target']): raise RuntimeError('query/pool row mismatch')
    itemn=_item_norm(seed);safe=p['safe_items'].astype(np.int32);mask=p['safe_mask'].astype(bool)
    maps={};audits={}
    for v,key in [('A3','q_a3'),('A4','q_a4')]:
        ti,tc,ts=_top5(safe,mask,q[key].astype(np.float32),itemn);path=RDIR/'assets'/f'seed{seed}_{v}_top5.npz';np.savez_compressed(path,users=users,target=target,top5_items=ti,top5_safe_cols=tc,top5_cos=ts)
        rng=np.random.default_rng(AUX_SELECT_BASE);choice=rng.integers(0,TOPK,size=len(users));sel=ti[np.arange(len(users)),choice];col=tc[np.arange(len(users)),choice]
        bbneg=p['safe_backbone_scores'][np.arange(len(users)),col];clneg=p['safe_full_scores'][np.arange(len(users)),col]
        tbb=p['target_backbone_score'];tcl=p['target_full_score'];pcos=np.sum(itemn[target]*itemn[sel],axis=1);ranks=col.astype(np.float32)+21
        audits[v]={'seed':seed,'variant':v,'events':int(len(users)),'top5_path':str(path),'top5_sha256':sha256_file(path),'rank':stats(ranks),'positive_negative_cosine':stats(pcos),
          'teacher_backbone_margin':stats(tbb-bbneg),'teacher_full_colift_margin':stats(tcl-clneg),'selected_negative_equals_positive':int((sel==target).sum()),
          'selected_negative_protected_rank_count':int((ranks<21).sum()),'TEST_ACCESSED':False};maps[v]=(ti,tc)
    tsi,tsc,tss=_top5(safe,mask,q['q_a4_shuffled'].astype(np.float32),itemn)
    true=maps['A4'][0];inter=np.asarray([len(set(map(int,true[r]))&set(map(int,tsi[r]))) for r in range(len(users))],np.float32)/TOPK
    a3=maps['A3'][0];inter34=np.asarray([len(set(map(int,a3[r]))&set(map(int,true[r]))) for r in range(len(users))],np.float32)/TOPK
    rng=np.random.default_rng(AUX_SELECT_BASE);ch=rng.integers(0,TOPK,size=len(users));sel3=a3[np.arange(len(users)),ch];sel4=true[np.arange(len(users)),ch]
    diag={'seed':seed,'A3_A4_top5_overlap':stats(inter34),'A3_A4_selected_negative_agreement_epoch1':float((sel3==sel4).mean()),
      'A4_true_shuffled_top5_overlap':stats(inter),'diffusion_negative_redundant':bool(inter34.mean()>.9),'TEST_ACCESSED':False}
    return audits,diag

def selected_aux_for_epoch(seed,variant,epoch):
    z=np.load(RDIR/'assets'/f'seed{seed}_{variant}_top5.npz');n=len(z['users']);rng=np.random.default_rng(AUX_SELECT_BASE+int(epoch)-1);ch=rng.integers(0,TOPK,size=n)
    return z['users'].astype(np.int64),z['target'].astype(np.int64),z['top5_items'][np.arange(n),ch].astype(np.int64)

def _historical_checkpoint(seed):
    m,ck,ds,tr,a=load_start(seed,False);del m;torch.cuda.empty_cache();gc.collect();return ck,a

def _make_data(config):
    ds=RecDataset(config);train,valid,_=ds.split();train.inter_num=len(train.df);valid.inter_num=len(valid.df)
    td=TrainDataLoader(config,train,batch_size=config['train_batch_size'],shuffle=True)
    vd=EvalDataLoader(config,valid,additional_dataset=train,batch_size=config['eval_batch_size'])
    return ds,train,valid,td,vd

def create_shared_init(seed):
    ck,a=_historical_checkpoint(seed);config=copy.deepcopy(ck['config']);config['gpu_id']=0;config['use_gpu']=True;config['device']=torch.device('cuda:0');config['data_path']=str(ROOT/'data')+'/'
    ds,tr,va,td,vd=_make_data(config)
    seed_all(seed);td.pretrain_setup();model=MSCA(config,td).to(config['device'])
    sd={k:v.detach().cpu() for k,v in model.state_dict().items()};sh=state_hash(sd);rng=capture_rng();items=np.asarray(td.all_items,dtype=np.int64)
    path=RDIR/'assets'/f'ROUND20_INIT_SEED{seed}.pt';torch.save({'protocol':PROTOCOL,'seed':seed,'state_dict':sd,'state_hash':sh,'rng':rng,'all_items':items,'historical_teacher_checkpoint':a['checkpoint'],'historical_teacher_epoch':a['checkpoint_epoch'],'TEST_ACCESSED':False},path)
    out={'seed':seed,'path':str(path),'file_sha256':sha256_file(path),'state_hash':sh,'historical_teacher_checkpoint':a['checkpoint'],'historical_teacher_epoch':a['checkpoint_epoch'],'normal_lr':float(config['learning_rate']),'optimizer':str(config['learner']),'epochs':int(config['epochs']),'stopping_step':int(config['stopping_step']),'valid_metric':str(config['valid_metric']),'scheduler':list(config['learning_rate_scheduler']),'TEST_ACCESSED':False}
    del model;torch.cuda.empty_cache();gc.collect();return out

def instantiate_from_shared_init(seed):
    pack=torch.load(RDIR/'assets'/f'ROUND20_INIT_SEED{seed}.pt',map_location='cpu',weights_only=False);ck,a=_historical_checkpoint(seed);config=copy.deepcopy(ck['config']);config['gpu_id']=0;config['use_gpu']=True;config['device']=torch.device('cuda:0');config['data_path']=str(ROOT/'data')+'/'
    ds,tr,va,td,vd=_make_data(config);model=MSCA(config,td).to(config['device']);model.load_state_dict(pack['state_dict'],strict=True);td.all_items=list(map(int,pack['all_items'].tolist()));restore_rng(pack['rng'])
    if state_hash(model.state_dict())!=pack['state_hash']: raise RuntimeError('shared init state mismatch')
    return model,config,td,vd,pack,a

def aux_step_positions(normal_steps,aux_batches):
    x=np.rint(np.linspace(0,normal_steps-1,aux_batches)).astype(int)
    if len(np.unique(x))!=aux_batches: raise RuntimeError('duplicate auxiliary step position')
    return x.tolist()

class Round20Trainer(Trainer):
    def __init__(self,config,model,seed,variant):
        super().__init__(config,model,False);self.seed=int(seed);self.variant=variant;self.normal_plan_hashes={};self.exposure={};self.aux_positions={}
        out=RDIR/'outputs'/f'seed{seed}/{variant}';out.mkdir(parents=True,exist_ok=True);self.saved_model_file=str(out/'best.pth')
    def _train_epoch(self,train_data,epoch_idx,loss_func=None):
        self.model.train();ep=epoch_idx+1;h=hashlib.sha256();loss_sum=0.;loss_batches=[];nsteps=0
        aux_by_step={};used=[]
        if self.variant in ('A3','A4'):
            au,ap,an=selected_aux_for_epoch(self.seed,self.variant,ep);n=len(au);rng=np.random.default_rng(AUX_ORDER_BASE+ep-1);order=rng.permutation(n);bs=int(self.config['train_batch_size']);chunks=[order[s:s+bs] for s in range(0,n,bs)];pos=aux_step_positions(len(train_data),len(chunks));self.aux_positions[str(ep)]=pos
            aux_by_step={int(step):(au[ix],ap[ix],an[ix]) for step,ix in zip(pos,chunks)}
        for batch_idx,interaction in enumerate(train_data):
            self.optimizer.zero_grad();u=interaction[0].detach().cpu().numpy();p=interaction[1].detach().cpu().numpy();n=interaction[2].detach().cpu().numpy();h.update(u.tobytes());h.update(p.tobytes());h.update(n.tobytes())
            if batch_idx in aux_by_step:
                au,ap,an=aux_by_step[batch_idx];aux=tensor3(self.model,au,ap,an);normal=(interaction[0],interaction[1],interaction[2]);loss,parts=round19_loss(self.model,normal,aux,LAMBDA_HARD);used.extend(map(int,au))
            else:
                loss=self.model.calculate_loss(interaction)
            if not torch.isfinite(loss): raise RuntimeError(f'nonfinite loss {self.seed} {self.variant} epoch{ep} batch{batch_idx}')
            loss.backward()
            if self.clip_grad_norm: torch.nn.utils.clip_grad_norm_(self.model.parameters(),**self.clip_grad_norm)
            self.optimizer.step();loss_sum+=float(loss.detach());loss_batches.append(loss.detach());nsteps+=1
        self.normal_plan_hashes[str(ep)]=h.hexdigest();dup=len(used)-len(set(used));self.exposure[str(ep)]={'unique_auxiliary_events':len(set(used)),'used_auxiliary_events':len(used),'duplicate_auxiliary_events':dup,'max_event_usage_per_epoch':0 if not used else 1 if dup==0 else 2,'normal_optimizer_steps':nsteps,'auxiliary_optimizer_steps_added':0}
        return loss_sum,loss_batches

def audit_shared_init(seed):
    pack=torch.load(RDIR/'assets'/f'ROUND20_INIT_SEED{seed}.pt',map_location='cpu',weights_only=False);rows={}
    for v in ('B0','A3','A4'):
        m,c,td,vd,p,a=instantiate_from_shared_init(seed);tr=Round20Trainer(c,m,seed,v);rows[v]={'state_hash':state_hash(m.state_dict()),'optimizer_state_entries':len(tr.optimizer.state),'optimizer_type':type(tr.optimizer).__name__,'lr':float(tr.optimizer.param_groups[0]['lr']),'weight_decay':float(tr.optimizer.param_groups[0]['weight_decay'])};del tr,m;torch.cuda.empty_cache();gc.collect()
    hs={x['state_hash'] for x in rows.values()};return {'seed':seed,'canonical_state_hash':pack['state_hash'],'variants':rows,'exact_equal':len(hs)==1 and pack['state_hash'] in hs,'TEST_ACCESSED':False}

def loss_parity_audit(seed):
    m,c,td,vd,p,a=instantiate_from_shared_init(seed);it=iter(td);interaction=next(it);normal=(interaction[0],interaction[1],interaction[2]);direct=m.calculate_loss(interaction);wrapped,_=round19_loss(m,normal,None,0.0);diff=float(abs(direct.detach()-wrapped.detach()))
    au,ap,an=selected_aux_for_epoch(seed,'A4',1);aux=tensor3(m,au[:min(len(au),len(interaction[0]))],ap[:min(len(ap),len(interaction[0]))],an[:min(len(an),len(interaction[0]))]);tot,parts=round19_loss(m,normal,aux,LAMBDA_HARD);vals={k:float(v.detach()) for k,v in parts.items()};err=abs((vals['L_total']-vals['L_MSCA'])-LAMBDA_HARD*vals['L_aux'])
    out={'seed':seed,'direct_original_loss':float(direct.detach()),'wrapper_lambda0':float(wrapped.detach()),'abs_loss_diff':diff,'loss_parity_pass':diff<1e-6,'aux_parts':vals,'aux_identity_error':float(err),'aux_loss_pass':err<1e-6,'TEST_ACCESSED':False};del m;torch.cuda.empty_cache();gc.collect();return out

def train_full_variant(seed,variant,verbose=False):
    m,c,td,vd,pack,a=instantiate_from_shared_init(seed);initial_hash=state_hash(m.state_dict());trainer=Round20Trainer(c,m,seed,variant);best_score,best_result=trainer.fit(td,valid_data=vd,saved=True,verbose=verbose)
    cp=torch.load(trainer.saved_model_file,map_location='cpu',weights_only=False);m.load_state_dict(cp['state_dict'],strict=True);ev=CachedEvaluator(seed,m).evaluate(m)
    hist=a['validation_metrics'];r20rel=float((ev['backbone']['R20']-hist['R20'])/hist['R20'])
    out={'protocol':PROTOCOL,'seed':seed,'variant':variant,'initial_state_hash':initial_hash,'canonical_init_hash':pack['state_hash'],'init_exact':initial_hash==pack['state_hash'],
      'optimizer':type(trainer.optimizer).__name__,'learning_rate':float(c['learning_rate']),'weight_decay':float(c['weight_decay']),'scheduler':list(c['learning_rate_scheduler']),
      'epochs_limit':int(c['epochs']),'stopping_step':int(c['stopping_step']),'valid_metric':str(c['valid_metric']),'best_epoch':int(cp['epoch']),'best_valid_score':float(cp['best_valid_score']),
      'best_valid_result_original_trainer':{k:float(v) for k,v in best_result.items()},'best_checkpoint':trainer.saved_model_file,'checkpoint_sha256':sha256_file(trainer.saved_model_file),
      'evaluation':ev,'normal_plan_hashes':trainer.normal_plan_hashes,'aux_step_positions':trainer.aux_positions,'aux_exposure':trainer.exposure,
      'historical_teacher_validation':hist,'historical_R20_relative_difference':r20rel,'baseline_reproduction_warning':bool(variant=='B0' and abs(r20rel)>.02),
      'diffusion_inference_calls':0,'TEST_ACCESSED':False}
    pth=RDIR/'evidence'/f'ROUND20_{variant}_SEED{seed}.json';pth.write_text(json.dumps(out,indent=2)+'\n')
    del trainer,m;torch.cuda.empty_cache();gc.collect();return out

def normal_plan_fairness(results):
    common=min(len(results[v]['normal_plan_hashes']) for v in ('B0','A3','A4'));mismatch=[]
    for ep in range(1,common+1):
        hs={results[v]['normal_plan_hashes'][str(ep)] for v in ('B0','A3','A4')}
        if len(hs)!=1:mismatch.append(ep)
    return {'common_epochs_compared':common,'mismatch_epochs':mismatch,'exact_reuse_on_common_epochs':len(mismatch)==0,'optimizer_steps_per_epoch':58,'TEST_ACCESSED':False}
