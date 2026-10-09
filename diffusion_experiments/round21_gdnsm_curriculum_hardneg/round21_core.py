from __future__ import annotations
import copy, gc, hashlib, json, math, random, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
RDIR=ROOT/'diffusion_experiments/round21_gdnsm_curriculum_hardneg'
PROTOCOL='ROUND21_GDNSM_CURRICULUM_HARDNEG_V1'
SEEDS=(999,1000)
LAMBDA_HN=.20
REFRESH_INTERVAL=5
DIFF_PRETRAIN_EPOCHS=5
DIFF_REFRESH_EPOCHS=1
DIFF_LR=1e-3
M=2
FORMAL_EPOCHS=60
ALL=('R10','N10','R20','N20','R50','N50')
PRIMARY=('R10','N10','R20','N20')

from models.msca import MSCA
from utils.dataset import RecDataset
from utils.dataloader import TrainDataLoader,EvalDataLoader
from utils.utils import init_seed
from modules.ranking import topk_from_embeddings,sha256_file
from pipelines.dataset_config import load_dataset_config
from pipelines.msca_assets import build_train_histories_and_validation
from diffusion_experiments.round19_safe_aux_hardneg.round19_core import CachedEvaluator
from diffusion_experiments.round20_positive_anchored_hardneg import round20_core as r20
from diffusion_experiments.round21_gdnsm_curriculum_hardneg.round21_diffusion import ConditionalNoiseMLP,LinearDDPM,epsilon_loss,generate,T


def stats(x):
    x=np.asarray(x,np.float64)
    return {'mean':float(x.mean()),'median':float(np.median(x)),'p10':float(np.quantile(x,.1)),'p50':float(np.quantile(x,.5)),'p90':float(np.quantile(x,.9)),'p99':float(np.quantile(x,.99)),'min':float(x.min()),'max':float(x.max())}
def utility(m,b):return float(np.mean([(m[k]-b[k])/b[k] for k in PRIMARY]))
def delta_pack(m,b):return {'U':utility(m,b),'relative_delta':{k:float((m[k]-b[k])/b[k]) for k in ALL},'primary_positive_count':int(sum(m[k]>b[k] for k in PRIMARY)),'overall_positive_count':int(sum(m[k]>b[k] for k in ALL))}
def state_hash(sd):
    h=hashlib.sha256()
    for k in sorted(sd):
        v=sd[k].detach().cpu().contiguous();h.update(k.encode());h.update(str(v.dtype).encode());h.update(np.asarray(v.shape,dtype=np.int64).tobytes());h.update(v.numpy().tobytes())
    return h.hexdigest()
def capture_rng():return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
def restore_rng(x):
    random.setstate(x['python']);np.random.set_state(x['numpy']);torch.set_rng_state(x['torch'])
    if torch.cuda.is_available() and x['cuda']:torch.cuda.set_rng_state_all(x['cuda'])
def curriculum_count(e):return 0 if e<10 else 2 if e<20 else 4 if e<30 else 6


def _make_data(config):
    ds=RecDataset(config);train,valid,_=ds.split();train.inter_num=len(train.df);valid.inter_num=len(valid.df)
    td=TrainDataLoader(config,train,batch_size=config['train_batch_size'],shuffle=True);vd=EvalDataLoader(config,valid,additional_dataset=train,batch_size=config['eval_batch_size'])
    return ds,train,valid,td,vd

def create_shared_init(seed):
    ck,a=r20._historical_checkpoint(seed);config=copy.deepcopy(ck['config']);config['gpu_id']=0;config['use_gpu']=True;config['device']=torch.device('cuda:0');config['data_path']=str(ROOT/'data')+'/'
    ds,tr,va,td,vd=_make_data(config);init_seed(int(seed));td.pretrain_setup();model=MSCA(config,td).to(config['device'])
    sd={k:v.detach().cpu() for k,v in model.state_dict().items()};sh=state_hash(sd);rng=capture_rng();items=np.asarray(td.all_items,dtype=np.int64)
    path=RDIR/'assets'/f'ROUND21_INIT_SEED{seed}.pt';torch.save({'protocol':PROTOCOL,'seed':seed,'state_dict':sd,'state_hash':sh,'rng':rng,'all_items':items,'config':config,'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False},path)
    out={'seed':seed,'path':str(path),'file_sha256':sha256_file(path),'state_hash':sh,'train_interactions':int(len(tr.df)),'normal_steps_per_epoch':int(len(td)),'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
    del model;torch.cuda.empty_cache();gc.collect();return out

def instantiate_shared(seed):
    path=RDIR/'assets'/f'ROUND21_INIT_SEED{seed}.pt'
    if not path.exists():create_shared_init(seed)
    pack=torch.load(path,map_location='cpu',weights_only=False);config=copy.deepcopy(pack['config']);config['device']=torch.device('cuda:0');config['gpu_id']=0;config['use_gpu']=True
    ds,tr,va,td,vd=_make_data(config);model=MSCA(config,td).to(config['device']);model.load_state_dict(pack['state_dict'],strict=True);td.all_items=list(map(int,pack['all_items'].tolist()));restore_rng(pack['rng'])
    if state_hash(model.state_dict())!=pack['state_hash']:raise RuntimeError('shared init mismatch')
    return model,config,td,vd,pack


class TrainEvents:
    def __init__(self,model):
        cfg=load_dataset_config('baby');self.histories,self.pseudo,self.pusers,self.vusers,self.vsets=build_train_histories_and_validation(cfg['resolved_paths']['interaction'],model.n_users)
        self.users=np.asarray([u for u,h in enumerate(self.histories) if len(h)>0],np.int64);U=[];P=[]
        for u in self.users:
            for i in self.histories[int(u)]:U.append(int(u));P.append(int(i))
        self.event_u=np.asarray(U,np.int64);self.event_p=np.asarray(P,np.int64)
    def refresh_pairs(self,refresh_idx):
        p=np.asarray([self.histories[int(u)][int(refresh_idx)%len(self.histories[int(u)])] for u in self.users],np.int64);return self.users.copy(),p
    def user_disjoint_split(self,seed):
        rng=np.random.default_rng(20262100+int(seed));perm=rng.permutation(self.users);cut=int(round(.9*len(perm)));train=set(map(int,perm[:cut]));mt=np.asarray([int(u) in train for u in self.event_u],bool)
        return (self.event_u[mt],self.event_p[mt]),(self.event_u[~mt],self.event_p[~mt]),np.asarray(sorted(set(map(int,perm[cut:]))),np.int64)


@torch.no_grad()
def current_snapshot(model):
    model.eval();fu,fi=model.forward(test=True);txt=model.text_trs(model.text_embedding.weight);vis=model.image_trs(model.image_embedding.weight)
    for name,x in [('user',fu),('item',fi),('text',txt),('visual',vis)]:
        if x.shape[1]!=64:raise RuntimeError(f'{name} dimension {x.shape}')
    return {'user':fu.detach(),'item':fi.detach(),'text':txt.detach(),'visual':vis.detach()}


def train_diffusion(net,opt,sched,snap,events,epochs,seed,batch=1024,max_events=None):
    u,p=events;n=len(u);losses=[];steps=0
    if max_events is not None:n=min(n,int(max_events));u=u[:n];p=p[:n]
    for ep in range(int(epochs)):
        rng=np.random.default_rng(int(seed)+ep);order=rng.permutation(n);net.train();torch.manual_seed(int(seed)+ep)
        for st in range(0,n,batch):
            ix=order[st:st+batch];ut=torch.as_tensor(u[ix],device=snap['user'].device);pt=torch.as_tensor(p[ix],device=snap['user'].device)
            x0=snap['item'][pt].detach();hu=snap['user'][ut].detach();tc=snap['text'][pt].detach();vc=snap['visual'][pt].detach();opt.zero_grad(set_to_none=True);loss,_,_,_=epsilon_loss(net,sched,x0,hu,tc,vc)
            if not torch.isfinite(loss):raise RuntimeError('nonfinite diffusion loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.0);opt.step();losses.append(float(loss.detach()));steps+=1
    return {'steps':steps,'loss':stats(losses)}

@torch.no_grad()
def epsilon_audit(net,sched,snap,events,seed,max_events=2048):
    u,p=events;u=u[:max_events];p=p[:max_events];dev=snap['user'].device;ut=torch.as_tensor(u,device=dev);pt=torch.as_tensor(p,device=dev)
    x0=snap['item'][pt];hu=snap['user'][ut];tc=snap['text'][pt];vc=snap['visual'][pt];net.eval();out={};allpred=[]
    bins={'low':list(range(0,8)),'middle':list(range(8,17)),'high':list(range(17,25))}
    for bi,(name,vals) in enumerate(bins.items()):
        g=torch.Generator(device=dev);g.manual_seed(20262180+int(seed)+bi);tv=torch.tensor(vals,device=dev);t=tv[torch.arange(len(u),device=dev)%len(tv)]
        eps=torch.randn(x0.shape,device=dev,generator=g);xt=sched.q_sample(x0,t,eps);pred=net(xt,t,hu,tc,vc)
        mse=((pred-eps)**2).mean(1).cpu().numpy();base=(eps**2).mean(1).cpu().numpy();ratio=float(mse.mean()/base.mean())
        out[name]={'epsilon_mse':stats(mse),'zero_baseline_mse':stats(base),'mse_ratio':ratio,'PASS':bool(np.isfinite(mse).all() and ratio<.98)};allpred.append(pred.cpu())
    pp=torch.cat(allpred);std=float(pp.std(dim=0).mean())
    return {'bins':out,'prediction_dim_std_mean':std,'noncollapse':bool(std>1e-4),'PASS':bool(all(v['PASS'] for v in out.values()) and std>1e-4)}

def _cos(a,b):return F.cosine_similarity(a,b,dim=1).detach().cpu().numpy()

@torch.no_grad()
def generate_six(net,sched,snap,u,p,seed_base,batch=256,include_user_only=False,user_override=None):
    dev=snap['user'].device;u=np.asarray(u,np.int64);p=np.asarray(p,np.int64);outs={k:[] for k in ('V','T','TV')};uo=[]
    over=None if user_override is None else np.asarray(user_override,np.int64)
    for st in range(0,len(u),batch):
        ub=u[st:st+batch];pb=p[st:st+batch];ut=torch.as_tensor(ub,device=dev);pt=torch.as_tensor(pb,device=dev)
        hu=snap['user'][ut] if over is None else snap['user'][torch.as_tensor(over[st:st+batch],device=dev)];tc=snap['text'][pt];vc=snap['visual'][pt]
        for mode in ('V','T','TV'):
            per=[generate(net,sched,hu,tc,vc,mode,int(seed_base)+j*100003+st).cpu() for j in range(M)];outs[mode].append(torch.stack(per,dim=1))
        if include_user_only:uo.append(generate(net,sched,hu,tc,vc,'U',int(seed_base)+st).cpu())
    z={k:torch.cat(v,0) for k,v in outs.items()}
    if include_user_only:z['U']=torch.cat(uo,0)
    return z

@torch.no_grad()
def mechanism_generation_audit(net,sched,snap,events,seed,max_users=512):
    u,p=events.refresh_pairs(0);u=u[:max_users];p=p[:max_users];dev=snap['user'].device
    gen=generate_six(net,sched,snap,u,p,20262200+int(seed)*10,include_user_only=True);ut=torch.as_tensor(u,device=dev);pt=torch.as_tensor(p,device=dev)
    hu=snap['user'][ut].cpu();pos=snap['item'][pt].cpu();tc=snap['text'][pt].cpu();vc=snap['visual'][pt].cpu();user_only=gen['U']
    real_norm=snap['item'].norm(dim=1).cpu().numpy();synth=torch.cat([gen['V'].reshape(-1,64),gen['T'].reshape(-1,64),gen['TV'].reshape(-1,64)],0);sn=synth.norm(dim=1).numpy()
    m2={'real_item_norm':stats(real_norm),'synthetic_norm':stats(sn)};m2['p99_ratio']=m2['synthetic_norm']['p99']/max(m2['real_item_norm']['p99'],1e-12);m2['PASS']=bool(np.isfinite(sn).all() and m2['p99_ratio']<=3.)
    uv=_cos(user_only,vc);utx=_cos(user_only,tc);vv=_cos(gen['V'][:,0,:],vc);tt=_cos(gen['T'][:,0,:],tc);tvv=_cos(gen['TV'][:,0,:],vc);tvt=_cos(gen['TV'][:,0,:],tc)
    m3={'user_only_visual':stats(uv),'V_visual':stats(vv),'user_only_text':stats(utx),'T_text':stats(tt),'TV_visual':stats(tvv),'TV_text':stats(tvt)}
    m3['PASS']=bool(vv.mean()>uv.mean() and tt.mean()>utx.mean() and tvv.mean()>uv.mean() and tvt.mean()>utx.mean())
    sh=np.roll(u,1);true_tv=gen['TV'][:,0,:];shg=generate_six(net,sched,snap,u,p,20262200+int(seed)*10,user_override=sh)['TV'][:,0,:]
    cs=_cos(true_tv,shg);dist=(true_tv-shg).norm(dim=1).numpy();score_true=(hu*true_tv).sum(1).numpy();score_shuf=(hu*shg).sum(1).numpy()
    m4={'cos_true_shuffled':stats(cs),'distance':stats(dist),'score_true':stats(score_true),'score_shuffled':stats(score_shuf),'mean_abs_score_delta':float(np.mean(np.abs(score_true-score_shuf)))};m4['PASS']=bool(np.isfinite(cs).all() and cs.mean()<.999 and dist.mean()>1e-3)
    ps=(hu*pos).sum(1).numpy();hard={};margins={}
    for mode in ('V','T','TV'):
        x=gen[mode].numpy();neg=np.einsum('bd,bmd->bm',hu.numpy(),x);mar=ps[:,None]-neg;margins[mode]=mar.reshape(-1)
        hard[mode]={'margin':stats(margins[mode]),'p_negative_ge_positive':float((neg>=ps[:,None]).mean()),'negative_score':stats(neg.reshape(-1))}
    m5={'levels':hard};m5['TV_harder_than_V_T']=bool(margins['TV'].mean()<margins['V'].mean() and margins['TV'].mean()<margins['T'].mean());m5['overwhelming_negative_warning']=bool(max(v['p_negative_ge_positive'] for v in hard.values())>=.80);m5['PASS']=bool(m5['TV_harder_than_V_T'] and not m5['overwhelming_negative_warning'])
    early=float(margins['V'].mean());middle=float(np.concatenate([margins['V'],margins['T']]).mean());late=float(np.concatenate([margins['V'],margins['T'],margins['TV']]).mean());m6={'early_V_margin':early,'middle_VT_margin':middle,'late_VTTV_margin':late,'PASS':bool(early>middle>late)}
    return {'M2_generated_norm':m2,'M3_modality_guidance':m3,'M4_user_conditioning':m4,'M5_ranking_hardness':m5,'M6_curriculum':m6,'samples':int(len(u))}


def run_mechanism(seed):
    model,config,td,vd,pack=instantiate_shared(seed);events=TrainEvents(model);train_ev,audit_ev,audit_users=events.user_disjoint_split(seed);snap=current_snapshot(model)
    saved=capture_rng();init_seed(20262300+int(seed));net=ConditionalNoiseMLP().to(model.device);opt=torch.optim.Adam(net.parameters(),lr=DIFF_LR);sched=LinearDDPM(model.device)
    tr=train_diffusion(net,opt,sched,snap,train_ev,DIFF_PRETRAIN_EPOCHS,20262400+int(seed));m1=epsilon_audit(net,sched,snap,audit_ev,seed)
    # generation audit uses TRAIN-only user-positive events; no Validation/Test labels are touched.
    ga=mechanism_generation_audit(net,sched,snap,events,seed);restore_rng(saved)
    gates={'M1':m1['PASS'],'M2':ga['M2_generated_norm']['PASS'],'M3':ga['M3_modality_guidance']['PASS'],'M4':ga['M4_user_conditioning']['PASS'],'M5':ga['M5_ranking_hardness']['PASS'],'M6':ga['M6_curriculum']['PASS']}
    out={'protocol':PROTOCOL,'seed':seed,'diffusion_train':tr,'M1_diffusion_learning':m1,**ga,'gates':gates,'MECHANISM_PASS':bool(all(gates.values())),'audit_user_count':int(len(audit_users)),'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
    (RDIR/'evidence'/f'ROUND21_DIFFUSION_AUDIT_SEED{seed}.json').write_text(json.dumps(out,indent=2)+'\n');(RDIR/'evidence'/f'ROUND21_GUIDANCE_AUDIT_SEED{seed}.json').write_text(json.dumps({'seed':seed,'M3':ga['M3_modality_guidance'],'M4':ga['M4_user_conditioning'],'TEST_ACCESSED':False},indent=2)+'\n');(RDIR/'evidence'/f'ROUND21_HARDNESS_AUDIT_SEED{seed}.json').write_text(json.dumps({'seed':seed,'M2':ga['M2_generated_norm'],'M5':ga['M5_ranking_hardness'],'M6':ga['M6_curriculum'],'TEST_ACCESSED':False},indent=2)+'\n')
    del net,opt,model;torch.cuda.empty_cache();gc.collect();return out

def array_hash(*arrs):
    h=hashlib.sha256()
    for a in arrs:
        x=np.asarray(a);h.update(str(x.dtype).encode());h.update(np.asarray(x.shape,dtype=np.int64).tobytes());h.update(x.tobytes())
    return h.hexdigest()

@torch.no_grad()
def build_control_cache(snap,events,refresh_idx,seed):
    u,p=events.refresh_pairs(refresh_idx);items,scores=topk_from_embeddings(snap['user'],snap['item'],u,events.histories,100,1024);rng=np.random.default_rng(20262500+int(seed)*100+int(refresh_idx));neg=np.empty((len(u),6),np.int64)
    bands=[(60,100),(30,60),(10,30)]
    for r in range(len(u)):
        for bi,(lo,hi) in enumerate(bands):
            pool=items[r,lo:hi];choice=rng.choice(len(pool),size=M,replace=False);neg[r,bi*M:(bi+1)*M]=pool[choice]
    # topk_from_embeddings already masks all TRAIN-observed items; assert discipline explicitly.
    viol=0
    for r,uid in enumerate(u):
        seen=set(events.histories[int(uid)]);viol+=sum(int(x) in seen or int(x)==int(p[r]) or int(x)<0 for x in neg[r])
    if viol:raise RuntimeError(f'C1 safe-negative violation {viol}')
    return {'user':u,'positive':p,'negative':neg,'hash':array_hash(u,p,neg),'violations':int(viol)}

@torch.no_grad()
def build_diffusion_cache(net,sched,snap,events,refresh_idx,seed):
    u,p=events.refresh_pairs(refresh_idx);g=generate_six(net,sched,snap,u,p,20262700+int(seed)*100+int(refresh_idx),batch=256)
    neg=torch.cat([g['V'],g['T'],g['TV']],dim=1).contiguous().float();norm=neg.norm(dim=2).numpy();return {'user':u,'positive':p,'negative':neg,'hash':array_hash(u,p,neg.numpy()),'norm':stats(norm.reshape(-1))}


def aux_positions(normal_steps,n_events,batch_size):
    n_chunks=int(math.ceil(n_events/batch_size));x=np.rint(np.linspace(0,normal_steps-1,n_chunks)).astype(int)
    if len(np.unique(x))!=n_chunks:raise RuntimeError('duplicate aux insertion position')
    return x.tolist()

def aux_plan(cache,epoch_idx,batch_size,normal_steps,seed):
    n=len(cache['user']);rng=np.random.default_rng(20262900+int(seed)*100+int(epoch_idx));order=rng.permutation(n);chunks=[order[s:s+batch_size] for s in range(0,n,batch_size)];pos=aux_positions(normal_steps,n,batch_size)
    return {int(step):ix for step,ix in zip(pos,chunks)},pos,array_hash(cache['user'],cache['positive'],order)

def synthetic_aux_loss(model,u,p,neg,k,variant):
    dev=model.device;ut=torch.as_tensor(u,device=dev,dtype=torch.long);pt=torch.as_tensor(p,device=dev,dtype=torch.long);fu,fi=model.forward(test=True);hu=fu[ut];hp=fi[pt];ps=(hu*hp).sum(1,keepdim=True)
    if variant=='C1':
        nt=torch.as_tensor(neg[:,:k],device=dev,dtype=torch.long);xn=fi[nt]
    else:
        xn=neg[:,:k].to(dev)
    ns=torch.einsum('bd,bkd->bk',hu,xn);return -F.logsigmoid(ps-ns).mean()


def save_checkpoint(model,seed,variant,epoch,score,net=None):
    p=RDIR/'outputs'/f'seed{seed}/{variant}/best.pt';p.parent.mkdir(parents=True,exist_ok=True);obj={'protocol':PROTOCOL,'seed':seed,'variant':variant,'epoch':int(epoch),'full_colift_R20':float(score),'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'TEST_ACCESSED':False}
    if net is not None:obj['diffusion_state_dict']={k:v.detach().cpu() for k,v in net.state_dict().items()}
    torch.save(obj,p);return p


def train_variant(seed,variant):
    model,config,td,vd,pack=instantiate_shared(seed);events=TrainEvents(model);ev=CachedEvaluator(seed,model);wd=float(config['weight_decay']) if config['weight_decay'] is not None else 0.;opt=torch.optim.Adam(model.parameters(),lr=float(config['learning_rate']),weight_decay=wd);fac=config['learning_rate_scheduler'];sched_lr=torch.optim.lr_scheduler.LambdaLR(opt,lr_lambda=lambda e:fac[0]**(e/fac[1]))
    net=dopt=dsched=None
    if variant=='D1':
        rr=capture_rng();init_seed(20263000+int(seed));net=ConditionalNoiseMLP().to(model.device);dopt=torch.optim.Adam(net.parameters(),lr=DIFF_LR);dsched=LinearDDPM(model.device);restore_rng(rr)
    best=-1.;best_epoch=-1;best_eval=None;best_path=None;trajectory=[];normal_hashes={};refresh_records=[];aux_event_hashes={};aux_positions_log={};cache=None
    for ep in range(FORMAL_EPOCHS):
        if variant in ('C1','D1') and ep%REFRESH_INTERVAL==0:
            ri=ep//REFRESH_INTERVAL;rr=capture_rng();snap=current_snapshot(model);drec={'refresh_index':ri,'epoch':ep}
            if variant=='D1':
                de=DIFF_PRETRAIN_EPOCHS if ep==0 else DIFF_REFRESH_EPOCHS;drec['diffusion_update']=train_diffusion(net,dopt,dsched,snap,(events.event_u,events.event_p),de,20263100+int(seed)*100+ri);cache=build_diffusion_cache(net,dsched,snap,events,ri,seed);drec['negative_norm']=cache['norm']
            else:cache=build_control_cache(snap,events,ri,seed);drec['safe_violations']=cache['violations']
            drec['cache_hash']=cache['hash'];drec['event_hash']=array_hash(cache['user'],cache['positive']);refresh_records.append(drec);restore_rng(rr)
        k=curriculum_count(ep);plan={};pos=[];eph='none'
        if variant in ('C1','D1') and k>0:
            plan,pos,eph=aux_plan(cache,ep,int(config['train_batch_size']),len(td),seed);aux_event_hashes[str(ep)]=eph;aux_positions_log[str(ep)]=pos
        model.train();hh=hashlib.sha256();sum_normal=0.;sum_aux=0.;steps=0;aux_batches=0
        for bi,interaction in enumerate(td):
            u=interaction[0].detach().cpu().numpy();p=interaction[1].detach().cpu().numpy();n=interaction[2].detach().cpu().numpy();hh.update(u.tobytes());hh.update(p.tobytes());hh.update(n.tobytes());opt.zero_grad(set_to_none=True);ln=model.calculate_loss(interaction);loss=ln
            if bi in plan:
                ix=plan[bi];la=synthetic_aux_loss(model,cache['user'][ix],cache['positive'][ix],cache['negative'][ix],k,variant);loss=ln+LAMBDA_HN*la;sum_aux+=float(la.detach());aux_batches+=1
            if not torch.isfinite(loss):raise RuntimeError(f'nonfinite recommender loss {seed} {variant} ep{ep} batch{bi}')
            loss.backward()
            if config['clip_grad_norm']:torch.nn.utils.clip_grad_norm_(model.parameters(),**config['clip_grad_norm'])
            opt.step();sum_normal+=float(ln.detach());steps+=1
        sched_lr.step();normal_hashes[str(ep)]=hh.hexdigest();res=ev.evaluate(model);score=float(res['colift']['R20'])
        rec={'epoch':ep,'g':k,'normal_steps':steps,'aux_batches':aux_batches,'normal_loss_mean':sum_normal/max(steps,1),'aux_loss_mean':None if aux_batches==0 else sum_aux/aux_batches,'backbone':res['backbone'],'colift':res['colift'],'full_colift_R20':score};trajectory.append(rec)
        print(json.dumps({'seed':seed,'variant':variant,'epoch':ep,'g':k,'R20':score,'R10':res['colift']['R10'],'N10':res['colift']['N10'],'N20':res['colift']['N20']}),flush=True)
        if score>best:
            best=score;best_epoch=ep;best_eval=res;best_path=save_checkpoint(model,seed,variant,ep,score,net)
    out={'protocol':PROTOCOL,'seed':seed,'variant':variant,'initial_state_hash':pack['state_hash'],'optimizer':'Adam','lr':float(config['learning_rate']),'weight_decay':wd,'epochs_trained':FORMAL_EPOCHS,'normal_optimizer_steps_total':int(FORMAL_EPOCHS*len(td)),'checkpoint_selection':'Full-CoLiftRec Validation R20','best_epoch':int(best_epoch),'best_full_colift_R20':float(best),'best_evaluation':best_eval,'checkpoint':str(best_path),'checkpoint_sha256':sha256_file(best_path),'normal_plan_hashes':normal_hashes,'refresh_records':refresh_records,'aux_event_hashes':aux_event_hashes,'aux_positions':aux_positions_log,'diffusion_inference_calls':0,'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
    (RDIR/'evidence'/f'ROUND21_{variant}_SEED{seed}.json').write_text(json.dumps(out,indent=2)+'\n');del model,opt,net,dopt;torch.cuda.empty_cache();gc.collect();return out



def fairness(seed,results):
    variants=('B0','C1','D1');init_equal=len({results[v]['initial_state_hash'] for v in variants})==1;normal_mismatch=[]
    for ep in range(FORMAL_EPOCHS):
        hs={results[v]['normal_plan_hashes'][str(ep)] for v in variants}
        if len(hs)!=1:normal_mismatch.append(ep)
    aux_mismatch=[];pos_mismatch=[]
    for ep in range(FORMAL_EPOCHS):
        if curriculum_count(ep)>0:
            if results['C1']['aux_event_hashes'].get(str(ep))!=results['D1']['aux_event_hashes'].get(str(ep)):aux_mismatch.append(ep)
            if results['C1']['aux_positions'].get(str(ep))!=results['D1']['aux_positions'].get(str(ep)):pos_mismatch.append(ep)
    steps={v:results[v]['normal_optimizer_steps_total'] for v in variants}
    out={'seed':seed,'initial_state_exact':init_equal,'normal_plan_mismatch_epochs':normal_mismatch,'aux_event_mismatch_epochs':aux_mismatch,'aux_position_mismatch_epochs':pos_mismatch,'normal_optimizer_steps':steps,'PASS':bool(init_equal and not normal_mismatch and not aux_mismatch and not pos_mismatch and len(set(steps.values()))==1),'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
    (RDIR/'evidence'/f'ROUND21_FAIRNESS_SEED{seed}.json').write_text(json.dumps(out,indent=2)+'\n');return out


def run_smoke():
    seed=999;model,config,td,vd,pack=instantiate_shared(seed);events=TrainEvents(model);snap=current_snapshot(model);saved=capture_rng();init_seed(20262199);net=ConditionalNoiseMLP().to(model.device);dopt=torch.optim.Adam(net.parameters(),lr=DIFF_LR);ds=LinearDDPM(model.device)
    de=train_diffusion(net,dopt,ds,snap,(events.event_u,events.event_p),1,20262199,max_events=2048);u,p=events.refresh_pairs(0);u=u[:8];p=p[:8];gen=generate_six(net,ds,snap,u,p,20262299,batch=8);shape=list(torch.cat([gen['V'],gen['T'],gen['TV']],1).shape);gn=stats(torch.cat([gen['V'],gen['T'],gen['TV']],1).norm(dim=2).numpy().reshape(-1))
    interaction=next(iter(td));opt=torch.optim.Adam(model.parameters(),lr=float(config['learning_rate']),weight_decay=float(config['weight_decay']));opt.zero_grad(set_to_none=True);ln=model.calculate_loss(interaction);neg=torch.cat([gen['V'],gen['T'],gen['TV']],1);la=synthetic_aux_loss(model,u,p,neg,2,'D1');loss=ln+LAMBDA_HN*la;loss.backward();grads=[x.grad for x in model.parameters() if x.grad is not None];grad_finite=bool(grads and all(torch.isfinite(x).all() for x in grads));opt.step();snap2=current_snapshot(model);dr=train_diffusion(net,dopt,ds,snap2,(events.event_u,events.event_p),1,20262399,max_events=1024);restore_rng(saved)
    out={'protocol':PROTOCOL,'seed':seed,'finite_loss':bool(torch.isfinite(loss).item()),'normal_loss':float(ln.detach()),'aux_loss':float(la.detach()),'gradient_finite':grad_finite,'generated_shape':shape,'generated_norm':gn,'V_sample':gen['V'][0,0,:6].tolist(),'T_sample':gen['T'][0,0,:6].tolist(),'TV_sample':gen['TV'][0,0,:6].tolist(),'g_epoch':{'0':curriculum_count(0),'10':curriculum_count(10),'20':curriculum_count(20),'30':curriculum_count(30)},'normal_optimizer_steps':1,'diffusion_optimizer_steps_initial':de['steps'],'diffusion_optimizer_steps_refresh':dr['steps'],'refresh_verified':True,'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
    out['PASS']=bool(out['finite_loss'] and grad_finite and shape==[8,6,64] and out['g_epoch']=={'0':0,'10':2,'20':4,'30':6})
    (RDIR/'evidence'/'ROUND21_SMOKE.json').write_text(json.dumps(out,indent=2)+'\n');del model,net,opt,dopt;torch.cuda.empty_cache();gc.collect();return out


def summarize_formal(results_by_seed):
    summary={'seeds':{},'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False};dcb=[];db=[]
    for seed,res in results_by_seed.items():
        b=res['B0']['best_evaluation']['colift'];c=res['C1']['best_evaluation']['colift'];d=res['D1']['best_evaluation']['colift'];x={'C1_vs_B0':delta_pack(c,b),'D1_vs_B0':delta_pack(d,b),'D1_vs_C1':delta_pack(d,c),'best_epochs':{v:int(res[v]['best_epoch']) for v in ('B0','C1','D1')},'best_metrics':{v:res[v]['best_evaluation']['colift'] for v in ('B0','C1','D1')}};summary['seeds'][str(seed)]=x;dcb.append(x['D1_vs_C1']['U']);db.append(x['D1_vs_B0']['U'])
    summary['mean_U_D1_vs_C1']=float(np.mean(dcb));summary['mean_U_D1_vs_B0']=float(np.mean(db));summary['D1_vs_C1_positive_both']=bool(all(x>0 for x in dcb));summary['target_diffusion_specific_0_5pct']=bool(summary['D1_vs_C1_positive_both'] and summary['mean_U_D1_vs_C1']>=.005);summary['target_total_1pct']=bool(summary['mean_U_D1_vs_B0']>=.01)
    mixed=(dcb[0]*dcb[1]<0);summary['failure_no_verified_gain']=bool(mixed and np.mean(np.abs(dcb))<.002);summary['failure_literature_faithful']=bool(all(x<0 for x in dcb))
    (RDIR/'evidence'/'ROUND21_VALIDATION_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n');return summary
