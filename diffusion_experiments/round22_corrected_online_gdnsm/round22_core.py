from __future__ import annotations
import copy,gc,hashlib,json,math,random,sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
RDIR=ROOT/'diffusion_experiments/round22_corrected_online_gdnsm'
PROTOCOL='ROUND22_CORRECTED_ONLINE_GDNSM_V1'
SEEDS=(999,1000)
WARMUP_EPOCHS=10
FORMAL_EPOCHS=60
LAMBDA_HN=.20
DIFF_LR=1e-3
T0_CANDIDATES=(2,4,6,8,10)
ALL=('R10','N10','R20','N20','R50','N50')
PRIMARY=('R10','N10','R20','N20')

from models.msca import MSCA
from utils.dataset import RecDataset
from utils.dataloader import TrainDataLoader,EvalDataLoader
from utils.utils import init_seed
from modules.ranking import sha256_file
from pipelines.dataset_config import load_dataset_config
from pipelines.msca_assets import build_train_histories_and_validation
from diffusion_experiments.round19_safe_aux_hardneg.round19_core import CachedEvaluator
from diffusion_experiments.round20_positive_anchored_hardneg import round20_core as r20
from diffusion_experiments.round22_corrected_online_gdnsm.round22_diffusion import ConditionalNoiseMLP,LinearDDPM,epsilon_loss,generate_partial,generate_trajectory,T

def stats(x):
    x=np.asarray(x,np.float64)
    return {'mean':float(x.mean()),'median':float(np.median(x)),'p10':float(np.quantile(x,.1)),'p50':float(np.quantile(x,.5)),'p90':float(np.quantile(x,.9)),'p99':float(np.quantile(x,.99)),'min':float(x.min()),'max':float(x.max())}

def state_hash(sd):
    h=hashlib.sha256()
    for k in sorted(sd):
        v=sd[k].detach().cpu().contiguous();h.update(k.encode());h.update(str(v.dtype).encode());h.update(np.asarray(v.shape,dtype=np.int64).tobytes());h.update(v.numpy().tobytes())
    return h.hexdigest()

def capture_rng():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}

def restore_rng(x):
    random.setstate(x['python']);np.random.set_state(x['numpy']);torch.set_rng_state(x['torch'])
    if torch.cuda.is_available() and x['cuda']:torch.cuda.set_rng_state_all(x['cuda'])

def utility(m,b):return float(np.mean([(m[k]-b[k])/b[k] for k in PRIMARY]))
def delta_pack(m,b):return {'U':utility(m,b),'relative_delta':{k:float((m[k]-b[k])/b[k]) for k in ALL},'primary_positive_count':int(sum(m[k]>b[k] for k in PRIMARY)),'overall_positive_count':int(sum(m[k]>b[k] for k in ALL))}
def curriculum_count(e):return 0 if e<10 else 1 if e<20 else 2 if e<30 else 3

def _make_data(config):
    ds=RecDataset(config);train,valid,_=ds.split();train.inter_num=len(train.df);valid.inter_num=len(valid.df)
    td=TrainDataLoader(config,train,batch_size=config['train_batch_size'],shuffle=True)
    vd=EvalDataLoader(config,valid,additional_dataset=train,batch_size=config['eval_batch_size'])
    return ds,train,valid,td,vd

def create_shared_init(seed):
    ck,a=r20._historical_checkpoint(seed);config=copy.deepcopy(ck['config']);config['gpu_id']=0;config['use_gpu']=True;config['device']=torch.device('cuda:0');config['data_path']=str(ROOT/'data')+'/'
    ds,tr,va,td,vd=_make_data(config);init_seed(int(seed));td.pretrain_setup();model=MSCA(config,td).to(config['device'])
    sd={k:v.detach().cpu() for k,v in model.state_dict().items()};sh=state_hash(sd);rng=capture_rng();items=np.asarray(td.all_items,dtype=np.int64)
    path=RDIR/'assets'/f'ROUND22_INIT_SEED{seed}.pt'
    torch.save({'protocol':PROTOCOL,'seed':seed,'state_dict':sd,'state_hash':sh,'rng':rng,'all_items':items,'config':config,'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False},path)
    out={'seed':seed,'path':str(path),'file_sha256':sha256_file(path),'state_hash':sh,'train_interactions':int(len(tr.df)),'normal_steps_per_epoch':int(len(td)),'TEST_ACCESSED':False}
    del model;torch.cuda.empty_cache();gc.collect();return out

def instantiate_shared(seed):
    path=RDIR/'assets'/f'ROUND22_INIT_SEED{seed}.pt'
    if not path.exists():create_shared_init(seed)
    pack=torch.load(path,map_location='cpu',weights_only=False);config=copy.deepcopy(pack['config']);config['device']=torch.device('cuda:0');config['gpu_id']=0;config['use_gpu']=True
    ds,tr,va,td,vd=_make_data(config);model=MSCA(config,td).to(config['device']);model.load_state_dict(pack['state_dict'],strict=True);td.all_items=list(map(int,pack['all_items'].tolist()));restore_rng(pack['rng'])
    if state_hash(model.state_dict())!=pack['state_hash']:raise RuntimeError('shared init mismatch')
    return model,config,td,vd,pack

class TrainEvents:
    def __init__(self,model):
        cfg=load_dataset_config('baby');self.histories,self.pseudo,self.pusers,self.vusers,self.vsets=build_train_histories_and_validation(cfg['resolved_paths']['interaction'],model.n_users)
        U=[];P=[]
        for u,h in enumerate(self.histories):
            for i in h:U.append(u);P.append(int(i))
        self.u=np.asarray(U,np.int64);self.p=np.asarray(P,np.int64);self.n_items=model.n_items
    def sample(self,n,seed):
        rng=np.random.default_rng(seed);ix=rng.choice(len(self.u),size=min(n,len(self.u)),replace=False);return self.u[ix],self.p[ix]
    def random_unobserved(self,users,positive,seed):
        rng=np.random.default_rng(seed);out=np.empty(len(users),np.int64)
        for r,(u,p) in enumerate(zip(users,positive)):
            bad=set(self.histories[int(u)]);bad.add(int(p))
            while True:
                j=int(rng.integers(0,self.n_items))
                if j not in bad:out[r]=j;break
        return out

def forward_bundle(model):
    fu,fi,collab,struct,image,text=model.forward(test=False)
    _,iu=torch.split(image,[model.n_users,model.n_items],0);_,it=torch.split(text,[model.n_users,model.n_items],0)
    return {'fu':fu,'fi':fi,'collab':collab,'struct':struct,'image':image,'text':text,'image_item':iu,'text_item':it}

def msca_loss_from_bundle(model,interaction,b):
    users,pos,neg=interaction[0],interaction[1],interaction[2]
    bpr=model.cal_bpr_loss(b['fu'][users],b['fi'][pos],b['fi'][neg]);reg=model.cal_reg_loss()
    cu,ci=torch.split(b['collab'],[model.n_users,model.n_items],0);su,si=torch.split(b['struct'],[model.n_users,model.n_items],0);vu,vi=torch.split(b['image'],[model.n_users,model.n_items],0);tu,ti=torch.split(b['text'],[model.n_users,model.n_items],0)
    mu=model.cal_cl_loss(cu[users],vu[users],model.tau)+model.cal_cl_loss(cu[users],tu[users],model.tau)
    mi=model.cal_cl_loss(ci[pos],vi[pos],model.tau)+model.cal_cl_loss(ci[pos],ti[pos],model.tau)
    cl=model.cal_cl_loss(cu[users],su[users],model.tau)+model.cal_cl_loss(ci[pos],si[pos],model.tau)
    return bpr+model.cl_weight*(cl+mu+mi)+model.reg_weight*reg

def _scale(x):
    mu=x.mean(0);r=torch.sqrt(((x-mu)**2).mean())
    if float(r)<1e-6:raise RuntimeError('latent RMS < 1e-6')
    return mu.detach(),r.detach()

@torch.no_grad()
def latent_stats(model):
    model.eval();b=forward_bundle(model);muI,rI=_scale(b['fi']);muU,rU=_scale(b['fu']);muT,rT=_scale(b['text_item']);muV,rV=_scale(b['image_item'])
    return {'muI':muI,'rI':rI,'muU':muU,'rU':rU,'muT':muT,'rT':rT,'muV':muV,'rV':rV,'raw_item_norm':stats(b['fi'].norm(dim=1).cpu().numpy()),'raw_user_norm':stats(b['fu'].norm(dim=1).cpu().numpy())}

def z(x,mu,r):return (x-mu)/r
def inv_item(x,s):return s['muI']+s['rI']*x

def batch_diff_inputs(b,interaction,s):
    u,p=interaction[0],interaction[1]
    return z(b['fi'][p].detach(),s['muI'],s['rI']),z(b['fu'][u].detach(),s['muU'],s['rU']),z(b['text_item'][p].detach(),s['muT'],s['rT']),z(b['image_item'][p].detach(),s['muV'],s['rV'])

def isolated_diff_step(net,opt,sched,x0,hu,tc,vc,seed):
    rng=capture_rng();g=torch.Generator(device=x0.device);g.manual_seed(int(seed));net.train();opt.zero_grad(set_to_none=True);loss,_,_,_=epsilon_loss(net,sched,x0,hu,tc,vc,g)
    if not torch.isfinite(loss):raise RuntimeError('nonfinite diffusion loss')
    loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.0);opt.step();restore_rng(rng);return float(loss.detach())

def aux_bpr(hu,hp,negatives):
    if not negatives:return torch.zeros((),device=hu.device)
    xn=torch.stack(negatives,1);ps=(hu*hp).sum(1,keepdim=True);ns=torch.einsum('bd,bkd->bk',hu,xn);return -F.logsigmoid(ps-ns).mean()

@torch.no_grad()
def p0_audit(model,events,epoch0_norm,seed):
    model.eval();b=forward_bundle(model);u,p=events.sample(4096,20263200+seed);n=events.random_unobserved(u,p,20263300+seed);ut=torch.as_tensor(u,device=model.device);pt=torch.as_tensor(p,device=model.device);nt=torch.as_tensor(n,device=model.device)
    sp=(b['fu'][ut]*b['fi'][pt]).sum(1).cpu().numpy();sn=(b['fu'][ut]*b['fi'][nt]).sum(1).cpu().numpy();now={'item':stats(b['fi'].norm(dim=1).cpu().numpy()),'user':stats(b['fu'].norm(dim=1).cpu().numpy())}
    return {'epoch0_norm':epoch0_norm,'epoch10_norm':now,'positive_score':stats(sp),'random_unobserved_score':stats(sn),'mean_score_gap':float(sp.mean()-sn.mean()),'PASS':bool(sp.mean()>sn.mean())}

def make_diff(seed,device):
    rng=capture_rng();init_seed(20263400+int(seed));net=ConditionalNoiseMLP().to(device);opt=torch.optim.Adam(net.parameters(),lr=DIFF_LR);sched=LinearDDPM(device);restore_rng(rng);return net,opt,sched

def save_warmup(seed,model,net,config,record):
    p=RDIR/'outputs'/f'preflight_seed{seed}_warmup.pt';p.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'protocol':PROTOCOL,'seed':seed,'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'diffusion':{k:v.detach().cpu() for k,v in net.state_dict().items()},'config':config,'record':record,'TEST_ACCESSED':False},p);return p

def run_warmup(seed):
    model,config,td,vd,pack=instantiate_shared(seed);events=TrainEvents(model);net,dopt,dsched=make_diff(seed,model.device);ropt=torch.optim.Adam(model.parameters(),lr=float(config['learning_rate']),weight_decay=float(config['weight_decay']) if config['weight_decay'] is not None else 0.);fac=config['learning_rate_scheduler'];rsched=torch.optim.lr_scheduler.LambdaLR(ropt,lr_lambda=lambda e:fac[0]**(e/fac[1]))
    with torch.no_grad():
        b0=forward_bundle(model);epoch0_norm={'item':stats(b0['fi'].norm(dim=1).cpu().numpy()),'user':stats(b0['fu'].norm(dim=1).cpu().numpy())}
    trajectory=[];normal_hash={}
    for ep in range(WARMUP_EPOCHS):
        s=latent_stats(model);hh=hashlib.sha256();nl=[];dl=[];model.train()
        for bi,interaction in enumerate(td):
            u=interaction[0].detach().cpu().numpy();p=interaction[1].detach().cpu().numpy();n=interaction[2].detach().cpu().numpy();hh.update(u.tobytes());hh.update(p.tobytes());hh.update(n.tobytes())
            ropt.zero_grad(set_to_none=True);b=forward_bundle(model);loss=msca_loss_from_bundle(model,interaction,b);x0,hu,tc,vc=batch_diff_inputs(b,interaction,s);dl.append(isolated_diff_step(net,dopt,dsched,x0,hu,tc,vc,202635000+seed*10000+ep*1000+bi));loss.backward()
            if config['clip_grad_norm']:torch.nn.utils.clip_grad_norm_(model.parameters(),**config['clip_grad_norm'])
            ropt.step();nl.append(float(loss.detach()))
        rsched.step();normal_hash[str(ep)]=hh.hexdigest();trajectory.append({'epoch':ep,'normal_loss':float(np.mean(nl)),'diff_loss':float(np.mean(dl)),'rms':{k:float(s[k]) for k in ('rI','rU','rT','rV')}});print(json.dumps({'phase':'warmup','seed':seed,'epoch':ep,'normal':trajectory[-1]['normal_loss'],'diff':trajectory[-1]['diff_loss']}),flush=True)
    p0=p0_audit(model,events,epoch0_norm,seed);record={'protocol':PROTOCOL,'seed':seed,'warmup_epochs':WARMUP_EPOCHS,'P0':p0,'trajectory':trajectory,'normal_plan_hashes':normal_hash,'initial_state_hash':pack['state_hash'],'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
    cp=save_warmup(seed,model,net,config,record);record['checkpoint']=str(cp);record['checkpoint_sha256']=sha256_file(cp);(RDIR/'evidence'/f'ROUND22_WARMUP_SEED{seed}.json').write_text(json.dumps(record,indent=2)+'\n')
    del model,net,ropt,dopt;torch.cuda.empty_cache();gc.collect();return record

def load_warmup(seed):
    p=RDIR/'outputs'/f'preflight_seed{seed}_warmup.pt';obj=torch.load(p,map_location='cpu',weights_only=False);model,config,td,vd,pack=instantiate_shared(seed);model.load_state_dict(obj['model'],strict=True);net,dopt,sched=make_diff(seed,model.device);net.load_state_dict(obj['diffusion'],strict=True);return model,net,sched,TrainEvents(model),obj['record']

@torch.no_grad()
def prior_audit(model,sched,s,seed):
    model.eval();b=forward_bundle(model);zi=z(b['fi'],s['muI'],s['rI']);dimvar=float(zi.var(dim=0,unbiased=False).mean());zn=zi.norm(dim=1).cpu().numpy();rng=np.random.default_rng(20263600+seed);ix=rng.choice(len(zi),size=min(4096,len(zi)),replace=False);zsub=zi[torch.as_tensor(ix,device=model.device)]
    g=torch.Generator(device=model.device);g.manual_seed(20263700+seed);eps=torch.randn(zsub.shape,device=model.device,generator=g);tt=torch.full((len(zsub),),T-1,device=model.device,dtype=torch.long);zt=sched.q_sample(zsub,tt,eps);g2=torch.Generator(device=model.device);g2.manual_seed(20263800+seed);gauss=torch.randn(zsub.shape,device=model.device,generator=g2)
    ratio=float(np.median(zt.norm(dim=1).cpu().numpy())/np.median(gauss.norm(dim=1).cpu().numpy()))
    return {'raw_item_norm':s['raw_item_norm'],'normalized_item':{'mean_dimension_variance':dimvar,'norm':stats(zn)},'terminal_q':{'mean_dimension_std':float(zt.std(dim=0,unbiased=False).mean()),'norm':stats(zt.norm(dim=1).cpu().numpy())},'gaussian':{'norm':stats(gauss.norm(dim=1).cpu().numpy())},'terminal_gaussian_median_norm_ratio':ratio,'TERMINAL_PASS':bool(.75<=ratio<=1.25)}

@torch.no_grad()
def condition_audit(model,net,sched,s,events,seed,n=2048):
    model.eval();net.eval();b=forward_bundle(model);u,p=events.sample(n,20263900+seed);ut=torch.as_tensor(u,device=model.device);pt=torch.as_tensor(p,device=model.device);x0=z(b['fi'][pt],s['muI'],s['rI']);hu=z(b['fu'][ut],s['muU'],s['rU']);tc=z(b['text_item'][pt],s['muT'],s['rT']);vc=z(b['image_item'][pt],s['muV'],s['rV']);sh=hu.roll(1,0)
    g=torch.Generator(device=model.device);g.manual_seed(20264000+seed);t=torch.randint(0,T,(len(u),),device=model.device,generator=g);eps=torch.randn(x0.shape,device=model.device,generator=g);xt=sched.q_sample(x0,t,eps);zero_t=torch.zeros_like(tc);zero_v=torch.zeros_like(vc)
    pred={'true':net(xt,t,hu,tc,vc),'shuf_user':net(xt,t,sh,tc,vc),'no_modality':net(xt,t,hu,zero_t,zero_v),'shuf_no_modality':net(xt,t,sh,zero_t,zero_v)}
    mse={k:float(((v-eps)**2).mean()) for k,v in pred.items()};au=(mse['shuf_user']-mse['true'])/mse['shuf_user'];am=(mse['no_modality']-mse['true'])/mse['no_modality']
    return {'mse':mse,'user_relative_advantage':float(au),'modality_relative_advantage':float(am),'P2_PASS':bool(au>0 and am>0)}

def _batch_conditions(b,u,p,s):
    ut=torch.as_tensor(u,device=b['fu'].device);pt=torch.as_tensor(p,device=b['fu'].device);hu_raw=b['fu'][ut];hp=b['fi'][pt];hu=z(hu_raw,s['muU'],s['rU']);tc=z(b['text_item'][pt],s['muT'],s['rT']);vc=z(b['image_item'][pt],s['muV'],s['rV']);return hu_raw,hp,hu,tc,vc,pt

@torch.no_grad()
def calibrate_t0(model,net,sched,s,events,seed=999):
    model.eval();net.eval();b=forward_bundle(model);u,p=events.sample(1024,20264100);real_p99=s['raw_item_norm']['p99'];acc={t:{'margin':[],'norm':[],'above':0,'n':0} for t in T0_CANDIDATES}
    for st in range(0,len(u),128):
        ub,pb=u[st:st+128],p[st:st+128];hu_raw,hp,hu,tc,vc,pt=_batch_conditions(b,ub,pb,s);g=torch.Generator(device=model.device);g.manual_seed(20264200+st);tr=generate_trajectory(net,sched,hu,tc,vc,g);ps=(hu_raw*hp).sum(1);B=len(ub)
        for t0 in T0_CANDIDATES:
            ztv=tr[t0][2*B:];xn=inv_item(ztv,s);ns=(hu_raw*xn).sum(1);m=(ps-ns).cpu().numpy();nn=xn.norm(dim=1).cpu().numpy();a=acc[t0];a['margin'].append(m);a['norm'].append(nn);a['above']+=int((ns>=ps).sum());a['n']+=B
    out={}
    for t0,a in acc.items():
        m=np.concatenate(a['margin']);nn=np.concatenate(a['norm']);prob=a['above']/a['n'];ratio=float(np.quantile(nn,.99)/real_p99);adm=bool(np.median(m)>0 and .05<=prob<=.35 and .5<=ratio<=2.);out[str(t0)]={'margin':stats(m),'p_negative_ge_positive':float(prob),'synthetic_norm':stats(nn),'synthetic_p99_real_p99_ratio':ratio,'ADMISSIBLE':adm}
    good=[t for t in T0_CANDIDATES if out[str(t)]['ADMISSIBLE']];sel=None if not good else min(good,key=lambda t:out[str(t)]['margin']['median']);ev={'protocol':PROTOCOL,'seed':999,'TRAIN_ONLY':True,'VALIDATION_ACCESSED':False,'TEST_ACCESSED':False,'candidates':out,'selected_t0':sel,'PASS':sel is not None};(RDIR/'evidence'/'ROUND22_T0_CALIBRATION.json').write_text(json.dumps(ev,indent=2)+'\n');return ev

@torch.no_grad()
def generation_audits(model,net,sched,s,events,seed,t0,n=512):
    model.eval();net.eval();b=forward_bundle(model);u,p=events.sample(n,20264300+seed);hu_raw,hp,hu,tc,vc,pt=_batch_conditions(b,u,p,s);sh=hu.roll(1,0);modes=('V','T','TV');true={k:[] for k in modes};shuf_tv=[];ps_all=[];text_sim={k:[] for k in modes};vis_sim={k:[] for k in modes};margins={k:[] for k in modes};above={k:0 for k in modes};norms={k:[] for k in modes}
    for st in range(0,len(u),128):
        en=min(st+128,len(u));g=torch.Generator(device=model.device);g.manual_seed(20264400+seed*10+st);parts,_=generate_partial(net,sched,hu[st:en],tc[st:en],vc[st:en],t0,g);g2=torch.Generator(device=model.device);g2.manual_seed(20264400+seed*10+st);shparts,_=generate_partial(net,sched,sh[st:en],tc[st:en],vc[st:en],t0,g2);ps=(hu_raw[st:en]*hp[st:en]).sum(1);ps_all.append(ps.cpu())
        for k in modes:
            xn=inv_item(parts[k],s);true[k].append(xn.cpu());ns=(hu_raw[st:en]*xn).sum(1);mar=(ps-ns).cpu().numpy();margins[k].append(mar);above[k]+=int((ns>=ps).sum());norms[k].append(xn.norm(dim=1).cpu().numpy());text_sim[k].append(F.cosine_similarity(xn,b['text_item'][pt[st:en]],dim=1).cpu().numpy());vis_sim[k].append(F.cosine_similarity(xn,b['image_item'][pt[st:en]],dim=1).cpu().numpy())
        shuf_tv.append(inv_item(shparts['TV'],s).cpu())
    true={k:torch.cat(v) for k,v in true.items()};shx=torch.cat(shuf_tv);cos=F.cosine_similarity(true['TV'],shx,dim=1).numpy();dist=(true['TV']-shx).norm(dim=1).numpy();score_true=(hu_raw.cpu()*true['TV']).sum(1).numpy();score_shuf=(hu_raw.cpu()*shx).sum(1).numpy();hard={}
    for k in modes:
        m=np.concatenate(margins[k]);nn=np.concatenate(norms[k]);hard[k]={'margin':stats(m),'p_negative_ge_positive':float(above[k]/len(u)),'norm':stats(nn),'text_view_similarity':stats(np.concatenate(text_sim[k])),'image_view_similarity':stats(np.concatenate(vis_sim[k]))}
    hierarchy=bool(hard['TV']['margin']['median']<hard['V']['margin']['median'] and hard['TV']['margin']['median']<hard['T']['margin']['median']);catastrophic=bool(hard['TV']['p_negative_ge_positive']>=.50);ratio=float(max(hard[k]['norm']['p99'] for k in modes)/s['raw_item_norm']['p99']);user={'cosine_true_vs_shuffled':stats(cos),'l2_distance':stats(dist),'true_user_score':stats(score_true),'shuffled_generated_score_under_true_user':stats(score_shuf),'mean_score_effect':float((score_true-score_shuf).mean()),'mean_abs_score_effect':float(np.abs(score_true-score_shuf).mean())}
    return {'P3_user_generation':user,'P4_modality_supportive':{k:{'text':hard[k]['text_view_similarity'],'image':hard[k]['image_view_similarity']} for k in modes},'P5_hardness':{'modes':hard,'TV_HARDER_THAN_V_AND_T':hierarchy,'TV_CATASTROPHIC':catastrophic},'inverse_synthetic_p99_real_p99_ratio':ratio,'INVERSE_NORM_PASS':bool(.5<=ratio<=2.)}

def preflight(seed,selected_t0=None):
    model,net,sched,events,warm=load_warmup(seed);s=latent_stats(model);p0=warm['P0'];p1=prior_audit(model,sched,s,seed);p2=condition_audit(model,net,sched,s,events,seed)
    if seed==999:
        cal=calibrate_t0(model,net,sched,s,events,seed);selected_t0=cal['selected_t0']
        if selected_t0 is None:
            out={'seed':seed,'P0':p0,'P1':p1,'P2':p2,'t0_calibration':cal,'selected_t0':None,'PREFLIGHT_PASS':False,'BLOCK_REASON':'T0_CALIBRATION_FAIL','TEST_ACCESSED':False};(RDIR/'evidence'/f'ROUND22_PREFLIGHT_SEED{seed}.json').write_text(json.dumps(out,indent=2)+'\n');return out
    ga=generation_audits(model,net,sched,s,events,seed,int(selected_t0));p1['inverse_synthetic_p99_real_p99_ratio']=ga['inverse_synthetic_p99_real_p99_ratio'];p1['INVERSE_NORM_PASS']=ga['INVERSE_NORM_PASS'];p1['PASS']=bool(p1['TERMINAL_PASS'] and p1['INVERSE_NORM_PASS']);(RDIR/'evidence'/f'ROUND22_PRIOR_AUDIT_SEED{seed}.json').write_text(json.dumps({'seed':seed,**p1,'TEST_ACCESSED':False},indent=2)+'\n');(RDIR/'evidence'/f'ROUND22_CONDITION_AUDIT_SEED{seed}.json').write_text(json.dumps({'seed':seed,**p2,'P3':ga['P3_user_generation'],'P4':ga['P4_modality_supportive'],'TEST_ACCESSED':False},indent=2)+'\n');(RDIR/'evidence'/f'ROUND22_HARDNESS_AUDIT_SEED{seed}.json').write_text(json.dumps({'seed':seed,'t0':selected_t0,**ga['P5_hardness'],'inverse_norm_ratio':ga['inverse_synthetic_p99_real_p99_ratio'],'TEST_ACCESSED':False},indent=2)+'\n')
    reasons=[]
    if not p0['PASS']:reasons.append('RECOMMENDER_WARMUP_NOT_READY')
    if not p1['PASS']:reasons.append('PRIOR_SCALE_FAIL')
    if not p2['P2_PASS']:reasons.append('CONDITION_DENOISING_FAIL')
    if ga['P5_hardness']['TV_CATASTROPHIC']:reasons.append('P5_CATASTROPHIC')
    out={'protocol':PROTOCOL,'seed':seed,'selected_t0':int(selected_t0),'P0':p0,'P1':p1,'P2':p2,'P3':ga['P3_user_generation'],'P4':ga['P4_modality_supportive'],'P5':ga['P5_hardness'],'PREFLIGHT_PASS':not reasons,'BLOCK_REASONS':reasons,'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False};(RDIR/'evidence'/f'ROUND22_PREFLIGHT_SEED{seed}.json').write_text(json.dumps(out,indent=2)+'\n');del model,net;torch.cuda.empty_cache();gc.collect();return out

def trajectory_audit():
    model,config,td,vd,pack=instantiate_shared(999);net,dopt,sched=make_diff(999,model.device);s=latent_stats(model);interaction=next(iter(td));b=forward_bundle(model);x0,hu,tc,vc=batch_diff_inputs(b,interaction,s);x0=x0[:4];hu=hu[:4];tc=tc[:4];vc=vc[:4];isolated_diff_step(net,dopt,sched,x0,hu,tc,vc,20264500);g=torch.Generator(device=model.device);g.manual_seed(20264600);tr=generate_trajectory(net,sched,hu,tc,vc,g);g2=torch.Generator(device=model.device);g2.manual_seed(20264600);tr2=generate_trajectory(net,sched,hu,tc,vc,g2);norm={str(k):stats(v.norm(dim=1).cpu().numpy()) for k,v in tr.items()};same=max(float((tr[k]-tr2[k]).abs().max()) for k in tr);dist=float((tr[4]-tr[0]).norm(dim=1).mean());out={'T':T,'trajectory_keys':sorted(tr.keys(),reverse=True),'state_count':len(tr),'shape':list(tr[0].shape),'finite_all':bool(all(torch.isfinite(v).all() for v in tr.values())),'same_noise_max_abs_diff':same,'t0_probe':4,'partial_full_mean_l2':dist,'norm_trajectory':norm,'PASS':bool(len(tr)==25 and same==0 and dist>0 and all(torch.isfinite(v).all() for v in tr.values())),'TEST_ACCESSED':False};(RDIR/'evidence'/'ROUND22_TRAJECTORY_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');del model,net,dopt;torch.cuda.empty_cache();gc.collect();return out

def run_smoke():
    model,config,td,vd,pack=instantiate_shared(999);net,dopt,sched=make_diff(999,model.device);s=latent_stats(model);interaction=next(iter(td));model.train();ropt=torch.optim.Adam(model.parameters(),lr=float(config['learning_rate']),weight_decay=float(config['weight_decay']));ropt.zero_grad(set_to_none=True);b=forward_bundle(model);loss=msca_loss_from_bundle(model,interaction,b);direct=model.calculate_loss(interaction);parity=float(abs(loss.detach()-direct.detach()));x0,hu,tc,vc=batch_diff_inputs(b,interaction,s);before=torch.get_rng_state().clone();dl=isolated_diff_step(net,dopt,sched,x0[:16],hu[:16],tc[:16],vc[:16],20264700);rng_same=bool(torch.equal(before,torch.get_rng_state()));g=torch.Generator(device=model.device);g.manual_seed(20264800);parts,tr=generate_partial(net,sched,hu[:16],tc[:16],vc[:16],4,g);raw={k:inv_item(v,s) for k,v in parts.items()};u=interaction[0][:16];p=interaction[1][:16];hn=aux_bpr(b['fu'][u],b['fi'][p],[raw['V']]);total=loss+LAMBDA_HN*hn;total.backward();finite=bool(torch.isfinite(total) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None));ropt.step();out={'protocol':PROTOCOL,'loss_parity_abs':parity,'diff_loss':dl,'dedicated_rng_preserved_global_torch':rng_same,'scale_rms':{k:float(s[k]) for k in ('rI','rU','rT','rV')},'normalized_target_norm':stats(x0[:16].norm(dim=1).detach().cpu().numpy()),'inverse_V_norm':stats(raw['V'].norm(dim=1).cpu().numpy()),'trajectory_keys':sorted(tr.keys(),reverse=True),'partial_t0':4,'partial_full_mean_l2':float((tr[4]-tr[0]).norm(dim=1).mean()),'online_immediate_use':True,'stale_synthetic_reuse_count':0,'finite_gradients':finite,'curriculum':{'0':curriculum_count(0),'10':curriculum_count(10),'20':curriculum_count(20),'30':curriculum_count(30)},'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False};out['PASS']=bool(parity<1e-6 and rng_same and finite and len(tr)==25 and out['partial_full_mean_l2']>0 and out['curriculum']=={'0':0,'10':1,'20':2,'30':3});(RDIR/'evidence'/'ROUND22_SMOKE.json').write_text(json.dumps(out,indent=2)+'\n');del model,net,ropt,dopt;torch.cuda.empty_cache();gc.collect();return out

def preflight_summary():
    a=json.loads((RDIR/'evidence'/'ROUND22_PREFLIGHT_SEED999.json').read_text());b=json.loads((RDIR/'evidence'/'ROUND22_PREFLIGHT_SEED1000.json').read_text());au=[a['P2']['user_relative_advantage'],b['P2']['user_relative_advantage']];p3=[a['P3']['mean_score_effect'],b['P3']['mean_score_effect']];cross_user=bool(au[0]>0 and au[1]>0);p3_same=bool(p3[0]*p3[1]>0 and abs(p3[0])>0 and abs(p3[1])>0);ok=bool(a['PREFLIGHT_PASS'] and b['PREFLIGHT_PASS'] and cross_user);out={'protocol':PROTOCOL,'selected_t0':a['selected_t0'],'seed999_pass':a['PREFLIGHT_PASS'],'seed1000_pass':b['PREFLIGHT_PASS'],'epsilon_user_advantages':{'999':au[0],'1000':au[1]},'user_condition_same_direction':cross_user,'generation_user_score_effects':{'999':p3[0],'1000':p3[1]},'generation_user_effect_same_direction':p3_same,'PREFLIGHT_PASS':ok,'FORMAL_ALLOWED':ok,'TEST_ACCESSED':False};(RDIR/'evidence'/'ROUND22_PREFLIGHT_SUMMARY.json').write_text(json.dumps(out,indent=2)+'\n');return out
