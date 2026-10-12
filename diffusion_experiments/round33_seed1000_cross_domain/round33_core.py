from __future__ import annotations
import copy, gc, hashlib, json, random
from pathlib import Path
import numpy as np
import torch

from utils.configurator import Config
from utils.dataset import RecDataset
from utils.dataloader import TrainDataLoader, EvalDataLoader
from utils.utils import init_seed
from models.msca import MSCA
from pipelines.dataset_config import load_dataset_config
from pipelines.msca_assets import build_train_histories_and_validation
from pipelines.evaluate_coliftrec import build_test_eval
from diffusion_experiments.round22_corrected_online_gdnsm import round22_core as r22
from diffusion_experiments.round24_stable_boundary_matched import round24_core as r24
from diffusion_experiments.round19_safe_aux_hardneg.round19_core import (
    _params, l2_normalize_rows, histories_csr, attribute_z, build_item_matrices,
    build_profiles, fit_backgrounds, score_coliftrec, rank_by_score,
    topk_from_embeddings, metrics_at, row_zscore,
)

ROOT=Path(__file__).resolve().parents[2]
RDIR=ROOT/'diffusion_experiments/round33_seed1000_cross_domain'
EVID=RDIR/'evidence'; LOGS=RDIR/'logs'; OUT=RDIR/'outputs'; ASSETS=RDIR/'assets'
for p in (EVID,LOGS,OUT,ASSETS): p.mkdir(parents=True,exist_ok=True)
PROTOCOL='ROUND33_SEED1000_CROSS_DOMAIN_M25_V1'
SEED=1000
DATASETS=('sports','elec')
VARIANTS=('B0','M25')
WARMUP_EPOCHS=10
FORMAL_EPOCHS=60
DIFF_STEPS_ACTIVE=2
LAMBDA_HN=.20
HALF_MARGIN=.25
THETA_MAX_DEG=5.0
EPS=1e-8
LOCAL_TS=(1,2,3,4,5)
ALL=('R10','N10','R20','N20','R50','N50')
PRIMARY=('R10','N10','R20','N20')
BANDS=r24.BANDS; BAND_NAMES=r24.BAND_NAMES

forward_bundle=r24.forward_bundle
msca_loss_from_bundle=r24.msca_loss_from_bundle
latent_stats=r24.latent_stats
valid_mask=r24.valid_mask
mine_current_ids=r24.mine_current_ids
rank_offsets=r24.rank_offsets
curriculum_count=r24.curriculum_count
aux_bpr=r24.aux_bpr
state_hash=r24.state_hash
stats=r24.stats
make_diff=r24.make_diff
z=r22.z
inv_item=r24.inv_item
sha256_file=r24.sha256_file


def json_write(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,default=lambda x:float(x) if isinstance(x,np.generic) else str(x))+'\n')


def delta_pack(m,b):
    rel={k:float((m[k]-b[k])/b[k]) for k in ALL}
    return {'U':float(np.mean([rel[k] for k in PRIMARY])),
            'relative_delta':rel,
            'absolute_delta':{k:float(m[k]-b[k]) for k in ALL},
            'primary_positive_count':int(sum(m[k]>b[k] for k in PRIMARY)),
            'overall_positive_count':int(sum(m[k]>b[k] for k in ALL))}


def protocol_hash():
    h=hashlib.sha256()
    for name in ('round33_core.py','round33_diffusion.py','round33_train.py','run_round33.py'):
        p=RDIR/name
        if p.exists(): h.update(name.encode()); h.update(p.read_bytes())
    h.update(json.dumps({'seed':SEED,'datasets':DATASETS,'lambda_HN':LAMBDA_HN,
        'half_margin':HALF_MARGIN,'theta_max_deg':THETA_MAX_DEG,'epochs':FORMAL_EPOCHS,
        'local_t':LOCAL_TS,'diff_steps_active':DIFF_STEPS_ACTIVE},sort_keys=True).encode())
    return h.hexdigest()


def _config(dataset):
    cfg=Config('MSCA',dataset,{
        'gpu_id':0,'use_gpu':True,'seed':[SEED],
        'data_path':str(ROOT/'data')+'/',
        'checkpoint_dir':str(OUT/dataset/'native_checkpoints'),
        'recommend_topk':str(OUT/dataset/'recommend_topk')+'/',
    })
    # Direct experiment construction: resolve every one-point hyperparameter list.
    for k in cfg['hyper_parameters']:
        v=cfg[k]
        if isinstance(v,(list,tuple)) and len(v)==1: cfg[k]=v[0]
    cfg['seed']=SEED
    cfg['gpu_id']=0; cfg['use_gpu']=True; cfg['device']=torch.device('cuda:0')
    return cfg


def _make_data(config):
    ds=RecDataset(config); train,valid,_=ds.split()
    train.inter_num=len(train.df); valid.inter_num=len(valid.df)
    td=TrainDataLoader(config,train,batch_size=config['train_batch_size'],shuffle=True)
    vd=EvalDataLoader(config,valid,additional_dataset=train,batch_size=config['eval_batch_size'])
    return ds,train,valid,td,vd


def _capture_rng(): return r22.capture_rng()
def _restore_rng(x): return r22.restore_rng(x)


def create_shared_init(dataset):
    dataset=str(dataset)
    config=_config(dataset)
    ds,tr,va,td,vd=_make_data(config)
    init_seed(SEED); td.pretrain_setup(); model=MSCA(config,td).to(config['device'])
    sd={k:v.detach().cpu() for k,v in model.state_dict().items()}; sh=state_hash(sd); rng=_capture_rng()
    items=np.asarray(td.all_items,dtype=np.int64)
    p=ASSETS/f'{dataset}_seed{SEED}_init.pt'
    torch.save({'protocol':PROTOCOL,'dataset':dataset,'seed':SEED,'state_dict':sd,
        'state_hash':sh,'rng':rng,'all_items':items},p)
    out={'dataset':dataset,'seed':SEED,'path':str(p),'sha256':sha256_file(p),'state_hash':sh,
         'n_users':int(model.n_users),'n_items':int(model.n_items),'train_interactions':int(len(tr.df)),
         'steps_per_epoch':int(len(td))}
    json_write(EVID/f'ROUND33_{dataset.upper()}_INIT.json',out)
    del model; torch.cuda.empty_cache(); gc.collect(); return out


def instantiate_shared(dataset):
    p=ASSETS/f'{dataset}_seed{SEED}_init.pt'
    if not p.exists(): create_shared_init(dataset)
    pack=torch.load(p,map_location='cpu',weights_only=False)
    config=_config(dataset); ds,tr,va,td,vd=_make_data(config)
    model=MSCA(config,td).to(config['device']); model.load_state_dict(pack['state_dict'],strict=True)
    td.all_items=list(map(int,pack['all_items'].tolist())); _restore_rng(pack['rng'])
    if state_hash(model.state_dict())!=pack['state_hash']: raise RuntimeError('shared init mismatch')
    return model,config,td,vd,pack


class TrainEvents:
    def __init__(self,dataset,model):
        cfg=load_dataset_config(dataset)
        self.histories,self.pseudo,self.pusers,self.vusers,self.vsets=build_train_histories_and_validation(
            cfg['resolved_paths']['interaction'],model.n_users)
        U=[];P=[]
        for u,h in enumerate(self.histories):
            for i in h: U.append(u);P.append(int(i))
        self.u=np.asarray(U,np.int64);self.p=np.asarray(P,np.int64);self.n_items=model.n_items
    def sample(self,n,seed):
        rng=np.random.default_rng(seed);ix=rng.choice(len(self.u),size=min(n,len(self.u)),replace=False)
        return self.u[ix],self.p[ix]


class DomainEvaluator:
    def __init__(self,dataset,model,split='validation'):
        self.dataset=dataset; self.device=model.device; self.split=split
        self.cfg=load_dataset_config(dataset); p=self.cfg['resolved_paths']; self.paths=p
        self.ccfg=self.cfg['coliftrec']; self.params=_params(self.ccfg)
        self.enabled={m:bool(self.ccfg[m]['enabled']) for m in ('text','attribute','visual')}
        hist,pseudo,pusers,vusers,vsets=build_train_histories_and_validation(p['interaction'],model.n_users)
        self.histories=hist; self.pseudo=pseudo; self.pusers=pusers
        if split=='validation': self.users=vusers; self.sets=vsets
        elif split=='test':
            th,tu,ts=build_test_eval(p['interaction'],model.n_users)
            if th!=hist: raise RuntimeError('TRAIN history mismatch for Test')
            self.users=tu; self.sets=ts
        else: raise ValueError(split)
        self.n_items=model.n_items
        self.text=l2_normalize_rows(np.asarray(np.load(p['text_feature'],mmap_mode='r'),dtype=np.float32))
        self.visual=l2_normalize_rows(np.asarray(np.load(p['visual_feature'],mmap_mode='r'),dtype=np.float32))
        hf=histories_csr(self.histories,self.n_items,users=self.users,mean=True)
        hp=histories_csr(self.pseudo,self.n_items,users=self.pusers,mean=True)
        self.text_full=l2_normalize_rows(np.asarray(hf@self.text,dtype=np.float32))
        self.text_pseudo=l2_normalize_rows(np.asarray(hp@self.text,dtype=np.float32))
        self.vis_full=l2_normalize_rows(np.asarray(hf@self.visual,dtype=np.float32))
        self.vis_pseudo=l2_normalize_rows(np.asarray(hp@self.visual,dtype=np.float32))
        ac=self.ccfg['attribute']
        self.mats,_=build_item_matrices(p['metadata'],self.n_items,min_df=int(ac.get('tfidf_min_df',2)),
            max_df=float(ac.get('tfidf_max_df',.8)),description_len=int(ac.get('description_len',128)),weights=ac.get('weights'))
        self.full_profiles=build_profiles(self.mats,self.histories,self.n_items)
        self.pseudo_profiles=build_profiles(self.mats,self.pseudo,self.n_items)
        self.aw=ac.get('weights')
    def sem(self,feat,prof,items,batch=128):
        outs=[]
        for st in range(0,len(items),batch):
            en=min(st+batch,len(items))
            outs.append(np.einsum('bld,bd->bl',feat[items[st:en]],prof[st:en],optimize=True).astype(np.float32))
        return row_zscore(np.concatenate(outs,0))
    @torch.no_grad()
    def evaluate(self,model,return_rank=False):
        model.eval(); fu,fi=model.forward(test=True)
        items,scores=topk_from_embeddings(fu,fi,self.users,self.histories,100,1024)
        pitems,_=topk_from_embeddings(fu,fi,self.pusers,self.pseudo,100,1024)
        back=metrics_at(items,self.users,self.sets)
        ztp=self.sem(self.text,self.text_pseudo,pitems,128)
        zte=self.sem(self.text,self.text_full,items,128)
        zvp=self.sem(self.visual,self.vis_pseudo,pitems,64)
        zve=self.sem(self.visual,self.vis_full,items,64)
        zap,_=attribute_z(self.mats,self.pseudo_profiles,self.pusers,pitems,256,self.aw)
        zae,_=attribute_z(self.mats,self.full_profiles,self.users,items,256,self.aw)
        bg=fit_backgrounds(pitems,ztp,zap,zvp,self.n_items)
        fs,_=score_coliftrec(scores,items,zte,zae,zve,bg,self.params,self.enabled)
        rank=rank_by_score(items,fs); full=metrics_at(rank,self.users,self.sets)
        out={'backbone':back,'colift':full,'user_count':int(len(self.users)),
             'candidate_shape':[int(items.shape[0]),int(items.shape[1])],'diffusion_inference_calls':0}
        if return_rank: out['ranked_items']=rank
        return out


def save_best(model,net,dataset,variant,epoch,score):
    p=OUT/dataset/variant/'best.pt'; p.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'protocol':PROTOCOL,'dataset':dataset,'seed':SEED,'variant':variant,'epoch':int(epoch),
        'full_colift_R20':float(score),'model_state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
        'diffusion_state_dict':None if net is None else {k:v.detach().cpu() for k,v in net.state_dict().items()}},p)
    return p
