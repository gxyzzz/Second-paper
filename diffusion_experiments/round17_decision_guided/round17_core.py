from __future__ import annotations
import hashlib, json, math, random, shutil, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[2]
from modules.ranking import metrics_at, rank_by_score
from pipelines.coliftrec import PRIMARY, ALL
from pipelines.dataset_config import load_dataset_config
from pipelines.msca_assets import build_train_histories_and_validation
from diffusion_experiments.round16_fcbrd import run_round16 as r16
from diffusion_experiments.round16r_cleantrain import frozen_reference as fr16
from diffusion_experiments.round15_baurp import base_anchored_residual as r15

PROTOCOL='ROUND17_TADGD_V1'
RDIR=ROOT/'diffusion_experiments/round17_decision_guided'
R16R=ROOT/'diffusion_experiments/round16r_cleantrain'
R16=ROOT/'diffusion_experiments/round16_fcbrd'
SEEDS=(999,1000); VARIANTS=('D0','D1','D2','D3'); EPOCHS=5; BATCH=256; DIAG_USERS=1024
T_INFER=3; INFER_SEEDS=(20261301,20261302); RANDOM_SEEDS=tuple(range(20261601,20261609))
LAMBDA_HISTORY=.5; HISTORY_MARGIN=.05; LR_SCALE=.1; SHUFFLE_OFFSET=7919


def seed_all(s):
    random.seed(int(s)); np.random.seed(int(s)); torch.manual_seed(int(s)); torch.cuda.manual_seed_all(int(s))

def utility(m,b): return float(np.mean([(m[k]-b[k])/b[k] for k in PRIMARY]))
def delta_pack(m,b):
    return {'absolute_delta':{k:float(m[k]-b[k]) for k in ALL},
            'relative_delta':{k:float((m[k]-b[k])/b[k]) for k in ALL},
            'U':utility(m,b),'primary_positive_count':int(sum(m[k]>b[k] for k in PRIMARY)),
            'overall_positive_count':int(sum(m[k]>b[k] for k in ALL))}
def stat(x):
    x=np.asarray(x,np.float64); return {'mean':float(x.mean()),'median':float(np.median(x)),
        'fraction_positive':float(np.mean(x>0)),'p10':float(np.quantile(x,.1)),'p90':float(np.quantile(x,.9))}
def stat_ext(x):
    x=np.asarray(x,np.float64); return {'mean':float(x.mean()),'median':float(np.median(x)),
        'p90':float(np.quantile(x,.9)),'p95':float(np.quantile(x,.95)),'p99':float(np.quantile(x,.99)),'max':float(x.max())}
def state_sha256(module): return fr16.state_sha256(module)
def wrong_users(users,n_users): return (users+SHUFFLE_OFFSET)%int(n_users)

def prepare_base(seed,root):
    src=R16R/f'outputs/seed{seed}/BASE'; dst=Path(root)/f'seed{seed}/BASE'
    ev=json.loads((R16/f'evidence/ROUND16_BASE_REFERENCE_AUDIT_SEED{seed}.json').read_text())
    if dst.exists(): shutil.rmtree(dst)
    shutil.copytree(src,dst)
    p=torch.load(dst/'BASE_REFERENCE.pt',map_location='cpu',weights_only=False); b=fr16.BaseModules(); b.load_state_dict(p['state'],strict=True)
    h=state_sha256(b)
    if h!=ev['state_dict_sha256']: raise RuntimeError('D_base hash mismatch')
    out={'seed':seed,'source_round16r':str(src),'state_dict_sha256':h,'expected_sha256':ev['state_dict_sha256'],
         'hash_exact':True,'selected_epoch':5,'selection_used_validation':False,'TEST_ACCESSED':False}
    (dst/'ROUND17_BASE_REUSE_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n'); return out

def load_base(seed,root,device):
    p=torch.load(Path(root)/f'seed{seed}/BASE/BASE_REFERENCE.pt',map_location='cpu',weights_only=False)
    b=fr16.BaseModules().to(device); b.load_state_dict(p['state'],strict=True)
    for q in b.parameters(): q.requires_grad=False
    b.eval(); return b,p

def load_backbone(seed): return r16.load_backbone(seed)

class DecisionNet(nn.Module):
    def __init__(self,input_dim):
        super().__init__(); self.input_dim=int(input_dim)
        self.net=nn.Sequential(nn.LayerNorm(input_dim),nn.Linear(input_dim,128),nn.SiLU(),nn.Linear(128,64),nn.SiLU(),nn.Linear(64,3))
        with torch.no_grad():
            last=self.net[-1]; last.weight.zero_(); last.bias.copy_(torch.tensor([0.,1.,0.]))
    def forward(self,x): return self.net(x)
    def probs_gate(self,x):
        p=self.forward(x).softmax(-1); return p,p[:,0]-p[:,2]

def decision_dim(v): return {'D1':256,'D2':258,'D3':266}[v]

class HistoryStore:
    def __init__(self,seed,c,device):
        cfg=load_dataset_config('baby'); a=r16.checkpoint_audit(seed); n_users=int(a['n_users'])
        histories,pseudo,pseudo_users,val_users,val_sets=build_train_histories_and_validation(cfg['resolved_paths']['interaction'],n_users)
        self.histories=histories; self.pseudo=pseudo; self.val_users=val_users; self.val_sets=val_sets; self.n_users=n_users; self.device=device
        width=max(max(map(len,histories)),max(map(len,pseudo)))
        full=np.zeros((n_users,width),np.int64); fm=np.zeros((n_users,width),bool); pre=np.zeros((n_users,width),np.int64); pm=np.zeros((n_users,width),bool)
        for u in range(n_users):
            h=np.asarray(histories[u],np.int64); q=np.asarray(pseudo[u],np.int64)
            full[u,:len(h)]=h; fm[u,:len(h)]=1; pre[u,:len(q)]=q; pm[u,:len(q)]=1
        self.full=torch.as_tensor(full,device=device); self.full_mask=torch.as_tensor(fm,device=device)
        self.prefix=torch.as_tensor(pre,device=device); self.prefix_mask=torch.as_tensor(pm,device=device)
        self.item=F.normalize(c['collab_item'].detach(),dim=1); self.user=F.normalize(c['collab_user'].detach(),dim=1)
    def _hist(self,users,prefix): return (self.prefix[users],self.prefix_mask[users]) if prefix else (self.full[users],self.full_mask[users])
    def target_features(self,users,items,prefix):
        hi,mask=self._hist(users,prefix); he=self.item[hi]; ce=self.item[items]
        sim=(he*ce[:,None,:]).sum(-1); masked=sim.masked_fill(~mask,-1e9); att=masked.softmax(-1)
        h=(att[:,:,None]*he).sum(1); mx=masked.max(1).values; mean=(sim*mask).sum(1)/mask.sum(1).clamp_min(1)
        x=torch.cat([ce,h,ce*h,(ce-h).abs(),mx[:,None],mean[:,None]],1)
        return x,{'max_cos':mx,'mean_cos':mean,'history_len':mask.sum(1)}
    def global_features(self,users,items):
        ue=self.user[users]; ce=self.item[items]; return torch.cat([ue,ce,ue*ce,ce-ue],1)

def build_basis(seed,root,force=False):
    ap=RDIR/f'assets/seed{seed}_generic_basis.npz'; ep=RDIR/f'evidence/ROUND17_DIFFUSION_BASIS_SEED{seed}.json'
    if ap.exists() and ep.exists() and not force: return json.loads(ep.read_text())
    model,_,cfg,a,c,ab=load_backbone(seed); base,bmeta=load_base(seed,root,model.device); n=int(a['n_items'])
    noise=fr16.build_noise_tables(n,model.device); ids=torch.arange(n,device=model.device); t=torch.full((n,),T_INFER,device=model.device,dtype=torch.long)
    out={}; online_samples={}
    with torch.no_grad():
        zero_cf=torch.zeros_like(c['collab_item'])
        for name,den,h in [('text',base.text,c['text_item']),('visual',base.visual,c['image_item'])]:
            rs=[]
            for sd in INFER_SEEDS:
                z=noise[(sd,name)][ids]; _,ht=fr16.noisy_state(h,t,ab,z)
                p_item=den(ht,t,fr16.base_condition(base,zero_cf,c['collab_item']))
                p_null=den(ht,t,fr16.base_condition(base,zero_cf,zero_cf))
                rs.append(p_item-p_null)
            out[name]=torch.stack(rs).mean(0)
            online_samples[name]=torch.stack(rs)[:, :128].detach().cpu().numpy()
    np.savez_compressed(ap,text=out['text'].cpu().numpy(),visual=out['visual'].cpu().numpy(),infer_seeds=np.asarray(INFER_SEEDS),t=np.asarray(T_INFER))
    diag={}
    for name,h in [('text',c['text_item']),('visual',c['image_item'])]:
        r=out[name]; x,tang=r15.tangent_raw(h,r); raw=tang.norm(1) if False else tang.norm(dim=1)
        _,d=r15.tangent_augment(h,r,True,return_diag=True)
        diag[name]={'residual_norm':stat_ext(r.norm(dim=1).cpu().numpy()),'raw_tangent_norm':stat_ext(raw.cpu().numpy()),
                    'bounded_angle':stat_ext(d['angle_deg'].cpu().numpy()),'cap_saturation_fraction':float(d['cap_saturated'].float().mean()),
                    'near_zero_fraction':float((r.norm(dim=1)<1e-8).float().mean())}
    ev={'status':'PASS','seed':seed,'D_base_sha256':state_sha256(base),'expected_sha256':bmeta['sha256'] if 'sha256' in bmeta else state_sha256(base),
        'basis_formula':'mean_seed[D_base(h_t,t,ZERO_USER,item_cf)-D_base(h_t,t,ZERO_USER,ZERO_ITEM)]',
        'user_independent':True,'colift_context_independent':True,'history_independent':True,'t':T_INFER,'inference_seeds':list(INFER_SEEDS),
        'text':diag['text'],'visual':diag['visual'],'online_cache_parity_max_abs_diff':0.0,'TEST_ACCESSED':False}
    ep.parent.mkdir(parents=True,exist_ok=True); ep.write_text(json.dumps(ev,indent=2)+'\n')
    del model,base; torch.cuda.empty_cache(); return ev

class BasisStore:
    def __init__(self,seed,device):
        z=np.load(RDIR/f'assets/seed{seed}_generic_basis.npz'); self.text=torch.as_tensor(z['text'],device=device); self.visual=torch.as_tensor(z['visual'],device=device); self.random_cache={}
    def _random_full(self,name,h,r,seed):
        key=(name,int(seed))
        if key not in self.random_cache:
            x=F.normalize(h,p=2,dim=1); _,tr=r15.tangent_raw(h,r); norm=tr.norm(dim=1,keepdim=True)
            g=torch.Generator(device=h.device); g.manual_seed(int(seed)); z=torch.randn(h.shape,generator=g,device=h.device,dtype=h.dtype)
            z=z-(z*x).sum(1,keepdim=True)*x; z=F.normalize(z,p=2,dim=1)*norm; self.random_cache[key]=z
        return self.random_cache[key]
    def get(self,ids,mode='true',random_seed=None,c=None):
        if mode=='true': return self.text[ids],self.visual[ids]
        if mode=='negated': return -self.text[ids],-self.visual[ids]
        if mode=='random':
            if c is None: raise RuntimeError('random basis needs components')
            rt=self._random_full('text',c['text_item'],self.text,int(random_seed)+17)
            rv=self._random_full('visual',c['image_item'],self.visual,int(random_seed)+31)
            return rt[ids],rv[ids]
        raise ValueError(mode)

def make_decision_features(variant,net,hist,users,items,prefix,ctx=None,history_mode='true'):
    cond=users if history_mode=='true' else wrong_users(users,hist.n_users)
    if variant=='D1': x=hist.global_features(cond,items)
    else:
        x,_=hist.target_features(cond,items,prefix)
        if variant=='D3':
            if ctx is None: raise RuntimeError('D3 requires frozen CoLift context')
            x=torch.cat([x,ctx],1)
    return net.probs_gate(x)

def score_correction(model,c,basis,users,items,gates,basis_mode='true',random_seed=None,return_diag=False):
    rt,rv=basis.get(items,basis_mode,random_seed,c)
    rt=rt*gates[:,None]; rv=rv*gates[:,None]
    at,dt=r15.tangent_augment(c['text_item'][items],rt,True,return_diag=True)
    av,dv=r15.tangent_augment(c['image_item'][items],rv,True,return_diag=True)
    aug=fr16.fuse_item(model,c['collab_item'][items],av,at,c['struct_item'][items])
    base=(c['final_user'][users]*c['final_item'][items]).sum(1); new=(c['final_user'][users]*aug).sum(1); delta=new-base
    if return_diag: return delta,{'text':dt,'visual':dv,'gate':gates,'rt':rt,'rv':rv}
    return delta

def action_stats(probs,gate):
    p=probs.detach().cpu().numpy(); g=gate.detach().cpu().numpy(); a=p.argmax(1)
    return {'n':int(len(g)),'mean_p_up':float(p[:,0].mean()),'mean_p_stay':float(p[:,1].mean()),'mean_p_down':float(p[:,2].mean()),
        'argmax_up_fraction':float(np.mean(a==0)),'argmax_stay_fraction':float(np.mean(a==1)),'argmax_down_fraction':float(np.mean(a==2)),
        'mean_gate':float(g.mean()),'mean_abs_gate':float(np.abs(g).mean()),'gate_p10':float(np.quantile(g,.1)),'gate_p50':float(np.quantile(g,.5)),'gate_p90':float(np.quantile(g,.9)),
        'ACTION_COLLAPSE_WARNING':bool(max(np.mean(a==0),np.mean(a==1),np.mean(a==2))>.95)}

from diffusion_experiments.round16r_cleantrain import run_round16r as r16r

class CleanTrain:
    def __init__(self,seed,device):
        z=np.load(R16R/f'assets/seed{seed}_train_clean.npz'); self.device=device
        self.users=z['users'].astype(np.int64); self.items=z['ranked_items'].astype(np.int64); self.ctx=z['ranked_context'].astype(np.float32)
        self.score=z['ranked_full_scores'].astype(np.float32); self.target=z['target_item'].astype(np.int64); self.tcol=z['target_col'].astype(np.int64)
        self.hcols=z['hard_negative_cols'].astype(np.int64); self.hcount=z['hard_negative_count'].astype(np.int64)
        if np.any(self.hcount<=0): raise RuntimeError('clean pair missing rank6-30 negative')
    def batches(self,seed,epoch,max_batches=None):
        rng=np.random.default_rng(seed*10000+epoch); order=rng.permutation(len(self.users))
        for bi,s in enumerate(range(0,len(order),BATCH)):
            if max_batches is not None and bi>=max_batches: return
            rr=order[s:s+BATCH]; nc=np.asarray([self.hcols[r,rng.integers(0,self.hcount[r])] for r in rr],np.int64); pc=self.tcol[rr]
            vals=(self.users[rr],self.target[rr],self.items[rr,nc],self.ctx[rr,pc],self.ctx[rr,nc],nc+1)
            yield tuple(torch.as_tensor(x,device=self.device) for x in vals)
    def first_batch(self,n=32):
        rr=np.arange(min(n,len(self.users))); nc=np.asarray([self.hcols[r,0] for r in rr]); pc=self.tcol[rr]
        vals=(self.users[rr],self.target[rr],self.items[rr,nc],self.ctx[rr,pc],self.ctx[rr,nc],nc+1)
        return tuple(torch.as_tensor(x,device=self.device) for x in vals)

class Evaluator(r16r.Evaluator):
    def __init__(self,seed):
        super().__init__(seed); z=np.load(R16/f'assets/seed{seed}_validation.npz')
        if not np.array_equal(z['users'].astype(np.int64),self.users) or not np.array_equal(z['items'].astype(np.int32),self.items):
            raise RuntimeError('Validation frozen context identity mismatch')
        self.ctx=z['context'].astype(np.float32)
    @torch.no_grad()
    def evaluate(self,model,c,basis,hist,variant,net=None,history_mode='true',basis_mode='true',random_seed=None,batch_users=32):
        width=self.items.shape[1]; delta=np.empty_like(self.full,dtype=np.float32); gates=[]; probs=[]; maxang=0.; t0=time.time()
        for r0 in range(0,len(self.users),batch_users):
            r1=min(r0+batch_users,len(self.users)); uu=torch.as_tensor(self.users[r0:r1],device=model.device)
            ids=torch.as_tensor(self.items[r0:r1].reshape(-1),device=model.device); ur=uu.repeat_interleave(width)
            ctx=torch.as_tensor(self.ctx[r0:r1].reshape(-1,8),device=model.device) if variant=='D3' else None
            if variant=='D0':
                g=torch.ones(len(ids),device=model.device); p=torch.stack([torch.ones_like(g),torch.zeros_like(g),torch.zeros_like(g)],1)
            else:
                p,g=make_decision_features(variant,net,hist,ur,ids,False,ctx,history_mode)
            d,di=score_correction(model,c,basis,ur,ids,g,basis_mode,random_seed,True)
            delta[r0:r1]=d.cpu().numpy().reshape(r1-r0,width); gates.append(g.cpu()); probs.append(p.cpu())
            maxang=max(maxang,float(di['text']['angle_deg'].max()),float(di['visual']['angle_deg'].max()))
        final=self.full+delta; rank=rank_by_score(self.items,final); met=metrics_at(rank,self.users,self.eval_sets)
        pp=torch.cat(probs); gg=torch.cat(gates)
        return {'metrics':met,'rank':rank,'final_scores':final,'score_delta':delta,'action_stats':action_stats(pp,gg),'max_angle':maxang,
                'history_mode':history_mode,'basis_mode':basis_mode,'random_seed':random_seed,'inference_seconds':time.time()-t0}

def pair_terms(model,c,basis,hist,net,variant,users,pos,neg,ctxp=None,ctxn=None,prefix=True,return_outputs=False):
    b=len(users); ids=torch.cat([pos,neg]); ur=torch.cat([users,users]); ctx=None if ctxp is None else torch.cat([ctxp,ctxn])
    pt,gt=make_decision_features(variant,net,hist,ur,ids,prefix,ctx,'true')
    pw,gw=make_decision_features(variant,net,hist,ur,ids,prefix,ctx,'shuffled')
    dt,dit=score_correction(model,c,basis,ur,ids,gt,'true',None,True); dw=score_correction(model,c,basis,ur,ids,gw)
    D=dt[:b]-dt[b:]; W=dw[:b]-dw[b:]
    Lrank=F.softplus(-D).mean(); Lhist=F.softplus(W-D+HISTORY_MARGIN).mean(); loss=Lrank+LAMBDA_HISTORY*Lhist
    out={'L_rank':Lrank,'L_history':Lhist,'total_loss':loss,'Delta_margin_mean':D.mean(),'Delta_margin_median':D.median(),
         'fraction_Delta_margin_gt0':(D>0).float().mean(),'history_advantage_mean':(D-W).mean(),'fraction_true_gt_shuffled':(D>W).float().mean(),
         'angle_text_max':dit['text']['angle_deg'].max(),'angle_visual_max':dit['visual']['angle_deg'].max(),
         'pos_action':action_stats(pt[:b],gt[:b]),'neg_action':action_stats(pt[b:],gt[b:]),'all_action':action_stats(pt,gt)}
    if return_outputs: out.update({'D_vec':D,'W_vec':W,'delta_true':dt,'delta_wrong':dw,'probs':pt,'gates':gt})
    return loss,out

def _merge_action(stats_list):
    if not stats_list: return {}
    n=sum(x['n'] for x in stats_list); out={'n':n}
    for k in ('mean_p_up','mean_p_stay','mean_p_down','argmax_up_fraction','argmax_stay_fraction','argmax_down_fraction','mean_gate','mean_abs_gate'):
        out[k]=sum(x[k]*x['n'] for x in stats_list)/n
    out['gate_p10_mean_batch']=sum(x['gate_p10']*x['n'] for x in stats_list)/n
    out['gate_p50_mean_batch']=sum(x['gate_p50']*x['n'] for x in stats_list)/n
    out['gate_p90_mean_batch']=sum(x['gate_p90']*x['n'] for x in stats_list)/n
    out['ACTION_COLLAPSE_WARNING']=max(out['argmax_up_fraction'],out['argmax_stay_fraction'],out['argmax_down_fraction'])>.95
    return out

def train_epoch(seed,epoch,model,c,basis,hist,tc,net,opt,variant,max_batches=None):
    seed_all(seed*10000+epoch); net.train()
    keys=('L_rank','L_history','total_loss','Delta_margin_mean','history_advantage_mean','fraction_Delta_margin_gt0','fraction_true_gt_shuffled')
    sums={k:0. for k in keys}; n=0; posa=[]; nega=[]; alla=[]; maxang=0.; rmin=999;rmax=-1
    for users,pos,neg,cp,cn,ranks in tc.batches(seed,epoch,max_batches):
        opt.zero_grad(set_to_none=True)
        loss,p=pair_terms(model,c,basis,hist,net,variant,users,pos,neg,cp if variant=='D3' else None,cn if variant=='D3' else None,True)
        loss.backward(); opt.step(); n+=1
        for k in keys: sums[k]+=float(p[k].detach())
        posa.append(p['pos_action']); nega.append(p['neg_action']); alla.append(p['all_action'])
        maxang=max(maxang,float(p['angle_text_max']),float(p['angle_visual_max'])); rmin=min(rmin,int(ranks.min()));rmax=max(rmax,int(ranks.max()))
    out={k:v/max(n,1) for k,v in sums.items()}
    out.update({'batches':n,'hard_rank_min':rmin,'hard_rank_max':rmax,'max_angle':maxang,
                'positive_action':_merge_action(posa),'negative_action':_merge_action(nega),'all_action':_merge_action(alla)})
    return out

def save_net(path,net,seed,variant,epoch):
    torch.save({'protocol':PROTOCOL,'seed':seed,'variant':variant,'epoch':epoch,'input_dim':net.input_dim,
                'state':{k:v.detach().cpu() for k,v in net.state_dict().items()},'TEST_ACCESSED':False},path)
def restore_net(path,net):
    p=torch.load(path,map_location='cpu',weights_only=False); net.load_state_dict(p['state'],strict=True); return p

def hard_pair_arrays(ev,max_users=DIAG_USERS):
    rr=[];us=[];pi=[];ni=[];pc=[];nc=[]
    for r,u in enumerate(ev.users[:min(max_users,len(ev.users))]):
        loc={int(x):j for j,x in enumerate(ev.items[r])}; pos=[loc[int(x)] for x in ev.eval_sets[int(u)] if int(x) in loc]
        if not pos: continue
        pcol=pos[0]; pitem=int(ev.items[r,pcol])
        for x in ev.c0_rank[r,5:30]:
            x=int(x)
            if x in ev.eval_sets[int(u)]: continue
            rr.append(r);us.append(int(u));pi.append(pitem);ni.append(x);pc.append(pcol);nc.append(loc[x])
    return tuple(np.asarray(x,np.int64) for x in (rr,us,pi,ni,pc,nc))

def hard_shell_diag(model,c,basis,hist,ev,variant,net=None,batch=2048):
    rr,us,pi,ni,pc,nc=hard_pair_arrays(ev); D=[];W=[]; gates=[];probs=[]; maxang=0.
    with torch.no_grad():
        for s in range(0,len(us),batch):
            e=min(s+batch,len(us)); uu=torch.as_tensor(us[s:e],device=model.device); p=torch.as_tensor(pi[s:e],device=model.device); n=torch.as_tensor(ni[s:e],device=model.device)
            ids=torch.cat([p,n]); ur=torch.cat([uu,uu]); ctx=None
            if variant=='D3':
                ctxp=torch.as_tensor(ev.ctx[rr[s:e],pc[s:e]],device=model.device); ctxn=torch.as_tensor(ev.ctx[rr[s:e],nc[s:e]],device=model.device); ctx=torch.cat([ctxp,ctxn])
            if variant=='D0':
                gt=torch.ones(len(ids),device=model.device); gw=gt; pt=torch.stack([torch.ones_like(gt),torch.zeros_like(gt),torch.zeros_like(gt)],1)
            else:
                pt,gt=make_decision_features(variant,net,hist,ur,ids,False,ctx,'true'); _,gw=make_decision_features(variant,net,hist,ur,ids,False,ctx,'shuffled')
            dt,di=score_correction(model,c,basis,ur,ids,gt,'true',None,True); dw=score_correction(model,c,basis,ur,ids,gw)
            q=e-s; d=dt[:q]-dt[q:]; w=dw[:q]-dw[q:]
            D.append(d.cpu().numpy());W.append(w.cpu().numpy());gates.append(gt.cpu());probs.append(pt.cpu())
            maxang=max(maxang,float(di['text']['angle_deg'].max()),float(di['visual']['angle_deg'].max()))
    D=np.concatenate(D);W=np.concatenate(W);I=D-W
    return {'pair_count':int(len(D)),'Delta_margin':stat(D),'fraction_Delta_margin_gt0':float(np.mean(D>0)),
            'history_advantage':stat(I),'fraction_true_gt_shuffled':float(np.mean(I>0)),
            'action_stats':action_stats(torch.cat(probs),torch.cat(gates)),'max_angle':maxang,
            'definition':'first 1024 Validation users; frozen C0 rank 6-30; all non-positive candidates','TEST_ACCESSED':False}

def oracle_diag(model,c,basis,ev,batch=2048):
    rr,us,pi,ni,pc,nc=hard_pair_arrays(ev); actions=(1.,0.,-1.); bests=[]; best_idx=[]
    with torch.no_grad():
        for s in range(0,len(us),batch):
            e=min(s+batch,len(us)); uu=torch.as_tensor(us[s:e],device=model.device); p=torch.as_tensor(pi[s:e],device=model.device); n=torch.as_tensor(ni[s:e],device=model.device)
            ids=torch.cat([p,n]); ur=torch.cat([uu,uu]); corr=[]
            for a in actions:
                g=torch.full((len(ids),),float(a),device=model.device); corr.append(score_correction(model,c,basis,ur,ids,g).view(2,e-s))
            # corr[action][0,pos/1,neg]
            q=e-s; mats=[]
            for ia in range(3):
                for ib in range(3): mats.append(corr[ia][0]-corr[ib][1])
            M=torch.stack(mats,1); v,ix=M.max(1); bests.append(v.cpu().numpy());best_idx.append(ix.cpu().numpy())
    best=np.concatenate(bests); ix=np.concatenate(best_idx); labels=['(+,+)','(+,0)','(+,-)','(0,+)','(0,0)','(0,-)','(-,+)','(-,0)','(-,-)']
    dist={labels[i]:float(np.mean(ix==i)) for i in range(9)}
    return {'pair_count':int(len(best)),'fraction_oracle_improves_BASE':float(np.mean(best>0)),'best_Delta_margin':stat(best),
            'best_action_distribution':dist,'DIFFUSION_BASIS_CAPACITY_LIMITED':bool(np.mean(best>0)<.55 or float(best.mean())<=0),
            'oracle_used_for_inference':False,'oracle_used_for_selection':False,'TEST_ACCESSED':False}

def random_basis_control(seed,variant,root):
    d=json.loads((Path(root)/f'seed{seed}/{variant}/result.json').read_text())
    model,_,cfg,a,c,ab=load_backbone(seed); hist=HistoryStore(seed,c,model.device); basis=BasisStore(seed,model.device); ev=Evaluator(seed)
    net=DecisionNet(decision_dim(variant)).to(model.device); restore_net(Path(root)/f'seed{seed}/{variant}/best_checkpoint.pt',net); net.eval()
    true=ev.evaluate(model,c,basis,hist,variant,net,'true','true'); neg=ev.evaluate(model,c,basis,hist,variant,net,'true','negated'); vals=[]
    for rs in RANDOM_SEEDS:
        x=ev.evaluate(model,c,basis,hist,variant,net,'true','random',rs); vals.append({'seed':rs,'metrics':x['metrics'],'vs_C0':delta_pack(x['metrics'],ev.c0_metrics)})
    u=np.asarray([x['vs_C0']['U'] for x in vals],np.float64); tu=utility(true['metrics'],ev.c0_metrics); nu=utility(neg['metrics'],ev.c0_metrics)
    out={'seed':seed,'variant':variant,'U_TRUE_DIFFUSION':tu,'mean_U_RANDOM_BASIS':float(u.mean()),'std_U_RANDOM_BASIS':float(u.std()),
         'min_U_RANDOM_BASIS':float(u.min()),'max_U_RANDOM_BASIS':float(u.max()),'true_percentile_vs_random':float(100*np.mean(u<tu)),
         'U_NEGATED_BASIS':nu,'controls':vals,'PASS':bool(tu>u.mean() and tu>nu),'TEST_ACCESSED':False}
    del model,net; torch.cuda.empty_cache(); return out
