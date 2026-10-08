from __future__ import annotations
import argparse, gc, json, random, shutil, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src')); sys.path.insert(0, str(ROOT))
from utils.dataloader import TrainDataLoader
from pipelines.msca_assets import load_msca_checkpoint
from modules.ranking import metrics_at, rank_by_score
from pipelines.coliftrec import PRIMARY, ALL
from diffusion_experiments.round9_cabrp import run_round9 as r9
from diffusion_experiments.round11_cdtc import run_round11 as r11
from diffusion_experiments.round13_iudlp import latent_diffusion as r13ld
from diffusion_experiments.round14_aihu import run_round14 as r14run
from diffusion_experiments.round15_baurp import base_anchored_residual as br

PROTOCOL='ROUND15_BAURP_V1'
SOURCE='394b6acb2a9597124ce2ceb863bf47b3890cfc02'
PREFLIGHT=(999,1000); EXPANSION=(1001,1002); VARIANTS=('A0','A1','A2')
EPOCHS=5; LR_SCALE=.1; PAIR_BATCH_USERS=256; DIAG_USERS=1024

def seed_all(s):
    random.seed(int(s)); np.random.seed(int(s)); torch.manual_seed(int(s)); torch.cuda.manual_seed_all(int(s))

def stat(x):
    x=np.asarray(x,np.float64)
    return {'mean':float(x.mean()),'median':float(np.median(x)),'fraction_positive':float(np.mean(x>0)),
            'p10':float(np.quantile(x,.1)),'p90':float(np.quantile(x,.9))}

def stat_angle(x):
    x=np.asarray(x,np.float64)
    return {'mean':float(x.mean()),'median':float(np.median(x)),'p90':float(np.quantile(x,.9)),
            'p95':float(np.quantile(x,.95)),'p99':float(np.quantile(x,.99)),'max':float(x.max())}

def utility(m,b): return float(np.mean([(m[k]-b[k])/b[k] for k in PRIMARY]))
def delta_pack(m,b):
    return {'absolute_delta':{k:float(m[k]-b[k]) for k in ALL},
            'relative_delta':{k:float((m[k]-b[k])/b[k]) for k in ALL},
            'U':utility(m,b),'primary_positive_count':int(sum(m[k]>b[k] for k in PRIMARY)),
            'overall_positive_count':int(sum(m[k]>b[k] for k in ALL))}

def checkpoint_audit(seed):
    a=json.loads((r9.paths(seed)['msca']/'audit.json').read_text())
    if a.get('TEST_ACCESSED') is not False: raise RuntimeError('contaminated checkpoint')
    return a

def sync(device):
    if str(device).startswith('cuda'): torch.cuda.synchronize(device)

def hard_lookup_tensor(seed,n_users,device):
    z,a=r14run.ensure_hard_pool(seed)
    if not a['TRAIN_ONLY'] or a['validation_positive_used'] or a['validation_metric_used'] or a['test_used']:
        raise RuntimeError('hard candidate provenance invalid')
    if a['fallback_ratio'] != 0 or a['observed_positive_violations'] != 0:
        raise RuntimeError('hard candidate integrity invalid')
    lookup=torch.full((n_users,z['hard_items'].shape[1]),-1,device=device,dtype=torch.long)
    users=torch.as_tensor(z['users'],device=device,dtype=torch.long)
    lookup[users]=torch.as_tensor(z['hard_items'],device=device,dtype=torch.long)
    return lookup,a

def load_training(seed,variant):
    a=checkpoint_audit(seed)
    model,ck,_,train_dataset=load_msca_checkpoint(Path(a['checkpoint']),0)
    br.freeze_recommender(model)
    cfg=ck['config']; train_data=TrainDataLoader(cfg,train_dataset,batch_size=cfg['train_batch_size'],shuffle=True)
    seed_all(202614700+seed); train_data.pretrain_setup()
    seed_all(202614000+seed); modules=br.BAURPModules().to(model.device)
    opt=torch.optim.Adam(list(modules.parameters()),lr=float(cfg['learning_rate'])*LR_SCALE,
                         weight_decay=float(cfg['weight_decay'] or 0.0))
    with torch.no_grad(): c={k:v.detach() for k,v in br.forward_components(model).items()}
    ab=br.cosine_alpha_bar(device=model.device); hard,ha=hard_lookup_tensor(seed,model.n_users,model.device)
    return model,modules,opt,ab,train_data,ck,cfg,a,c,hard,ha

def save_modules(path,modules,epoch,metrics,mechanism,variant,seed):
    torch.save({'protocol':PROTOCOL,'seed':seed,'variant':variant,'epoch':epoch,'metrics':metrics,
                'mechanism':mechanism,'modules_state':{k:v.detach().cpu() for k,v in modules.state_dict().items()},
                'TEST_ACCESSED':False},path)

def restore(path,modules):
    p=torch.load(path,map_location='cpu',weights_only=False); modules.load_state_dict(p['modules_state'],strict=True); return p

def variant_pair_scores(model,modules,c,users,items,variant,ab,noise,condition_mode='true',return_diag=False):
    if variant=='A0':
        base=(c['final_user'][users]*c['final_item'][items]).sum(1)
        if condition_mode=='zero':
            z=torch.zeros((len(items),br.LATENT_DIM),device=items.device)
            if return_diag:
                dz=torch.zeros(len(items),device=items.device)
                d={'text':{'angle_deg':dz,'cap_saturated':dz.bool(),'raw_tangent_norm':dz,'bounded_tangent_norm':dz},
                   'visual':{'angle_deg':dz,'cap_saturated':dz.bool(),'raw_tangent_norm':dz,'bounded_tangent_norm':dz}}
                return base,base,z,z,d
            return base,base,z,z
        cond=users if condition_mode=='true' else br.shuffled_user_ids(users,model.n_users)
        if condition_mode not in ('true','shuffled'): raise ValueError(condition_mode)
        b,a,rt,rv=r13ld.pair_scores(model,modules,c,users,items,'D2',ab,noise,rho=br.RHO,condition_user_ids=cond)
        if return_diag:
            _,dt=br.tangent_augment(c['text_item'][items],rt,False,return_diag=True)
            _,dv=br.tangent_augment(c['image_item'][items],rv,False,return_diag=True)
            return b,a,rt,rv,{'text':dt,'visual':dv}
        return b,a,rt,rv
    return br.pair_scores(model,modules,c,users,items,ab,noise,bounded=(variant=='A2'),
                          condition_mode=condition_mode,return_diag=return_diag)

class ValidationEvaluator(r14run.ValidationEvaluator):
    @torch.no_grad()
    def evaluate_variant(self,model,modules,c,variant,condition_mode='true',batch_users=PAIR_BATCH_USERS):
        model.eval(); modules.eval(); sync(model.device); t0=time.time()
        ab=br.cosine_alpha_bar(device=model.device); noise=br.build_noise_tables(model.n_items,model.device)
        width=self.items.shape[1]; delta=np.empty_like(self.msca,dtype=np.float32); rn=0.0; rc=0
        for r0 in range(0,len(self.users),batch_users):
            r1=min(r0+batch_users,len(self.users)); u=torch.as_tensor(self.users[r0:r1],device=model.device)
            ids=torch.as_tensor(self.items[r0:r1].reshape(-1),device=model.device); ur=u.repeat_interleave(width)
            base,aug,rt,rv=variant_pair_scores(model,modules,c,ur,ids,variant,ab,noise,condition_mode)
            delta[r0:r1]=(aug-base).float().cpu().numpy().reshape(r1-r0,width)
            rn+=float(rt.norm(dim=1).sum()+rv.norm(dim=1).sum()); rc+=2*len(ids)
        final=self.full+delta; rank=rank_by_score(self.items,final); met=metrics_at(rank,self.users,self.eval_sets); sync(model.device)
        return {'metrics':met,'rank':rank,'candidate_items':self.items,'final_scores':final,'score_delta':delta,
                'mean_residual_norm':rn/max(rc,1),'inference_seconds':time.time()-t0,'condition_mode':condition_mode}

def diag_pairs(ev,n,hard_shell):
    n=min(n,len(ev.users)); users=ev.users[:n]
    pos=np.asarray([min(ev.eval_sets[int(u)]) for u in users],np.int64); neg=[]
    pool=ev.c0_rank if hard_shell else ev.items
    for r,u in enumerate(users):
        ps=ev.eval_sets[int(u)]; cand=pool[r,5:30] if hard_shell else pool[r]
        neg.append(next(int(x) for x in cand if int(x) not in ps))
    return users,pos,np.asarray(neg,np.int64)

def identity_diag(model,modules,c,ev,variant,hard_shell=False,n=DIAG_USERS):
    users,pos,neg=diag_pairs(ev,n,hard_shell); n=len(users)
    ut=torch.as_tensor(users,device=model.device); ids=torch.as_tensor(np.concatenate([pos,neg]),device=model.device)
    su=torch.cat([ut,ut]); ab=br.cosine_alpha_bar(device=model.device); noise=br.build_noise_tables(model.n_items,model.device)
    with torch.no_grad():
        _,st,rt,rv=variant_pair_scores(model,modules,c,su,ids,variant,ab,noise,'true')
        _,ss,rst,rsv=variant_pair_scores(model,modules,c,su,ids,variant,ab,noise,'shuffled')
    mt=(st[:n]-st[n:]).cpu().numpy(); ms=(ss[:n]-ss[n:]).cpu().numpy()
    out={'sample_n':n,'hard_shell_rank_6_30':bool(hard_shell),'same_item_noise_timestep':True,'t_infer':br.T_INFER}
    for name,a,b in [('text',rt,rst),('visual',rv,rsv)]:
        out[name]={'true_residual_norm':stat(a.norm(dim=1).cpu().numpy()),
                   'shuffled_residual_norm':stat(b.norm(dim=1).cpu().numpy()),
                   'true_minus_shuffled_norm':stat((a-b).norm(dim=1).cpu().numpy()),
                   'cos_true_vs_shuffled':stat(F.cosine_similarity(a,b,dim=1).cpu().numpy())}
    out['margin_true_minus_shuffled']=stat(mt-ms); out['TEST_ACCESSED']=False
    return out

def angle_diag(model,modules,c,ev,variant,n=1000):
    width=ev.items.shape[1]; take=min(n,len(ev.users)*width)
    flat_items=ev.items.reshape(-1)[:take]; row=np.arange(take)//width; us=ev.users[row]
    u=torch.as_tensor(us,device=model.device); ids=torch.as_tensor(flat_items,device=model.device)
    ab=br.cosine_alpha_bar(device=model.device); noise=br.build_noise_tables(model.n_items,model.device)
    with torch.no_grad():
        *_,d=variant_pair_scores(model,modules,c,u,ids,variant,ab,noise,'true',True)
    out={'sample_pairs':take,'theta_max_deg':br.THETA_MAX_DEG,'tangent_norm_cap':br.TANGENT_NORM_CAP}
    for m in ('text','visual'):
        ang=d[m]['angle_deg'].cpu().numpy(); out[m]=stat_angle(ang)
        out[m]['cap_saturation_fraction']=float(d[m]['cap_saturated'].float().mean().cpu())
        out[m]['raw_tangent_norm_mean']=float(d[m]['raw_tangent_norm'].mean().cpu())
        out[m]['bounded_tangent_norm_mean']=float(d[m]['bounded_tangent_norm'].mean().cpu())
    out['TEST_ACCESSED']=False; return out

def train_epoch(model,modules,opt,ab,train_data,c,variant,epoch,seed,hard,max_batches=None):
    seed_all(seed*10000+epoch); model.eval(); modules.train(); torch.cuda.reset_peak_memory_stats(model.device)
    keys=('L_rec','L_user','L_base','L_base_text','L_base_visual','L_bound','total_loss','m_true','m_shuf',
          'm_true_minus_shuf','fraction_true_gt_shuf','raw_residual_text_true','raw_residual_text_shuf',
          'raw_residual_visual_true','raw_residual_visual_shuf','bounded_tangent_text','bounded_tangent_visual',
          'angle_text_mean','angle_text_p95','angle_text_max','angle_visual_mean','angle_visual_p95','angle_visual_max',
          'cap_fraction_text','cap_fraction_visual','sampled_hard_rank_mean')
    sums={k:0.0 for k in keys}; n=0; rmin=99;rmax=-1;pmin=99;pmax=-1; t0=time.time(); params=list(modules.parameters()); gn=0.0
    for bi,interaction in enumerate(train_data):
        if max_batches is not None and bi>=max_batches: break
        opt.zero_grad(set_to_none=True)
        loss,parts=(br.training_loss_a0(model,modules,interaction,c,ab,hard) if variant=='A0'
                    else br.training_loss(model,modules,interaction,c,variant,ab,hard))
        if not torch.isfinite(loss): raise RuntimeError('nonfinite loss')
        loss.backward(); gn=br.grad_l2(params)
        if any(p.grad is not None for p in model.parameters()): raise RuntimeError('frozen recommender gradient')
        clip=model.config['clip_grad_norm'] if hasattr(model,'config') else None
        if clip: torch.nn.utils.clip_grad_norm_(params,**clip)
        opt.step(); n+=1
        for k in keys: sums[k]+=float(parts[k].detach())
        rmin=min(rmin,int(parts['reconstruction_t_min']));rmax=max(rmax,int(parts['reconstruction_t_max']))
        pmin=min(pmin,int(parts['preference_t_min']));pmax=max(pmax,int(parts['preference_t_max']))
    sec=time.time()-t0; out={k:v/max(n,1) for k,v in sums.items()}
    out.update({'gradient_norm':gn,'batches':n,'seconds':sec,
                'peak_gpu_memory_GiB':float(torch.cuda.max_memory_allocated(model.device)/(1024**3)),
                'reconstruction_t_range':[rmin,rmax],'preference_t_range':[pmin,pmax],'NaN_count':0,'OOM':False})
    return out

def run_variant(seed,variant,root):
    r14run.baseline(seed,root)
    out=root/f'seed{seed}'/variant; shutil.rmtree(out,ignore_errors=True); out.mkdir(parents=True,exist_ok=True)
    model,mods,opt,ab,train_data,ck,cfg,a,c,hard,ha=load_training(seed,variant); ev=ValidationEvaluator(seed)
    logs=[]; t0=time.time()
    for ep in range(1,EPOCHS+1):
        tr=train_epoch(model,mods,opt,ab,train_data,c,variant,ep,seed,hard)
        val=ev.evaluate_variant(model,mods,c,variant,'true')
        mech=identity_diag(model,mods,c,ev,variant,False)
        hardmech=identity_diag(model,mods,c,ev,variant,True)
        bound=angle_diag(model,mods,c,ev,variant)
        bd=r11.boundary_diag(ev.c0_rank,val['rank'],ev.users,ev.eval_sets)
        sane=bool(mech['margin_true_minus_shuffled']['mean']>=0)
        rec={'epoch':ep,'train':tr,'validation_metrics':val['metrics'],'U_vs_C0':utility(val['metrics'],ev.c0_metrics),
             'selection_mechanism':mech,'hard_shell_mechanism':hardmech,'bound':bound,'boundary':bd,'mechanism_sane':sane}
        logs.append(rec); save_modules(out/f'checkpoint_ep{ep}.pt',mods,ep,val['metrics'],mech,variant,seed)
        print(json.dumps({'seed':seed,'variant':variant,'epoch':ep,'U':rec['U_vs_C0'],
                          'R20':val['metrics']['R20'],'mech':mech['margin_true_minus_shuffled']['mean'],
                          'hard_mech':hardmech['margin_true_minus_shuffled']['mean'],
                          'frac':hardmech['margin_true_minus_shuffled']['fraction_positive'],
                          'resT':tr['raw_residual_text_true'],'resV':tr['raw_residual_visual_true'],
                          'angleT':tr['angle_text_mean'],'angleV':tr['angle_visual_mean'],
                          'capT':tr['cap_fraction_text'],'capV':tr['cap_fraction_visual']},sort_keys=True),flush=True)
    sane=[x for x in logs if x['mechanism_sane']]; mechanism_failure=not sane
    candidates=sane if sane else logs; chosen=max(candidates,key=lambda x:x['validation_metrics']['R20'])
    best_epoch=chosen['epoch']; shutil.copy2(out/f'checkpoint_ep{best_epoch}.pt',out/'best_checkpoint.pt'); restore(out/'best_checkpoint.pt',mods)
    vals={mode:ev.evaluate_variant(model,mods,c,variant,mode) for mode in ('true','shuffled','zero')}
    sel=identity_diag(model,mods,c,ev,variant,False); hardmech=identity_diag(model,mods,c,ev,variant,True)
    bound=angle_diag(model,mods,c,ev,variant); bd=r11.boundary_diag(ev.c0_rank,vals['true']['rank'],ev.users,ev.eval_sets)
    result={'status':'PASS','protocol':PROTOCOL,'seed':seed,'variant':variant,'starting_checkpoint':a['checkpoint'],
            'best_epoch':best_epoch,'mechanism_failure_all_epochs':mechanism_failure,'pass_eligible':not mechanism_failure,
            'C0_metrics':ev.c0_metrics,'best_metrics':vals['true']['metrics'],'best_vs_C0':delta_pack(vals['true']['metrics'],ev.c0_metrics),
            'identity_ablation':{mode:{'metrics':vals[mode]['metrics'],'vs_C0':delta_pack(vals[mode]['metrics'],ev.c0_metrics)} for mode in vals},
            'selection_mechanism':sel,'hard_shell_mechanism':hardmech,'bound_diagnostic':bound,'boundary':bd,
            'epoch_logs':logs,'hard_candidate_audit':ha,'elapsed_seconds':time.time()-t0,
            'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
    np.savez_compressed(out/'best_validation_rankings.npz',users=ev.users,ranked_items=vals['true']['rank'],
                        candidate_items=ev.items,final_scores=vals['true']['final_scores'])
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':'PASS','seed':seed,'variant':variant,'best_epoch':best_epoch,
                      'U_true':result['identity_ablation']['true']['vs_C0']['U'],
                      'U_shuf':result['identity_ablation']['shuffled']['vs_C0']['U'],
                      'U_zero':result['identity_ablation']['zero']['vs_C0']['U'],
                      'hard_mech':hardmech['margin_true_minus_shuffled']},sort_keys=True),flush=True)
    del model,mods,opt; torch.cuda.empty_cache(); gc.collect(); return result

def smoke(root,evid):
    evid.mkdir(parents=True,exist_ok=True); seed=999
    model,mods,opt,ab,train_data,ck,cfg,a,c,hard,ha=load_training(seed,'A2'); ev=ValidationEvaluator(seed)
    rows=br.parameter_audit(model,mods); rec_train=[x for x in rows if x['group']=='recommender' and x['trainable']]
    (evid/'ROUND15_PARAMETER_AUDIT.json').write_text(json.dumps({'rows':rows,'recommender_trainable_count':len(rec_train),'TEST_ACCESSED':False},indent=2)+'\n')
    if rec_train: raise RuntimeError('recommender trainable')

    old=json.loads((ROOT/'diffusion_experiments/round14_aihu/outputs/seed999/A3/result.json').read_text())
    oldcp=ROOT/'diffusion_experiments/round14_aihu/outputs/seed999/A3/best_checkpoint.pt'
    oldmods=br.BAURPModules().to(model.device); restore(oldcp,oldmods)
    p=ev.evaluate_variant(model,oldmods,c,'A0','true')
    parity={'metric_max_abs_diff':max(abs(p['metrics'][k]-old['best_metrics'][k]) for k in ALL),
            'metrics_new':p['metrics'],'metrics_round14':old['best_metrics'],'hard_candidate_same_source':True,
            'PASS':all(abs(p['metrics'][k]-old['best_metrics'][k])<=1e-12 for k in ALL),'TEST_ACCESSED':False}
    (evid/'ROUND15_SCORE_PARITY_AUDIT.json').write_text(json.dumps(parity,indent=2)+'\n')
    if not parity['PASS']: raise RuntimeError('A0 parity fail')

    interaction=next(iter(train_data)); users,pos,orig_neg=interaction[0],interaction[1],interaction[2]
    ids=torch.cat([pos[:128],orig_neg[:128]]); uids=torch.cat([users[:128],users[:128]])
    t=torch.arange(1,len(ids)+1,device=model.device)%br.T_LATENT+1
    base,baseparts=br.base_terms(model,mods,c,uids,ids,t,ab)
    opt.zero_grad(); base.backward(); base_grad=br.grad_l2(list(mods.parameters()))
    base_audit={'true_reconstruction_anchor':False,'zero_base_reconstruction_anchor':True,'base_grad_norm':base_grad,
                'base_grad_finite':bool(np.isfinite(base_grad)),'PASS':bool(np.isfinite(base_grad) and base_grad>0),'TEST_ACCESSED':False}
    (evid/'ROUND15_BASE_ANCHOR_AUDIT.json').write_text(json.dumps(base_audit,indent=2)+'\n')
    if not base_audit['PASS']: raise RuntimeError('base anchor fail')

    neg,_=br.sample_hard_negatives(users[:128],hard); tp=torch.full((128,),br.T_INFER,device=model.device,dtype=torch.long)
    pref=br.preference_terms(model,mods,c,users[:128],pos[:128],neg,tp,ab,True,return_outputs=True)
    params=list(mods.parameters()); grads={}
    for name in ('L_rec','L_user'):
        opt.zero_grad(set_to_none=True); pref[name].backward(retain_graph=True); grads[name]=br.grad_l2(params)
    gref=torch.autograd.grad(pref['L_rec']+pref['L_user'],[pref['p0t'],pref['p0v']],allow_unused=True,retain_graph=True)
    grad_audit={'L_rec_grad_norm':grads['L_rec'],'L_user_grad_norm':grads['L_user'],
                'p0_reference_autograd_none':[x is None for x in gref],'p0_reference_detached':all(x is None for x in gref),
                'PASS':all(np.isfinite(list(grads.values()))) and min(grads.values())>0 and all(x is None for x in gref),
                'TEST_ACCESSED':False}
    (evid/'ROUND15_GRADIENT_AUDIT.json').write_text(json.dumps(grad_audit,indent=2)+'\n')
    if not grad_audit['PASS']: raise RuntimeError('preference gradient fail')

    bdiag=angle_diag(model,mods,c,ev,'A2',1000); bpass=max(bdiag['text']['max'],bdiag['visual']['max'])<=5.01
    bdiag['PASS']=bpass; (evid/'ROUND15_BOUND_AUDIT.json').write_text(json.dumps(bdiag,indent=2)+'\n')
    if not bpass: raise RuntimeError('bound fail')
    hard_audit={'source_round14':ha,'PASS':ha['TRAIN_ONLY'] and not ha['validation_positive_used'] and not ha['validation_metric_used']
                and not ha['test_used'] and ha['fallback_ratio']==0 and ha['observed_positive_violations']==0,'TEST_ACCESSED':False}
    (evid/'ROUND15_HARD_CANDIDATE_AUDIT.json').write_text(json.dumps(hard_audit,indent=2)+'\n')
    if not hard_audit['PASS']: raise RuntimeError('hard shell fail')
    uid=identity_diag(model,mods,c,ev,'A2',True,1024)
    same_ok=uid['text']['true_minus_shuffled_norm']['mean']>0 and uid['visual']['true_minus_shuffled_norm']['mean']>0
    (evid/'ROUND15_USER_IDENTITY_AUDIT.json').write_text(json.dumps({'diagnostic':uid,'PASS':same_ok,'TEST_ACCESSED':False},indent=2)+'\n')
    if not same_ok: raise RuntimeError('identity residual identical')
    out={'status':'PASS','A0_parity':parity,'base_anchor':base_audit,'gradient':grad_audit,'bound':bdiag,
         'hard':hard_audit,'identity':uid,'TEST_ACCESSED':False}
    (evid/'ROUND15_SMOKE.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'status':'PASS','checks':{'A0_parity':parity['PASS'],'base':base_audit['PASS'],
          'gradient':grad_audit['PASS'],'bound':bpass,'hard':hard_audit['PASS'],'identity':same_ok}},sort_keys=True))
    del model,mods,opt,oldmods; torch.cuda.empty_cache(); return out

def load_result(root,seed,v):
    r=json.loads((root/f'seed{seed}'/v/'result.json').read_text())
    if r['status']!='PASS' or r.get('TEST_ACCESSED') is not False: raise RuntimeError('invalid result')
    return r

def gate_preflight(root,evid):
    rows={s:{v:load_result(root,s,v) for v in VARIANTS} for s in PREFLIGHT}
    a2=[rows[s]['A2'] for s in PREFLIGHT]
    mech=[x['hard_shell_mechanism']['margin_true_minus_shuffled'] for x in a2]
    U=[x['identity_ablation']['true']['vs_C0']['U'] for x in a2]
    Us=[x['identity_ablation']['shuffled']['vs_C0']['U'] for x in a2]
    bound=[x['bound_diagnostic'] for x in a2]
    primary_floor=[min(x['best_vs_C0']['relative_delta'][k] for k in PRIMARY) for x in a2]
    gateM=all(m['mean']>0 and m['fraction_positive']>=.55 for m in mech)
    gateI=all(t>s for t,s in zip(U,Us))
    gateU=all(x>.001 for x in U) and float(np.mean(U))>=.002 and all(x>=-.005 for x in primary_floor)
    gateB=all(max(x['text']['max'],x['visual']['max'])<=5.01 for x in bound)
    warnings=[s for s,x in zip(PREFLIGHT,bound) if max(x['text']['cap_saturation_fraction'],x['visual']['cap_saturation_fraction'])>.80]
    gates={'Gate_M':gateM,'Gate_I':gateI,'Gate_U':gateU,'Gate_B':gateB}; open_gate=all(gates.values())
    out={'status':'PREFLIGHT_DECISION','rows':{str(s):{v:rows[s][v] for v in VARIANTS} for s in PREFLIGHT},
         'gates':gates,'BOUND_SATURATION_WARNING_seeds':warnings,'EXPANSION_OPEN':open_gate,
         'seed1001_not_opened':not open_gate,'seed1002_not_opened':not open_gate,
         'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
    evid.mkdir(parents=True,exist_ok=True); (evid/'ROUND15_PREFLIGHT_RESULTS.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'EXPANSION_OPEN':open_gate,'gates':gates,'U_true':dict(zip(PREFLIGHT,U)),
                      'U_shuffled':dict(zip(PREFLIGHT,Us)),'bound_warning':warnings},sort_keys=True))
    return out

def final(root,evid):
    pf=json.loads((evid/'ROUND15_PREFLIGHT_RESULTS.json').read_text()); expanded=pf['EXPANSION_OPEN']
    seeds=list(PREFLIGHT)+(list(EXPANSION) if expanded else [])
    rows={s:load_result(root,s,'A2') for s in seeds}
    U=np.asarray([rows[s]['identity_ablation']['true']['vs_C0']['U'] for s in seeds],np.float64)
    mech=[rows[s]['hard_shell_mechanism']['margin_true_minus_shuffled'] for s in seeds]
    ident=[rows[s]['identity_ablation']['true']['vs_C0']['U']>rows[s]['identity_ablation']['shuffled']['vs_C0']['U'] for s in seeds]
    if not expanded:
        if not pf['gates']['Gate_B']: verdict='ROUND15_BOUND_FAIL'
        elif not pf['gates']['Gate_I']: verdict='ROUND15_IDENTITY_FAIL'
        elif pf['gates']['Gate_M'] and not pf['gates']['Gate_U']: verdict='ROUND15_MECHANISM_PASS_UTILITY_FAIL'
        elif not pf['gates']['Gate_U']: verdict='ROUND15_MAGNITUDE_FAIL'
        else: verdict='ROUND15_FAIL'
    else:
        mpass=sum(m['mean']>0 and m['fraction_positive']>=.55 for m in mech); ipass=sum(ident); positive=int(np.sum(U>0))
        verdict='ROUND15_STRONG_PASS' if positive>=3 and U.mean()>=.0015 and mpass>=3 and ipass>=3 else 'ROUND15_FAIL'
    summary={'status':'VALIDATION_DECISION','expanded':expanded,'seeds':seeds,
             'A2_U_true':{str(s):rows[s]['identity_ablation']['true']['vs_C0']['U'] for s in seeds},
             'mean_U':float(U.mean()),'std_U':float(U.std()),'range_U':[float(U.min()),float(U.max())],
             'positive_seeds':int(np.sum(U>0)),'mechanism_pass_seeds':int(sum(m['mean']>0 and m['fraction_positive']>=.55 for m in mech)),
             'identity_pass_seeds':int(sum(ident)),'rows':{str(s):rows[s] for s in seeds},'verdict':verdict,
             'seed1001_not_opened':not expanded,'seed1002_not_opened':not expanded,
             'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
    (evid/'ROUND15_VALIDATION_RESULTS.json').write_text(json.dumps(summary,indent=2)+'\n'); write_report(summary,pf,evid)
    print(json.dumps({'verdict':verdict,'expanded':expanded,'mean_U':summary['mean_U']},sort_keys=True)); return summary

def write_report(s,pf,evid):
    L=['# Round15 Validation Report — Base-Anchored Bounded User Residual Purification','',
       f"Final verdict: **{s['verdict']}**",f"Expansion executed: **{s['expanded']}**",'',
       '## Preflight Gates','```text',json.dumps(pf['gates'],indent=2),'```','',
       '## Results','| Seed | Variant | U_true | U_shuffled | U_zero | hard m_true-m_shuf | frac | R10 | N10 | R20 | N20 |',
       '|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for ss in map(str,s['seeds']):
        source=pf['rows'][ss] if ss in pf['rows'] else None; variants=VARIANTS if source else ('A2',)
        for v in variants:
            x=source[v] if source else s['rows'][ss]
            ia=x['identity_ablation']; hm=x['hard_shell_mechanism']['margin_true_minus_shuffled']; m=x['best_metrics']
            L.append(f"| {ss} | {v} | {100*ia['true']['vs_C0']['U']:+.4f}% | {100*ia['shuffled']['vs_C0']['U']:+.4f}% | {100*ia['zero']['vs_C0']['U']:+.4f}% | {hm['mean']:+.6f} | {hm['fraction_positive']:.4f} | {m['R10']:.8f} | {m['N10']:.8f} | {m['R20']:.8f} | {m['N20']:.8f} |")
    L+=['','## Summary',f"- mean U: {100*s['mean_U']:+.4f}%",f"- std U: {100*s['std_U']:.4f}%",
        f"- positive seeds: {s['positive_seeds']}",f"- mechanism-pass seeds: {s['mechanism_pass_seeds']}",
        f"- identity-pass seeds: {s['identity_pass_seeds']}",'','Test, Sports and Electronics remained CLOSED.']
    (evid/'ROUND15_REPORT.md').write_text('\n'.join(L)+'\n')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--mode',choices=['smoke','formal','preflight','final'],required=True)
    ap.add_argument('--seed',type=int); ap.add_argument('--variant',choices=VARIANTS)
    ap.add_argument('--root',default=str(ROOT/'diffusion_experiments/round15_baurp/outputs'))
    ap.add_argument('--evidence',default=str(ROOT/'diffusion_experiments/round15_baurp/evidence'))
    a=ap.parse_args(); root=Path(a.root); evid=Path(a.evidence)
    if a.mode=='smoke': smoke(root,evid)
    elif a.mode=='formal': run_variant(a.seed,a.variant,root)
    elif a.mode=='preflight': gate_preflight(root,evid)
    else: final(root,evid)
if __name__=='__main__': main()
