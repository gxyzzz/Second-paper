from __future__ import annotations
import argparse,gc,json,shutil
from pathlib import Path
import numpy as np
import torch
from diffusion_experiments.round17_decision_guided.round17_core import *

def file_sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def grad_norm(m):
    return float(sum(float(p.grad.detach().float().pow(2).sum()) for p in m.parameters() if p.grad is not None)**.5)

def preflight(root,evid):
    root=Path(root); evid=Path(evid); root.mkdir(parents=True,exist_ok=True); evid.mkdir(parents=True,exist_ok=True)
    param={'protocol':PROTOCOL,'seeds':{},'TEST_ACCESSED':False}; hist_a={'protocol':PROTOCOL,'seeds':{},'TEST_ACCESSED':False}; basis_a={'protocol':PROTOCOL,'seeds':{},'TEST_ACCESSED':False}; smoke={'protocol':PROTOCOL,'seeds':{},'TEST_ACCESSED':False}
    for seed in SEEDS:
        reuse=prepare_base(seed,root); ba=build_basis(seed,root,True); basis_a['seeds'][str(seed)]=ba
        model,_,cfg,a,c,ab=load_backbone(seed); base,bmeta=load_base(seed,root,model.device); hs=HistoryStore(seed,c,model.device); bs=BasisStore(seed,model.device); tc=CleanTrain(seed,model.device); ev=Evaluator(seed)
        mh0=state_sha256(model); bh0=state_sha256(base); clean=R16R/f'assets/seed{seed}_train_clean.npz'; valctx=R16/f'assets/seed{seed}_validation.npz'
        pnet=DecisionNet(258).to(model.device)
        param['seeds'][str(seed)]={'backbone_trainable_params':sum(p.numel() for p in model.parameters() if p.requires_grad),'D_base_trainable_params':sum(p.numel() for p in base.parameters() if p.requires_grad),'CoLift_trainable_params':0,'DecisionNet_trainable_params':sum(p.numel() for p in pnet.parameters() if p.requires_grad),'backbone_hash':mh0,'D_base_hash':bh0,'clean_asset_sha256':file_sha256(clean),'validation_context_sha256':file_sha256(valctx),'optimizer_lr':float(cfg['learning_rate'])*LR_SCALE,'optimizer_weight_decay':float(cfg['weight_decay'] or 0.0)}
        # history audit across every retained clean pseudo user
        mismatch=0; incl=0
        for u,t in zip(tc.users,tc.target):
            u=int(u);t=int(t); mismatch+=int(t!=int(hs.histories[u][-1])); incl+=int(t in set(hs.pseudo[u]))
        uu,pp,nn,cp,cn,ranks=tc.first_batch(); x,meta=hs.target_features(uu,pp,True)
        hist_a['seeds'][str(seed)]={'clean_users':int(len(tc.users)),'target_is_last_event_mismatch':mismatch,'TRAIN_target_self_inclusion':incl,'TRAIN_history_source':'canonical prefix h[:-1]','Validation_history_source':'complete TRAIN history','Validation_label_used_in_history':False,'Test_used':False,'target_aware_shape':list(x.shape),'target_aware_dim':int(x.shape[1]),'history_len_min':int(meta['history_len'].min()),'history_len_max':int(meta['history_len'].max()),'PASS':mismatch==0 and incl==0 and x.shape[1]==258}
        sr={'base_reuse':reuse,'variants':{}}
        for v in ('D1','D2','D3'):
            net=DecisionNet(decision_dim(v)).to(model.device); ctxp=cp if v=='D3' else None;ctxn=cn if v=='D3' else None
            with torch.no_grad():
                ids=torch.cat([pp,nn]); ur=torch.cat([uu,uu]); ctx=None if ctxp is None else torch.cat([ctxp,ctxn]); pr,g0=make_decision_features(v,net,hs,ur,ids,True,ctx,'true')
            initial=float(g0.abs().max()); net.zero_grad(set_to_none=True)
            loss,parts=pair_terms(model,c,bs,hs,net,v,uu,pp,nn,ctxp,ctxn,True); loss.backward(); ng=grad_norm(net)
            frozen=all(p.grad is None for p in model.parameters()) and all(p.grad is None for p in base.parameters()); ma=max(float(parts['angle_text_max']),float(parts['angle_visual_max']))
            sr['variants'][v]={'input_dim':decision_dim(v),'loss':float(loss.detach()),'decision_grad_norm':ng,'frozen_grad_none':frozen,'initial_gate_max_abs':initial,'gate_min':float(g0.min()),'gate_max':float(g0.max()),'max_angle':ma,'loss_finite':bool(torch.isfinite(loss)),'PASS':bool(ng>0 and frozen and initial<=1e-8 and ma<=5.01 and torch.isfinite(loss))}
        sr['backbone_hash_unchanged']=state_sha256(model)==mh0;sr['D_base_hash_unchanged']=state_sha256(base)==bh0;sr['PASS']=all(x['PASS'] for x in sr['variants'].values()) and sr['backbone_hash_unchanged'] and sr['D_base_hash_unchanged']; smoke['seeds'][str(seed)]=sr
        del model,base,pnet;torch.cuda.empty_cache();gc.collect()
    param['PASS']=all(x['backbone_trainable_params']==0 and x['D_base_trainable_params']==0 and x['CoLift_trainable_params']==0 and x['DecisionNet_trainable_params']>0 for x in param['seeds'].values())
    hist_a['PASS']=all(x['PASS'] for x in hist_a['seeds'].values()); basis_a['PASS']=all(x['status']=='PASS' for x in basis_a['seeds'].values()); smoke['PASS']=all(x['PASS'] for x in smoke['seeds'].values()) and param['PASS'] and hist_a['PASS'] and basis_a['PASS']
    (evid/'ROUND17_PARAMETER_AUDIT.json').write_text(json.dumps(param,indent=2)+'\n');(evid/'ROUND17_HISTORY_AUDIT.json').write_text(json.dumps(hist_a,indent=2)+'\n');(evid/'ROUND17_DIFFUSION_BASIS_AUDIT.json').write_text(json.dumps(basis_a,indent=2)+'\n');(evid/'ROUND17_PREFLIGHT_AUDIT.json').write_text(json.dumps(smoke,indent=2)+'\n')
    print(json.dumps({'PASS':smoke['PASS'],'parameter':param['PASS'],'history':hist_a['PASS'],'basis':basis_a['PASS']},sort_keys=True)); return smoke

def run_d0(seed,root,evid):
    model,_,cfg,a,c,ab=load_backbone(seed); hs=HistoryStore(seed,c,model.device); bs=BasisStore(seed,model.device); ev=Evaluator(seed)
    true=ev.evaluate(model,c,bs,hs,'D0',None,'true','true'); neg=ev.evaluate(model,c,bs,hs,'D0',None,'true','negated'); mech=hard_shell_diag(model,c,bs,hs,ev,'D0',None)
    out={'status':'PASS','protocol':PROTOCOL,'seed':seed,'variant':'D0','trainable':False,'C0_metrics':ev.c0_metrics,'metrics':true['metrics'],'vs_C0':delta_pack(true['metrics'],ev.c0_metrics),
         'NEGATED':delta_pack(neg['metrics'],ev.c0_metrics),'hard_shell_mechanism':mech,'action_stats':true['action_stats'],'max_angle':true['max_angle'],'TEST_ACCESSED':False}
    p=Path(evid)/f'ROUND17_D0_SEED{seed}.json';p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'seed':seed,'variant':'D0','U':out['vs_C0']['U'],'R20':out['metrics']['R20']},sort_keys=True));del model;torch.cuda.empty_cache();return out

def run_variant(seed,variant,root,evid,max_batches=None):
    root=Path(root); outdir=root/f'seed{seed}/{variant}'; shutil.rmtree(outdir,ignore_errors=True);outdir.mkdir(parents=True,exist_ok=True)
    model,_,cfg,a,c,ab=load_backbone(seed); hs=HistoryStore(seed,c,model.device); bs=BasisStore(seed,model.device);tc=CleanTrain(seed,model.device);ev=Evaluator(seed)
    net=DecisionNet(decision_dim(variant)).to(model.device);seed_all(202617000+seed+10*VARIANTS.index(variant));opt=torch.optim.Adam(net.parameters(),lr=float(cfg['learning_rate'])*LR_SCALE,weight_decay=float(cfg['weight_decay'] or 0.0))
    mh0=state_sha256(model); base,_=load_base(seed,root,model.device);bh0=state_sha256(base);logs=[]
    for ep in range(1,EPOCHS+1):
        tr=train_epoch(seed,ep,model,c,bs,hs,tc,net,opt,variant,max_batches); net.eval(); val=ev.evaluate(model,c,bs,hs,variant,net,'true','true'); sh=ev.evaluate(model,c,bs,hs,variant,net,'shuffled','true'); mech=hard_shell_diag(model,c,bs,hs,ev,variant,net)
        sane=mech['Delta_margin']['mean']>0 and mech['history_advantage']['mean']>0;save_net(outdir/f'checkpoint_ep{ep}.pt',net,seed,variant,ep)
        rec={'epoch':ep,'train':tr,'validation_metrics':val['metrics'],'U_vs_C0':utility(val['metrics'],ev.c0_metrics),'U_history_shuffled':utility(sh['metrics'],ev.c0_metrics),'mechanism':mech,'mechanism_sane':bool(sane)};logs.append(rec)
        print(json.dumps({'seed':seed,'variant':variant,'epoch':ep,'U':rec['U_vs_C0'],'U_shuf':rec['U_history_shuffled'],'R20':val['metrics']['R20'],'D':mech['Delta_margin']['mean'],'Dfrac':mech['fraction_Delta_margin_gt0'],'H':mech['history_advantage']['mean'],'Hfrac':mech['fraction_true_gt_shuffled'],'gate_abs':mech['action_stats']['mean_abs_gate']},sort_keys=True),flush=True)
    sane=[x for x in logs if x['mechanism_sane']]; inelig=not bool(sane); chosen=max(sane if sane else logs,key=lambda x:x['validation_metrics']['R20']); bep=chosen['epoch'];shutil.copy2(outdir/f'checkpoint_ep{bep}.pt',outdir/'best_checkpoint.pt');restore_net(outdir/'best_checkpoint.pt',net);net.eval()
    true=ev.evaluate(model,c,bs,hs,variant,net,'true','true');shuf=ev.evaluate(model,c,bs,hs,variant,net,'shuffled','true');neg=ev.evaluate(model,c,bs,hs,variant,net,'true','negated');mech=hard_shell_diag(model,c,bs,hs,ev,variant,net)
    out={'status':'PASS','protocol':PROTOCOL,'seed':seed,'variant':variant,'best_epoch':bep,'PASS_INELIGIBLE':inelig,'selection_policy':'mechanism-sane mean Delta_margin>0 AND mean true-minus-shuffled>0; then max Validation R20; else max R20 report-only',
         'C0_metrics':ev.c0_metrics,'best_metrics':true['metrics'],'best_vs_C0':delta_pack(true['metrics'],ev.c0_metrics),
         'history_control':{'label':'GLOBAL_USER_SHUFFLED' if variant=='D1' else ('HISTORY_SHUFFLED_CTX_FIXED' if variant=='D3' else 'HISTORY_SHUFFLED'), 'metrics':shuf['metrics'],'vs_C0':delta_pack(shuf['metrics'],ev.c0_metrics)},
         'negated_basis':{'metrics':neg['metrics'],'vs_C0':delta_pack(neg['metrics'],ev.c0_metrics)},'hard_shell_mechanism':mech,'action_stats':true['action_stats'],'epoch_logs':logs,
         'frozen_reference':{'backbone_hash_initial':mh0,'backbone_hash_final':state_sha256(model),'D_base_hash_initial':bh0,'D_base_hash_final':state_sha256(base),'hashes_unchanged':bool(state_sha256(model)==mh0 and state_sha256(base)==bh0)},
         'bound':{'max_angle':max(true['max_angle'],shuf['max_angle'],neg['max_angle'])},'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
    (outdir/'result.json').write_text(json.dumps(out,indent=2)+'\n');(Path(evid)/f'ROUND17_{variant}_SEED{seed}.json').write_text(json.dumps(out,indent=2)+'\n')
    np.savez_compressed(outdir/'best_validation_rankings.npz',users=ev.users,ranked_items=true['rank'],candidate_items=ev.items,final_scores=true['final_scores'])
    del model,base,net,opt;torch.cuda.empty_cache();gc.collect();return out

def run_oracle(seed,evid):
    model,_,cfg,a,c,ab=load_backbone(seed); hs=HistoryStore(seed,c,model.device); bs=BasisStore(seed,model.device);ev=Evaluator(seed)
    out=oracle_diag(model,c,bs,ev);out.update({'seed':seed});p=Path(evid)/f'ROUND17_ORACLE_SEED{seed}.json';p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'seed':seed,'oracle_frac':out['fraction_oracle_improves_BASE'],'oracle_mean':out['best_Delta_margin']['mean']},sort_keys=True));del model;torch.cuda.empty_cache();return out

def run_random(seed,variant,root,evid):
    out=random_basis_control(seed,variant,root);(Path(evid)/f'ROUND17_RANDOM_BASIS_{variant}_SEED{seed}.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'seed':seed,'variant':variant,'U':out['U_TRUE_DIFFUSION'],'RND':out['mean_U_RANDOM_BASIS'],'NEG':out['U_NEGATED_BASIS'],'PASS':out['PASS']},sort_keys=True));return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',required=True,choices=['preflight','d0','variant','oracle','random']);ap.add_argument('--seed',type=int);ap.add_argument('--variant',choices=('D1','D2','D3'));ap.add_argument('--max-batches',type=int);ap.add_argument('--root',default=str(RDIR/'outputs'));ap.add_argument('--evidence',default=str(RDIR/'evidence'));a=ap.parse_args()
    if a.mode=='preflight': preflight(a.root,a.evidence)
    elif a.mode=='d0': run_d0(a.seed,a.root,a.evidence)
    elif a.mode=='variant': run_variant(a.seed,a.variant,a.root,a.evidence,a.max_batches)
    elif a.mode=='oracle': run_oracle(a.seed,a.evidence)
    elif a.mode=='random': run_random(a.seed,a.variant,a.root,a.evidence)
if __name__=='__main__': main()
