from __future__ import annotations
import gc, hashlib
import numpy as np
import torch
from diffusion_experiments.round30_candidate_side_diffusion_purification import round30_core as c
from diffusion_experiments.round30_candidate_side_diffusion_purification.round30_colift_reference import FrozenCoLiftReference
from diffusion_experiments.round30_candidate_side_diffusion_purification.round30_candidate_field import FieldComputer, dynamic_field
from diffusion_experiments.round30_candidate_side_diffusion_purification import round30_diffusion as d
from modules.ranking import topk_from_embeddings, metrics_at, rank_by_score, row_zscore, histories_csr, l2_normalize_rows
from modules.attribute import attribute_z, build_item_matrices, build_profiles
from modules.coliftrec import fit_backgrounds, score_coliftrec
from pipelines.coliftrec import _params
from pipelines.dataset_config import load_dataset_config
from pipelines.msca_assets import build_train_histories_and_validation

class FrozenRankingContext:
    def __init__(self,seed,model):
        self.seed=int(seed); self.model=model; self.cfg=load_dataset_config('baby'); p=self.cfg['resolved_paths']; self.ccfg=self.cfg['coliftrec']; self.params=_params(self.ccfg); self.enabled={m:bool(self.ccfg[m]['enabled']) for m in ('text','attribute','visual')}
        self.hist,self.pseudo,self.pusers,self.vusers,self.vsets=build_train_histories_and_validation(p['interaction'],model.n_users); self.n_items=model.n_items
        text=l2_normalize_rows(np.asarray(np.load(p['text_feature'],mmap_mode='r'),dtype=np.float32)); visual=l2_normalize_rows(np.asarray(np.load(p['visual_feature'],mmap_mode='r'),dtype=np.float32)); hf=histories_csr(self.hist,self.n_items,users=self.vusers,mean=True); hp=histories_csr(self.pseudo,self.n_items,users=self.pusers,mean=True)
        tf=l2_normalize_rows(np.asarray(hf@text,dtype=np.float32)); tp=l2_normalize_rows(np.asarray(hp@text,dtype=np.float32)); vf=l2_normalize_rows(np.asarray(hf@visual,dtype=np.float32)); vp=l2_normalize_rows(np.asarray(hp@visual,dtype=np.float32))
        self.text=torch.as_tensor(text,device=model.device); self.visual=torch.as_tensor(visual,device=model.device); self.tf=torch.as_tensor(tf,device=model.device); self.tp=torch.as_tensor(tp,device=model.device); self.vf=torch.as_tensor(vf,device=model.device); self.vp=torch.as_tensor(vp,device=model.device)
        ac=self.ccfg['attribute']; self.mats,_=build_item_matrices(p['metadata'],self.n_items,min_df=int(ac.get('tfidf_min_df',2)),max_df=float(ac.get('tfidf_max_df',.8)),description_len=int(ac.get('description_len',128)),weights=ac.get('weights')); self.af=build_profiles(self.mats,self.hist,self.n_items); self.ap=build_profiles(self.mats,self.pseudo,self.n_items); self.aw=ac.get('weights'); self.prepare()
    def sem(self,feat,prof,items,batch):
        outs=[]
        for st in range(0,len(items),batch):
            en=min(st+batch,len(items)); ids=torch.as_tensor(items[st:en],device=self.model.device); outs.append(torch.einsum('bld,bd->bl',feat[ids],prof[st:en]).cpu())
        return row_zscore(torch.cat(outs).numpy())
    @torch.no_grad()
    def prepare(self):
        self.model.eval(); self.fu,self.fi=self.model.forward(test=True); self.vi,self.vs=topk_from_embeddings(self.fu,self.fi,self.vusers,self.hist,100,1024); pi,ps=topk_from_embeddings(self.fu,self.fi,self.pusers,self.pseudo,100,1024)
        ztp=self.sem(self.text,self.tp,pi,512); ztv=self.sem(self.text,self.tf,self.vi,512); zvp=self.sem(self.visual,self.vp,pi,128); zvv=self.sem(self.visual,self.vf,self.vi,128); zap,_=attribute_z(self.mats,self.ap,self.pusers,pi,256,self.aw); zav,_=attribute_z(self.mats,self.af,self.vusers,self.vi,256,self.aw); bg=fit_backgrounds(pi,ztp,zap,zvp,self.n_items)
        self.full_scores,self.parts=score_coliftrec(self.vs,self.vi,ztv,zav,zvv,bg,self.params,self.enabled); self.b0_ranked=rank_by_score(self.vi,self.full_scores); self.b0_metrics=metrics_at(self.b0_ranked,self.vusers,self.vsets); self.b0_order=np.argsort(-self.full_scores,axis=1,kind='stable'); self.candidate_hash=hashlib.sha256(np.ascontiguousarray(self.vi).tobytes()).hexdigest()
        expected=c.load_round28_b0(self.seed)['best_evaluation']['colift']; err=max(abs(self.b0_metrics[k]-expected[k]) for k in c.ALL)
        if err>1e-10: raise RuntimeError(f'Round30 B0 reproduction mismatch seed{self.seed}: {err}')
        self.b0_reproduction_max_abs_error=float(err)

def _static(z,sl,device):
    return {'support_ids':torch.as_tensor(z['support_ids'][sl],device=device,dtype=torch.long),'locality':z['locality'][sl],'local_mask':torch.as_tensor(z['local_mask'][sl],device=device),'q':z['q'][sl],'pref_mask':torch.as_tensor(z['pref_mask'][sl],device=device),'history_size':z['history_size'][sl]}
def _cat(rows,key): return np.concatenate([np.asarray(x[key]).reshape(-1) for x in rows]) if rows else np.array([])

@torch.no_grad()
def evaluate_d7(seed,net,sched,model,events,reference=None,max_users=None,write_evidence=False):
    ctx=FrozenRankingContext(seed,model); n=len(ctx.vusers) if max_users is None else min(int(max_users),len(ctx.vusers)); rows=np.arange(n); users=ctx.vusers[rows]; vi=ctx.vi[rows].copy(); vs=ctx.vs[rows].copy(); full=ctx.full_scores[rows].copy(); order0=ctx.b0_order[rows]; cols=order0[:,:c.TOP_PURIFY]; ids=np.take_along_axis(vi,cols,axis=1); flat_u=np.repeat(users,c.TOP_PURIFY); flat_i=ids.reshape(-1)
    ref=reference or FrozenCoLiftReference(seed,model,events); comp=FieldComputer(seed,model,events,ref); pairz=comp.compute_pairs(flat_u,flat_i,256); b=c.forward_bundle(model); s=c.latent_stats(model); base_pur=vs.copy(); dir_rows=[]; geom_rows=[]
    for st in range(0,len(flat_u),256):
        en=min(st+256,len(flat_u)); sl=slice(st,en); ut=torch.as_tensor(flat_u[sl],device=model.device,dtype=torch.long); it=torch.as_tensor(flat_i[sl],device=model.device,dtype=torch.long); field=dynamic_field(b,it,_static(pairz,sl,model.device)); dirs,diag=d.predict_ensemble(net,sched,b,ut,it,s,field); purified,geom=d.purify_candidates(b['fu'][ut],b['fi'][it],dirs,field['confidence']); newscore=(b['fu'][ut]*purified).sum(1).cpu().numpy(); rr=np.arange(st,en); userrow=rr//c.TOP_PURIFY; localcol=rr%c.TOP_PURIFY; original_col=cols[userrow,localcol]; base_pur[userrow,original_col]=newscore; dir_rows.append(diag); geom_rows.append(geom)
    zbase=row_zscore(base_pur); d7=zbase.copy()
    if ctx.enabled.get('text',False): d7+=ctx.params.alpha_text*ctx.parts['text'][rows]
    if ctx.enabled.get('attribute',False): d7+=ctx.params.alpha_attribute*ctx.parts['attribute'][rows]
    if ctx.enabled.get('visual',False): d7+=ctx.params.alpha_visual*ctx.parts['visual'][rows]
    ranked=rank_by_score(vi,d7); subsets={int(u):ctx.vsets[int(u)] for u in users}; metrics=metrics_at(ranked,users,subsets); bmetrics=metrics_at(ctx.b0_ranked[rows],users,subsets)
    top30_changed=np.mean([not np.array_equal(ctx.b0_ranked[rows][r,:30],ranked[r,:30]) for r in range(n)]); top10_changed=np.mean([set(ctx.b0_ranked[rows][r,:10])!=set(ranked[r,:10]) for r in range(n)]); top20_changed=np.mean([set(ctx.b0_ranked[rows][r,:20])!=set(ranked[r,:20]) for r in range(n)]); score_abs=np.abs(d7-full)
    geom={'confidence':c.stats(_cat(geom_rows,'confidence')),'angle_deg':c.stats(_cat(geom_rows,'angle_deg')),'norm_relative_error':c.stats(_cat(geom_rows,'norm_relative_error')),'base_score_delta':c.stats(_cat(geom_rows,'score_delta'))}
    shortcut=d.shortcut_stats({k:_cat(dir_rows,k) for k in ('g_field','g_diff','A_true','A_shuf','DeltaA','raw_residual_norm','sign_agreement')})
    change={'Top30_order_changed_fraction':float(top30_changed),'Top10_membership_changed_fraction':float(top10_changed),'Top20_membership_changed_fraction':float(top20_changed),'mean_absolute_final_score_delta':float(score_abs.mean()),'median_absolute_final_score_delta':float(np.median(score_abs)),'fraction_any_final_score_change':float((score_abs.max(1)>1e-12).mean())}
    out={'protocol':c.PROTOCOL,'protocol_hash':c.protocol_hash(),'seed':int(seed),'variant':c.VARIANT,'users_evaluated':int(n),'candidate_top100_hash':ctx.candidate_hash,'CANDIDATE_SET_CHANGED':False,'ADDITIONAL_RANKING_NETWORK':False,'COLIFT_PARAMETERS_CHANGED':False,'BACKBONE_CHANGED':False,'purification_scope':'B0_FULL_COLIFT_FINAL_RANK_1_30_ONLY','B0':bmetrics,'D7':metrics,'D7_vs_B0':c.comparison(metrics,bmetrics),'ranking_change':change,'geometry':geom,'shortcut':shortcut,'b0_reproduction_max_abs_error':ctx.b0_reproduction_max_abs_error,'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
    if write_evidence: c.json_write(c.EVID/f'ROUND30_SEED{seed}_VALIDATION.json',out)
    del comp; torch.cuda.empty_cache(); gc.collect(); return out
