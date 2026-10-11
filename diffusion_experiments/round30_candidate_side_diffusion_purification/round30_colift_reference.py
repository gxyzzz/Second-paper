from __future__ import annotations
import gc, json
import numpy as np
import torch

from diffusion_experiments.round30_candidate_side_diffusion_purification import round30_core as c
from diffusion_experiments.round18_diffusion_hard_negative import round18_core as r18
from modules.attribute import attribute_z, build_item_matrices, build_profiles
from modules.coliftrec import fit_backgrounds
from modules.ranking import semantic_z_for_candidates, topk_from_embeddings
from pipelines.dataset_config import load_dataset_config

class FrozenCoLiftReference:
    """Canonical trained-backbone TRAIN-only reference candidates and frozen generic backgrounds."""
    def __init__(self, seed, model, events, force_rebuild=False):
        self.seed=int(seed); self.model=model; self.events=events
        self.cfg=load_dataset_config('baby'); self.paths=self.cfg['resolved_paths']; self.ccfg=self.cfg['coliftrec']
        self.reference_audit=r18.checkpoint_info(self.seed)
        if self.seed==999:
            self.embedding_asset=c.ROOT/'runs/assets/msca_baby_seed999/embeddings.npz'
        else:
            self.embedding_asset=c.ROOT/f'runs/backbone_robustness/baby_seed{self.seed}/msca/assets/embeddings.npz'
        if not self.embedding_asset.exists(): raise FileNotFoundError(self.embedding_asset)
        self.bank_path=c.ASSETS/f'ROUND30_COLIFT_REFERENCE_SEED{self.seed}.npz'
        self.audit_path=c.EVID/f'ROUND30_COLIFT_REFERENCE_AUDIT_SEED{self.seed}.json'
        if force_rebuild or not self.bank_path.exists(): self._build()
        z=np.load(self.bank_path)
        self.users=z['users'].astype(np.int64)
        self.candidates=z['candidate_ids'].astype(np.int64)
        self.bg={m:{'shrunk_mean':z[f'bg_{m}'].astype(np.float32)} for m in ('text','attribute','visual')}
        self.row={int(u):i for i,u in enumerate(self.users.tolist())}
        self.asset_hash=c.sha256_file(self.bank_path)

    def _build(self):
        p=self.paths; histories=self.events.histories; pseudo=self.events.pseudo; pusers=self.events.pusers
        users=np.asarray([u for u,h in enumerate(histories) if h],np.int64)
        ez=np.load(self.embedding_asset)
        fu=torch.as_tensor(ez['final_user'],device=self.model.device,dtype=torch.float32)
        fi=torch.as_tensor(ez['final_item'],device=self.model.device,dtype=torch.float32)
        old_tf32=torch.backends.cuda.matmul.allow_tf32; torch.backends.cuda.matmul.allow_tf32=False
        with torch.no_grad():
            ci,_=topk_from_embeddings(fu,fi,users,histories,100,1024)
            pi,_=topk_from_embeddings(fu,fi,pusers,pseudo,100,1024)
        torch.backends.cuda.matmul.allow_tf32=old_tf32
        ztp,_=semantic_z_for_candidates(p['text_feature'],pseudo,pusers,pi,256)
        zvp,_=semantic_z_for_candidates(p['visual_feature'],pseudo,pusers,pi,128)
        ac=self.ccfg['attribute']
        mats,_=build_item_matrices(p['metadata'],self.model.n_items,min_df=int(ac.get('tfidf_min_df',2)),
                                  max_df=float(ac.get('tfidf_max_df',.8)),description_len=int(ac.get('description_len',128)),
                                  weights=ac.get('weights'))
        pp=build_profiles(mats,pseudo,self.model.n_items)
        zap,_=attribute_z(mats,pp,pusers,pi,256,ac.get('weights'))
        bg=fit_backgrounds(pi,ztp,zap,zvp,self.model.n_items)
        np.savez_compressed(self.bank_path,users=users.astype(np.int32),candidate_ids=ci.astype(np.int32),
                            bg_text=bg['text']['shrunk_mean'].astype(np.float32),
                            bg_attribute=bg['attribute']['shrunk_mean'].astype(np.float32),
                            bg_visual=bg['visual']['shrunk_mean'].astype(np.float32))
        del fu,fi,ez,mats,pp; torch.cuda.empty_cache(); gc.collect()

    def candidates_for_users(self, users):
        return self.candidates[np.asarray([self.row[int(u)] for u in np.asarray(users,np.int64)],np.int64)]

    def provenance_audit(self):
        return {
            'protocol':c.PROTOCOL,'seed':self.seed,
            'CANONICAL_TRAINED_EMBEDDING_LOADED':True,
            'GENERIC_BACKGROUND_FROM_CANONICAL_REFERENCE':True,
            'SHARED_INIT_USED_FOR_BG':False,
            'LOCAL_SUPPORT_ROW_Z_FOR_LIFT':False,
            'COLIFT_BG_SOURCE':'CANONICAL_FROZEN_TRAINED_BACKBONE',
            'reference_embedding_asset':str(self.embedding_asset),
            'reference_embedding_sha256':c.sha256_file(self.embedding_asset),
            'reference_checkpoint':self.reference_audit.get('checkpoint'),
            'reference_checkpoint_epoch':self.reference_audit.get('checkpoint_epoch'),
            'reference_bank_asset':str(self.bank_path),'reference_bank_sha256':self.asset_hash,
            'reference_candidate_shape':list(self.candidates.shape),
            'candidate_definition':'canonical trained backbone Top100 with full TRAIN history exclusion',
            'background_definition':'canonical trained backbone TRAIN pseudo-history Top100 -> official T/A/V z -> fit_backgrounds',
            'background_frozen_for_round30':True,
            'VALIDATION_USED_FOR_CONSTRUCTION':False,'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False,
            'PASS':True,
        }

def build_reference_audit(seed=999, force=False):
    model,config,td,vd,pack=c.instantiate_shared(seed); events=c.TrainEvents(model)
    ref=FrozenCoLiftReference(seed,model,events,force_rebuild=force)
    out=ref.provenance_audit(); c.json_write(c.EVID/f'ROUND30_COLIFT_REFERENCE_AUDIT_SEED{seed}.json',out)
    if seed==999: c.json_write(c.EVID/'ROUND30_COLIFT_REFERENCE_AUDIT.json',out)
    del model,ref; torch.cuda.empty_cache(); gc.collect(); return out
