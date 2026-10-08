from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT))

from pipelines.dataset_config import load_dataset_config
from pipelines.msca_assets import load_msca_checkpoint, build_train_histories_and_validation
from pipelines.coliftrec import _params
from modules.ranking import topk_from_embeddings, semantic_z_for_candidates, rank_by_score, sha256_file
from modules.attribute import build_item_matrices, build_profiles, attribute_z
from modules.coliftrec import score_coliftrec
from diffusion_experiments.round9_cabrp import run_round9 as r9
from diffusion_experiments.round13_iudlp import latent_diffusion as r13ld


def checkpoint_audit(seed:int):
    p=r9.paths(seed)['msca']/'audit.json'
    a=json.loads(p.read_text())
    if a.get('TEST_ACCESSED') is not False:
        raise RuntimeError('contaminated checkpoint asset')
    return a


def build(seed:int, out_path:Path, audit_path:Path, smoke_users:int|None=None):
    cfg=load_dataset_config('baby')
    paths=cfg['resolved_paths']
    a=checkpoint_audit(seed)
    model, ck, _, _ = load_msca_checkpoint(Path(a['checkpoint']),0)
    r13ld.freeze_recommender(model)
    n_users,n_items=int(a['n_users']),int(a['n_items'])
    histories,pseudo_hist,pseudo_users,valid_users,valid_sets=build_train_histories_and_validation(paths['interaction'],n_users)
    train_users=np.asarray([u for u,h in enumerate(histories) if h],dtype=np.int64)
    if smoke_users is not None:
        train_users=train_users[:int(smoke_users)]
    with torch.no_grad():
        c=r13ld.forward_components(model)
        items,msca=topk_from_embeddings(c['final_user'],c['final_item'],train_users,histories,top_l=100,batch_users=1024)
    zt,_=semantic_z_for_candidates(paths['text_feature'],histories,train_users,items,batch_users=256)
    zv,_=semantic_z_for_candidates(paths['visual_feature'],histories,train_users,items,batch_users=128)
    acfg=cfg['coliftrec']['attribute']
    mats,_=build_item_matrices(paths['metadata'],n_items,min_df=int(acfg.get('tfidf_min_df',2)),max_df=float(acfg.get('tfidf_max_df',.8)),description_len=int(acfg.get('description_len',128)),weights=acfg.get('weights'))
    profiles=build_profiles(mats,histories,n_items)
    za,_=attribute_z(mats,profiles,train_users,items,batch=256,weights=acfg.get('weights'))
    bgf=np.load(r9.paths(seed)['colift']/'backgrounds.npz')
    backgrounds={
      'text':{'shrunk_mean':bgf['mu_text']},
      'attribute':{'shrunk_mean':bgf['mu_attribute']},
      'visual':{'shrunk_mean':bgf['mu_visual']},
    }
    p=_params(cfg['coliftrec'])
    enabled={m:bool(cfg['coliftrec'][m]['enabled']) for m in ('text','attribute','visual')}
    full,_=score_coliftrec(msca,items,zt,za,zv,backgrounds,p,enabled)
    ranked=rank_by_score(items,full)
    hard=ranked[:,5:30].astype(np.int32,copy=True)
    hard_ranks=np.tile(np.arange(6,31,dtype=np.int16),(len(train_users),1))
    observed_violations=0
    for r,u in enumerate(train_users):
        hs=set(map(int,hard[r]))
        observed_violations += sum(int(i) in hs for i in histories[int(u)])
    if observed_violations:
        raise RuntimeError(f'hard pool contains observed TRAIN positives: {observed_violations}')
    out_path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out_path,users=train_users,hard_items=hard,hard_ranks=hard_ranks,top100=ranked.astype(np.int32),source_msca_top100=items.astype(np.int32))
    hist={f'{lo}-{hi}':int(np.sum((hard_ranks>=lo)&(hard_ranks<=hi))) for lo,hi in ((6,10),(11,15),(16,20),(21,25),(26,30))}
    audit={
      'status':'PASS','seed':int(seed),'mode':'smoke' if smoke_users is not None else 'formal',
      'checkpoint':a['checkpoint'],'checkpoint_sha256':a['checkpoint_sha256'],
      'interaction_path':str(paths['interaction']),'interaction_sha256':sha256_file(paths['interaction']),
      'backgrounds_path':str((r9.paths(seed)['colift']/'backgrounds.npz').resolve()),
      'definition':'Frozen MSCA Top100 masked by x_label==0 TRAIN histories, reranked by frozen Full CoLiftRec; hard region rank 6-30.',
      'TRAIN_ONLY':True,'validation_positive_used':False,'validation_metric_used':False,'test_used':False,
      'train_users_with_valid_hard_pool':int(len(train_users)),'mean_candidates_per_user':float(hard.shape[1]),'median_candidates_per_user':float(hard.shape[1]),
      'fallback_ratio':0.0,'mean_frozen_rank':float(hard_ranks.mean()),'rank_histogram':hist,
      'observed_positive_violations':int(observed_violations),'hard_shape':list(hard.shape),
      'output':str(out_path.resolve()),'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False,
    }
    audit_path.parent.mkdir(parents=True,exist_ok=True)
    audit_path.write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,sort_keys=True))
    del model
    torch.cuda.empty_cache()
    return audit

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--audit',required=True)
    ap.add_argument('--smoke-users',type=int)
    z=ap.parse_args()
    build(z.seed,Path(z.out),Path(z.audit),z.smoke_users)
