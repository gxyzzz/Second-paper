from __future__ import annotations
import gc,json
from pathlib import Path
import numpy as np
import torch
from diffusion_experiments.round32_round26_parameter_sweep import round32_core as c
from diffusion_experiments.round20_positive_anchored_hardneg.run_round20_test import TestEvaluator,evaluate_detail,bootstrap

R26E=c.ROOT/'diffusion_experiments/round26_preference_directed_diffusion/evidence'
R31E=c.ROOT/'diffusion_experiments/round31_round26_baby_test_confirmation/evidence'

def _r26_b0_train(seed):
    return json.loads((R26E/f'ROUND26_B0_SEED{seed}.json').read_text())

def _r31_summary():
    return json.loads((R31E/'ROUND31_TEST_SUMMARY.json').read_text())

def _load_model_from_round32(seed,variant):
    tr=json.loads((c.EVID/f'ROUND32_{variant}_SEED{seed}_TRAIN.json').read_text())
    p=Path(tr['best_checkpoint'])
    if c.sha256_file(p)!=tr['checkpoint_sha256']:
        raise RuntimeError(f'checkpoint SHA mismatch {seed} {variant}')
    cp=torch.load(p,map_location='cpu',weights_only=False)
    if cp.get('protocol')!=c.PROTOCOL or int(cp.get('seed',-1))!=seed or cp.get('variant')!=variant or int(cp.get('epoch',-1))!=int(tr['best_epoch']):
        raise RuntimeError('Round32 checkpoint metadata mismatch')
    model,_,_,_,_=c.instantiate_shared(seed)
    model.load_state_dict(cp['model_state_dict'],strict=True)
    model.eval()
    return model,tr

def _load_round26_b0_model(seed):
    tr=_r26_b0_train(seed)
    p=Path(tr['best_checkpoint'])
    if c.sha256_file(p)!=tr['checkpoint_sha256']:
        raise RuntimeError('Round26 B0 SHA mismatch')
    cp=torch.load(p,map_location='cpu',weights_only=False)
    model,_,_,_,_=c.instantiate_shared(seed)
    model.load_state_dict(cp['model_state_dict'],strict=True)
    model.eval()
    return model,tr

def ensure_b0_test_cache(seed):
    npz=c.ASSETS/f'ROUND32_B0_TEST_PERUSER_SEED{seed}.npz'
    jp=c.EVID/f'ROUND32_B0_REFERENCE_SEED{seed}.json'
    if npz.exists() and jp.exists():
        x=json.loads(jp.read_text())
        arr=np.load(npz)['primary']
        return x,arr
    ev=TestEvaluator(seed)
    model,tr=_load_round26_b0_model(seed)
    agg,arr=evaluate_detail(ev,model)
    ref=_r31_summary()['seeds'][str(seed)]['Test']['B0']
    parity=max(abs(float(agg['colift'][k])-float(ref[k])) for k in c.ALL)
    if parity>1e-12:
        raise RuntimeError(f'Round31 B0 Test parity failed seed{seed}: {parity}')
    np.savez_compressed(npz,primary=arr)
    out={'protocol':c.PROTOCOL,'seed':seed,'source':'Round26 B0 Validation-selected checkpoint + Round31 Test reference',
         'best_epoch':tr['best_epoch'],'checkpoint':tr['best_checkpoint'],'checkpoint_sha256':tr['checkpoint_sha256'],
         'Validation':tr['best_evaluation']['colift'],'Test':agg['colift'],'round31_test_parity_max_abs_error':parity,
         'per_user_asset':str(npz),'TEST_ACCESSED':True,'BABY_TEST_ROLE':'DEVELOPMENT_BENCHMARK_AFTER_ROUND31',
         'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
    c.json_write(jp,out)
    del model,ev
    torch.cuda.empty_cache()
    gc.collect()
    return out,arr

def evaluate_variant(seed,variant):
    cfg=c.apply_variant(variant)
    b0,b0arr=ensure_b0_test_cache(seed)
    model,tr=_load_model_from_round32(seed,variant)
    ev=TestEvaluator(seed)
    agg,arr=evaluate_detail(ev,model)
    if agg['diffusion_inference_calls']!=0:
        raise RuntimeError('Diffusion inference must be zero at Test')
    vmetrics=tr['best_evaluation']['colift']
    vdelta=c.delta_pack(vmetrics,b0['Validation'])
    tdelta=c.delta_pack(agg['colift'],b0['Test'])
    boot=bootstrap(b0arr,arr,1000,20263200+seed+sum(map(ord,variant)))
    out={'protocol':c.PROTOCOL,'protocol_hash':c.protocol_hash(),'seed':seed,'variant':variant,'config':cfg,
         'best_epoch':tr['best_epoch'],'checkpoint':tr['best_checkpoint'],'checkpoint_sha256':tr['checkpoint_sha256'],
         'Validation':vmetrics,'Validation_vs_B0':vdelta,'Test':agg['colift'],'Test_vs_B0':tdelta,
         'bootstrap':boot,'mechanism':tr['mechanism_at_best'],'test_user_count':agg['test_user_count'],
         'candidate_shape':agg['candidate_shape'],'diffusion_inference_calls':agg['diffusion_inference_calls'],
         'BABY_TEST_ROLE':'DEVELOPMENT_BENCHMARK_AFTER_ROUND31','TEST_USED_FOR_PARAMETER_COMPARISON':True,
         'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
    c.json_write(c.EVID/f'ROUND32_{variant}_SEED{seed}.json',out)
    print(json.dumps({'seed':seed,'variant':variant,'best_epoch':tr['best_epoch'],'V_U4':vdelta['U'],
                      'T_U4':tdelta['U'],'T_primary':tdelta['primary_positive_count'],'bootstrap':boot},sort_keys=True),flush=True)
    del model,ev
    torch.cuda.empty_cache()
    gc.collect()
    return out

def smoke_test_evaluator(seed=999):
    ref,arr=ensure_b0_test_cache(seed)
    return {'seed':seed,'B0_Test':ref['Test'],'round31_parity':ref['round31_test_parity_max_abs_error'],
            'per_user_shape':list(arr.shape),'PASS':bool(ref['round31_test_parity_max_abs_error']<=1e-12),
            'TEST_ACCESSED':True,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
