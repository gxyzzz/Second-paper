from __future__ import annotations
import gc, hashlib, json
from pathlib import Path
import numpy as np
import torch
from diffusion_experiments.round31_round26_baby_test_confirmation import round31_core as c
from diffusion_experiments.round26_preference_directed_diffusion import round26_core as r26
from diffusion_experiments.round20_positive_anchored_hardneg.run_round20_test import TestEvaluator, evaluate_detail
from diffusion_experiments.round18_diffusion_hard_negative.round18_core import topk_from_embeddings

def _hash_array(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def _gt_hash(users,sets):
    h=hashlib.sha256()
    for u in np.asarray(users,dtype=np.int64):
        arr=np.asarray(sorted(int(x) for x in sets[int(u)]),dtype=np.int64)
        h.update(np.asarray([int(u),len(arr)],dtype=np.int64).tobytes()); h.update(arr.tobytes())
    return h.hexdigest()
def _params_hash(ev): return hashlib.sha256(repr(ev.params).encode()).hexdigest()

def load_model(seed,variant):
    src=c.source_result(seed,variant); p=Path(src['best_checkpoint']); cp=torch.load(p,map_location='cpu',weights_only=False)
    if cp.get('protocol')!=c.SOURCE_PROTOCOL or int(cp.get('seed',-1))!=seed or cp.get('variant')!=variant or int(cp.get('epoch',-1))!=int(src['best_epoch']): raise RuntimeError(f'checkpoint metadata mismatch {seed} {variant}')
    model,config,td,vd,pack=r26.instantiate_shared(seed); model.load_state_dict(cp['model_state_dict'],strict=True); model.eval(); return model

@torch.no_grad()
def candidate_audit(ev,model):
    model.eval(); fu,fi=model.forward(test=True); ti,_=topk_from_embeddings(fu,fi,ev.test_users,ev.h.histories,100,1024)
    return {'test_user_hash':_hash_array(np.asarray(ev.test_users,dtype=np.int64)),'candidate_ids_hash':_hash_array(np.asarray(ti,dtype=np.int64)),'candidate_set_hash':_hash_array(np.sort(np.asarray(ti,dtype=np.int64),axis=1)),'ground_truth_hash':_gt_hash(ev.test_users,ev.test_sets),'candidate_shape':[int(ti.shape[0]),int(ti.shape[1])],'test_user_count':int(len(ev.test_users)),'interaction_sha256':c.sha256_file(ev.paths['interaction']),'colift_params_hash':_params_hash(ev),'candidate_protocol':'topk_from_embeddings(frozen checkpoint, TRAIN history exclusion, K=100)','metric_pipeline':'existing evaluate_detail -> Full CoLiftRec -> metrics_at'}

def _metric_diff(a,b): return max(abs(float(a[k])-float(b[k])) for k in c.ALL)

def run_checkpoint(seed,variant):
    audit=json.loads((c.EVID/'ROUND31_CHECKPOINT_AUDIT.json').read_text())
    if not audit.get('PASS'): raise RuntimeError('checkpoint audit must PASS before Baby Test')
    reps=[]
    for rep in (1,2):
        ev=TestEvaluator(seed); model=load_model(seed,variant); agg,_=evaluate_detail(ev,model); proto=candidate_audit(ev,model)
        if agg.get('diffusion_inference_calls')!=0: raise RuntimeError('Round26 D3 Test must use zero diffusion inference calls')
        reps.append({'repeat':rep,'metrics':agg,'protocol_audit':proto})
        del model,ev; torch.cuda.empty_cache(); gc.collect()
    backerr=_metric_diff(reps[0]['metrics']['backbone'],reps[1]['metrics']['backbone']); colerr=_metric_diff(reps[0]['metrics']['colift'],reps[1]['metrics']['colift'])
    ordered_candidate_hash_exact=reps[0]['protocol_audit']['candidate_ids_hash']==reps[1]['protocol_audit']['candidate_ids_hash']
    candidate_set_hash_exact=reps[0]['protocol_audit']['candidate_set_hash']==reps[1]['protocol_audit']['candidate_set_hash']
    proto_exact=all(reps[0]['protocol_audit'][k]==reps[1]['protocol_audit'][k] for k in ('test_user_hash','ground_truth_hash','candidate_shape','test_user_count','interaction_sha256','colift_params_hash'))
    repeat_pass=bool(backerr<=1e-12 and colerr<=1e-12 and proto_exact)
    out={'protocol':c.PROTOCOL,'protocol_hash':c.protocol_hash(),'source_protocol':c.SOURCE_PROTOCOL,'seed':seed,'variant':variant,'source_best_epoch':c.source_result(seed,variant)['best_epoch'],'source_checkpoint':c.source_result(seed,variant)['best_checkpoint'],'source_checkpoint_sha256':c.source_result(seed,variant)['checkpoint_sha256'],'repeats':reps,'canonical_metrics':reps[0]['metrics'],'repeat_backbone_max_abs_error':backerr,'repeat_colift_max_abs_error':colerr,'repeat_candidate_protocol_exact':proto_exact,'repeat_ordered_candidate_hash_exact':ordered_candidate_hash_exact,'repeat_candidate_set_hash_exact':candidate_set_hash_exact,'REPRODUCIBILITY_PASS':repeat_pass,'test_used_for_selection':False,'no_post_test_tuning':True,'TEST_ACCESSED':True,'BABY_TEST_STATUS':'OPENED_FOR_ROUND31_FROZEN_CONFIRMATION','SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
    c.json_write(c.EVID/f'ROUND31_{variant}_SEED{seed}_TEST.json',out)
    if not repeat_pass: raise RuntimeError(f'Test reproducibility BLOCK {seed} {variant}')
    return out

def protocol_audit():
    rows={}
    for seed in c.SEEDS:
        rows[str(seed)]={}
        for variant in c.VARIANTS:
            x=json.loads((c.EVID/f'ROUND31_{variant}_SEED{seed}_TEST.json').read_text()); rows[str(seed)][variant]=x
    user_hashes={rows[str(s)][v]['repeats'][0]['protocol_audit']['test_user_hash'] for s in c.SEEDS for v in c.VARIANTS}; gt_hashes={rows[str(s)][v]['repeats'][0]['protocol_audit']['ground_truth_hash'] for s in c.SEEDS for v in c.VARIANTS}; interaction_hashes={rows[str(s)][v]['repeats'][0]['protocol_audit']['interaction_sha256'] for s in c.SEEDS for v in c.VARIANTS}; param_hashes={rows[str(s)][v]['repeats'][0]['protocol_audit']['colift_params_hash'] for s in c.SEEDS for v in c.VARIANTS}; shapes={tuple(rows[str(s)][v]['repeats'][0]['protocol_audit']['candidate_shape']) for s in c.SEEDS for v in c.VARIANTS}
    all_rep=all(rows[str(s)][v]['REPRODUCIBILITY_PASS'] for s in c.SEEDS for v in c.VARIANTS)
    out={'protocol':c.PROTOCOL,'protocol_hash':c.protocol_hash(),'test_evaluator_source':'diffusion_experiments.round20_positive_anchored_hardneg.run_round20_test.TestEvaluator/evaluate_detail','test_split':'official Baby Test via build_test_eval','same_test_users_across_all_checkpoints':len(user_hashes)==1,'same_ground_truth_across_all_checkpoints':len(gt_hashes)==1,'same_interaction_file_across_all_checkpoints':len(interaction_hashes)==1,'same_full_colift_parameters_across_all_checkpoints':len(param_hashes)==1,'same_candidate_shape_across_all_checkpoints':shapes=={(19445,100)},'candidate_ids_note':'ordered and set candidate hashes are recorded diagnostically; the blocking reproducibility requirement follows the Round31 guide: all metrics exact within 1e-12 plus identical Test users, ground truth, split, candidate-generation protocol, shape, interaction source and Full CoLift parameters; B0 and D3 candidate outputs may differ because the frozen trained checkpoint is the only intended change','raw_hashes':{str(s):{v:rows[str(s)][v]['repeats'][0]['protocol_audit'] for v in c.VARIANTS} for s in c.SEEDS},'all_checkpoint_repeats_exact':all_rep,'diffusion_inference_calls_all_zero':all(rows[str(s)][v]['canonical_metrics']['diffusion_inference_calls']==0 for s in c.SEEDS for v in c.VARIANTS),'ONLY_CHANGE_B0_VS_D3':'trained frozen checkpoint','TEST_USED_FOR_SELECTION':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
    out['PASS']=bool(out['same_test_users_across_all_checkpoints'] and out['same_ground_truth_across_all_checkpoints'] and out['same_interaction_file_across_all_checkpoints'] and out['same_full_colift_parameters_across_all_checkpoints'] and out['same_candidate_shape_across_all_checkpoints'] and all_rep and out['diffusion_inference_calls_all_zero'])
    c.json_write(c.EVID/'ROUND31_TEST_PROTOCOL_AUDIT.json',out)
    if not out['PASS']: raise RuntimeError('Round31 Test protocol audit failed')
    return out

def summarize():
    val=c.validation_summary()
    out={
        'protocol':c.PROTOCOL,
        'protocol_hash':c.protocol_hash(),
        'source_protocol':c.SOURCE_PROTOCOL,
        'seeds':{},
        'Validation_mean_U4':float(val['mean_U_D3_vs_B0']),
        'TEST_ACCESSED':True,
        'TEST_USED_FOR_SELECTION':False,
        'SPORTS_ACCESSED':False,
        'ELECTRONICS_ACCESSED':False,
    }
    tus=[]
    for seed in c.SEEDS:
        b=json.loads((c.EVID/f'ROUND31_B0_SEED{seed}_TEST.json').read_text())
        d=json.loads((c.EVID/f'ROUND31_D3_SEED{seed}_TEST.json').read_text())
        vm=val['seeds'][str(seed)]
        cmp=c.delta_pack(d['canonical_metrics']['colift'],b['canonical_metrics']['colift'])
        tus.append(cmp['U4'])
        out['seeds'][str(seed)]={
            'Validation':{
                'B0':vm['best_metrics']['B0'],
                'D3':vm['best_metrics']['D3'],
                'D3_vs_B0':{
                    'U4':vm['D3_vs_B0']['U'],
                    'relative_delta':vm['D3_vs_B0']['relative_delta'],
                    'primary_positive_count':vm['D3_vs_B0']['primary_positive_count'],
                    'overall_positive_count':vm['D3_vs_B0']['overall_positive_count'],
                },
            },
            'Test':{
                'B0':b['canonical_metrics']['colift'],
                'D3':d['canonical_metrics']['colift'],
                'D3_vs_B0':cmp,
            },
            'reproducibility':{
                'B0':b['REPRODUCIBILITY_PASS'],
                'D3':d['REPRODUCIBILITY_PASS'],
            },
        }
    mean=float(np.mean(tus))
    out['Test_mean_U4']=mean
    pos=[x>0 for x in tus]
    strong=bool(
        all(pos) and mean>=.01 and
        all(out['seeds'][str(s)]['Test']['D3_vs_B0']['primary_positive_count']>=3 for s in c.SEEDS)
    )
    if strong:
        status='STRONG_PASS'
    elif all(pos):
        status='PASS_BELOW_TARGET'
    elif any(pos):
        status='MIXED'
    else:
        status='FAIL'
    out['ROUND31_STATUS']=status
    out['validation_to_test_mean_U4_delta']=float(mean-out['Validation_mean_U4'])
    out['NO_POST_TEST_RESCUE']=True
    c.json_write(c.EVID/'ROUND31_TEST_SUMMARY.json',out)
    return out
