from __future__ import annotations
import json
from pathlib import Path
import torch
from diffusion_experiments.round31_round26_baby_test_confirmation import round31_core as c

def checkpoint_audit():
    rows={}; passed=True
    for seed in c.SEEDS:
        rows[str(seed)]={}
        for variant in c.VARIANTS:
            src=c.source_result(seed,variant); p=Path(src['best_checkpoint']); exists=p.exists(); actual_sha=c.sha256_file(p) if exists else None
            cp=torch.load(p,map_location='cpu',weights_only=False) if exists else {}
            checks={
                'checkpoint_exists':bool(exists),
                'evidence_protocol_exact':src.get('protocol')==c.SOURCE_PROTOCOL,
                'evidence_protocol_hash_exact':src.get('protocol_hash')==c.SOURCE_PROTOCOL_HASH,
                'evidence_seed_exact':int(src.get('seed',-1))==seed,
                'evidence_variant_exact':src.get('variant')==variant,
                'best_epoch_expected':int(src.get('best_epoch',-1))==c.EXPECTED_EPOCHS[(seed,variant)],
                'checkpoint_sha256_exact':actual_sha==src.get('checkpoint_sha256'),
                'checkpoint_protocol_exact':cp.get('protocol')==c.SOURCE_PROTOCOL,
                'checkpoint_seed_exact':int(cp.get('seed',-1))==seed,
                'checkpoint_variant_exact':cp.get('variant')==variant,
                'checkpoint_epoch_exact':int(cp.get('epoch',-1))==int(src.get('best_epoch',-2)),
                'model_state_dict_present':'model_state_dict' in cp,
            }
            ok=all(checks.values()); passed=passed and ok
            rows[str(seed)][variant]={
                'best_epoch':src.get('best_epoch'),'expected_epoch':c.EXPECTED_EPOCHS[(seed,variant)],
                'best_checkpoint':str(p),'evidence_checkpoint_sha256':src.get('checkpoint_sha256'),'actual_checkpoint_sha256':actual_sha,
                'validation_metrics':src.get('best_evaluation',{}).get('colift'),'source_protocol':src.get('protocol'),'source_protocol_hash':src.get('protocol_hash'),
                'checkpoint_metadata':{'protocol':cp.get('protocol'),'seed':cp.get('seed'),'variant':cp.get('variant'),'epoch':cp.get('epoch'),'TEST_ACCESSED':cp.get('TEST_ACCESSED')},
                'checks':checks,'PASS':ok,
            }
    out={'protocol':c.PROTOCOL,'protocol_hash':c.protocol_hash(),'source_protocol':c.SOURCE_PROTOCOL,'source_protocol_hash':c.SOURCE_PROTOCOL_HASH,'rows':rows,'PASS':bool(passed),'BABY_TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
    c.json_write(c.EVID/'ROUND31_CHECKPOINT_AUDIT.json',out)
    if not passed: raise RuntimeError('Round31 checkpoint integrity audit failed; Test remains closed')
    return out

def write_config():
    val=c.validation_summary()
    out={'protocol':c.PROTOCOL,'protocol_hash':c.protocol_hash(),'objective':'Frozen Round26 Baby Test confirmation: B0 vs D3 only','dataset':'Amazon Baby','source_protocol':c.SOURCE_PROTOCOL,'source_protocol_hash':c.SOURCE_PROTOCOL_HASH,'seeds':list(c.SEEDS),'variants':list(c.VARIANTS),'frozen_round26':{'warmup_epochs':10,'formal_epochs':60,'lambda_HN':0.20,'theta_max_deg':5.0,'half_margin':0.50,'local_t':[1,2,3,4,5],'diff_steps_active':2,'checkpoint_selection':'Round26 frozen best Full CoLiftRec Validation R20'},'expected_epochs':{f'{s}_{v}':c.EXPECTED_EPOCHS[(s,v)] for s in c.SEEDS for v in c.VARIANTS},'validation_mean_U4_D3_vs_B0':val['mean_U_D3_vs_B0'],'test_repeats_per_checkpoint':2,'determinism_tolerance':1e-12,'test_pipeline':'existing TestEvaluator + evaluate_detail from round20/round18; build_test_eval; Top100; Full CoLiftRec; metrics_at','diffusion_inference_calls_expected':0,'test_used_for_selection':False,'no_post_test_tuning':True,'BABY_TEST_OPEN_FOR_ROUND31_ONLY':True,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False,'gpu_policy':'physical GPU0 NVIDIA GeForce RTX 5090'}
    c.json_write(c.EVID/'ROUND31_CONFIG.json',out); return out
