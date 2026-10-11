from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np

ROOT=Path('/home/gxy/code/Second-paper')
RDIR=ROOT/'diffusion_experiments/round31_round26_baby_test_confirmation'
EVID=RDIR/'evidence'; LOGS=RDIR/'logs'
for p in (EVID,LOGS): p.mkdir(parents=True,exist_ok=True)
R26=ROOT/'diffusion_experiments/round26_preference_directed_diffusion'
R26_EVID=R26/'evidence'
PROTOCOL='ROUND31_ROUND26_FROZEN_BABY_TEST_CONFIRMATION_V1'
SOURCE_PROTOCOL='ROUND26_PREFERENCE_DIRECTED_DIFFUSION_V1'
SOURCE_PROTOCOL_HASH='025c86c326d5877cb6b0fd91cd03efa85ea01c1dc4b8bbe04b1e5ebc3ce77939'
SEEDS=(999,1000); VARIANTS=('B0','D3')
EXPECTED_EPOCHS={(999,'B0'):39,(999,'D3'):24,(1000,'B0'):31,(1000,'D3'):22}
ALL=('R10','N10','R20','N20','R50','N50'); PRIMARY=('R10','N10','R20','N20')

def json_write(path,obj): Path(path).write_text(json.dumps(obj,indent=2,default=lambda x:float(x) if isinstance(x,np.generic) else str(x))+'\n')
def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def source_result(seed,variant): return json.loads((R26_EVID/f'ROUND26_{variant}_SEED{seed}.json').read_text())
def validation_summary(): return json.loads((R26_EVID/'ROUND26_VALIDATION_SUMMARY.json').read_text())
def delta_pack(m,b):
    absd={k:float(m[k]-b[k]) for k in ALL}; rel={k:float((m[k]-b[k])/b[k]) for k in ALL}
    return {'absolute_delta':absd,'relative_delta':rel,'U4':float(np.mean([rel[k] for k in PRIMARY])),'primary_positive_count':int(sum(m[k]>b[k] for k in PRIMARY)),'overall_positive_count':int(sum(m[k]>b[k] for k in ALL))}
def protocol_hash():
    h=hashlib.sha256()
    for name in ('round31_core.py','round31_test_eval.py','round31_audits.py','run_round31.py'):
        p=RDIR/name
        if p.exists(): h.update(name.encode()); h.update(p.read_bytes())
    h.update(json.dumps({'source_protocol':SOURCE_PROTOCOL,'seeds':SEEDS,'variants':VARIANTS,'test_repeats':2,'candidate_topk':100,'checkpoint_selection':'frozen Round26 Validation best checkpoint','diffusion_inference_calls':0,'sports':False,'electronics':False},sort_keys=True).encode())
    return h.hexdigest()
