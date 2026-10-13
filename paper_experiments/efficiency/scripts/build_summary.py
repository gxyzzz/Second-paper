from __future__ import annotations
import csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
def main():
    ev=ROOT/'paper_experiments/efficiency/evidence'
    params=json.loads((ev/'PARAMETER_COUNT.json').read_text()) if (ev/'PARAMETER_COUNT.json').exists() else {'datasets':{}}
    train=list(csv.DictReader((ev/'TRAINING_BENCHMARK.csv').open())) if (ev/'TRAINING_BENCHMARK.csv').exists() else []
    infer=list(csv.DictReader((ev/'INFERENCE_BENCHMARK.csv').open())) if (ev/'INFERENCE_BENCHMARK.csv').exists() else []
    rows=[]
    for d in ['baby','sports','elec']:
        p=params['datasets'].get(d,{})
        for method in ['MSCA','MSCA+CoLiftRec','MSCA+CoLiftRec+PDBR']:
            t=next((x for x in train if x['dataset']==d and x['method']==method),{})
            q=next((x for x in infer if x['dataset']==d and x['method']==method),{})
            if method=='MSCA': trainp=p.get('MSCA_total_training_parameters'); deploy=p.get('MSCA_deployed_inference_parameters')
            elif method=='MSCA+CoLiftRec': trainp=p.get('MSCA_plus_CoLiftRec_total_training_parameters'); deploy=p.get('MSCA_plus_CoLiftRec_deployed_inference_parameters')
            else: trainp=p.get('MSCA_plus_CoLiftRec_plus_PDBR_total_training_parameters'); deploy=p.get('MSCA_plus_CoLiftRec_plus_PDBR_deployed_inference_parameters')
            rows.append({'dataset':d,'method':method,'train_params':trainp,'deploy_params':deploy,'train_ms_per_batch':t.get('ms_per_batch'),'peak_vram_GB':t.get('peak_cuda_allocated_GB'),'inference_ms_per_user':q.get('median_ms_per_user'),'diffusion_inference_calls':q.get('diffusion_inference_calls','0')})
    fields=['dataset','method','train_params','deploy_params','train_ms_per_batch','peak_vram_GB','inference_ms_per_user','diffusion_inference_calls']
    with (ev/'EFFICIENCY_SUMMARY.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    print('wrote',ev/'EFFICIENCY_SUMMARY.csv')
if __name__=='__main__':main()
