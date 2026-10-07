from __future__ import annotations
import argparse,json,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from diffusion_experiments.modules.round6r1_train_common import cfg_r1,make_base_plan,load_edges,sha

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--seed',type=int,required=True); ap.add_argument('--out',required=True); ap.add_argument('--mode',choices=['smoke','formal'],default='formal'); a=ap.parse_args(); out=Path(a.out)
    if out.exists() and any(out.iterdir()): raise RuntimeError('refuse overwrite nonempty output')
    out.mkdir(parents=True,exist_ok=True); c=cfg_r1(); pdx=ROOT/c['protocol_dir']; fit=load_edges(pdx/'fit_edges.csv'); epochs=2 if a.mode=='smoke' else int(c['training']['max_epochs']); t=time.time(); U,P,N=make_base_plan(a.seed,fit,out,epochs); np.savez_compressed(out/'base_plan.npz',users=U,pos=P,neg=N)
    r={'status':'COMPLETE','mode':a.mode,'seed':a.seed,'epochs':epochs,'events_per_epoch':int(U.shape[1]),'plan_sha256':sha(out/'base_plan.npz'),'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'elapsed_seconds':time.time()-t,'access':{'DEV':False,'INTERNAL':False,'CONFIRM':False,'TEST':False}}
    (out/'result.json').write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r,sort_keys=True))
if __name__=='__main__': main()
