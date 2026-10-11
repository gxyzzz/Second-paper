from __future__ import annotations
import argparse, json
from diffusion_experiments.round30_candidate_side_diffusion_purification import round30_core as c
from diffusion_experiments.round30_candidate_side_diffusion_purification.round30_colift_reference import build_reference_audit
from diffusion_experiments.round30_candidate_side_diffusion_purification.round30_audits import anchor_audit,q_audit,target_stability,smoke_and_preflight,formal_validation,finalize_round30,write_config
from diffusion_experiments.round30_candidate_side_diffusion_purification.round30_train_diffusion import train_diffusion

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('mode',choices=['reference','anchor','q','stability','preflight','formal','validate','finalize','config']); ap.add_argument('--seed',type=int,default=999); a=ap.parse_args()
    if a.mode=='reference': out=build_reference_audit(a.seed)
    elif a.mode=='anchor': out=anchor_audit(a.seed)
    elif a.mode=='q': out=q_audit(a.seed)
    elif a.mode=='stability': out=target_stability(a.seed)
    elif a.mode=='preflight': out=smoke_and_preflight()
    elif a.mode=='formal': out=train_diffusion(a.seed,epochs=c.FORMAL_EPOCHS,formal=True)
    elif a.mode=='validate': out=formal_validation(a.seed)
    elif a.mode=='finalize': out=finalize_round30()
    else: out=write_config()
    print(json.dumps(out,indent=2,default=str))
if __name__=='__main__': main()
