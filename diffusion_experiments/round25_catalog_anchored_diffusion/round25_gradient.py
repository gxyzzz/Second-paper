from __future__ import annotations
import gc
import torch
from diffusion_experiments.round25_catalog_anchored_diffusion import round25_core as c

def run():
    model,config,td,vd,pack=c.instantiate_shared(999)
    events=c.TrainEvents(model); mask=c.valid_mask(td,model.n_items,model.device)
    inter=next(iter(td)); one=[x[:4] for x in inter]
    b=c.forward_bundle(model); ids,_,_=c.mine_current_ids(b,one,events,mask,999,10,0,1)
    loss=c.aux_bpr(b['fu'][one[0]],b['fi'][one[1]],[b['fi'][ids[0]]])
    gfu,gfi=torch.autograd.grad(loss,(b['fu'],b['fi']))
    c1={'user_grad_norm':float(gfu[one[0]].norm()),'positive_item_grad_norm':float(gfi[one[1]].norm()),'selected_real_negative_grad_norm':float(gfi[ids[0]].norm())}
    net,dopt,sched=c.make_diff(999,model.device); s=c.latent_stats(model); b=c.forward_bundle(model)
    c.diffusion_inner_step(net,dopt,sched,b,one,s,ids,20286000)
    deltas,_=c.generate_refinement(net,sched,b,one,s,ids,20286001)
    delta=deltas[0]; final=b['fi'][ids[0]]+delta
    loss2=c.aux_bpr(b['fu'][one[0]],b['fi'][one[1]],[final])
    gfu2,gfi2=torch.autograd.grad(loss2,(b['fu'],b['fi']))
    d2={'user_grad_norm':float(gfu2[one[0]].norm()),'positive_item_grad_norm':float(gfi2[one[1]].norm()),'selected_real_negative_grad_norm':float(gfi2[ids[0]].norm()),'Delta_diff_requires_grad':bool(delta.requires_grad),'identity_path_verified':bool((not delta.requires_grad) and float(gfi2[ids[0]].norm())>0)}
    out={'protocol':c.PROTOCOL,'C1FULL':c1,'D2':d2,'PASS':bool(c1['selected_real_negative_grad_norm']>0 and d2['selected_real_negative_grad_norm']>0 and not delta.requires_grad and d2['user_grad_norm']>0 and d2['positive_item_grad_norm']>0),'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
    c.json_write(c.EVID/'ROUND25_GRADIENT_AUDIT.json',out)
    del model,net,dopt; torch.cuda.empty_cache(); gc.collect(); return out

if __name__=='__main__':
    import json; print(json.dumps(run(),indent=2))
