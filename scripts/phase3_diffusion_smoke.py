from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from modules.diffusion import NativeTVX0Denoiser, cosine_alpha_bar, training_loss, purify_indices
from modules.semantic_purifier import condition_beta, joint_native_state, blend_native

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/phase3_diffusion/smoke'
OUT.mkdir(parents=True,exist_ok=True)

emb=np.load(ROOT/'runs/assets/msca_baby_seed999/embeddings.npz')
collab=emb['collab_item'].astype(np.float32)
final=emb['final_item'].astype(np.float32)
conds={str(b):condition_beta(collab,final,b) for b in (0.0,0.5,1.0)}
for b,c in conds.items():
    assert c.shape==(7050,64) and np.isfinite(c).all()
assert np.allclose(conds['0.0'], condition_beta(collab,collab,0.0))
assert np.allclose(conds['1.0'], condition_beta(final,final,0.0))

df=pd.read_csv(ROOT/'data/baby/baby.inter',sep='\t',usecols=['itemID','x_label'])
train_ids=np.sort(df.loc[df.x_label==0,'itemID'].unique().astype(np.int64))
assert len(train_ids)==7047
assert set(train_ids.tolist()) == set(df.loc[df.x_label==0,'itemID'].astype(int).tolist())
text=np.load(ROOT/'data/baby/text_feat.npy',mmap_mode='r',allow_pickle=False)
visual=np.load(ROOT/'data/baby/image_feat.npy',mmap_mode='r',allow_pickle=False)
assert text.shape==(7050,384) and visual.shape==(7050,4096)
ids=train_ids[:8]
x_np=joint_native_state(text[ids],visual[ids])
assert x_np.shape==(8,4480) and np.isfinite(x_np).all()

device='cuda'
torch.manual_seed(999)
model=NativeTVX0Denoiser(4480,cond_dim=64,hidden=1024,time_dim=64).to(device)
x=torch.from_numpy(x_np).to(device)
c=torch.from_numpy(conds['0.5'][ids]).to(device)
t=torch.tensor([1,2,3,4,5,6,7,8],device=device)
noise=torch.randn_like(x)
ab=cosine_alpha_bar(50).to(device)
loss,detail=training_loss(model,x,c,t,noise,ab,lambda_ctr=0.1,p_uncond=0.15)
assert torch.isfinite(loss)
loss.backward()
grad_finite=all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
assert grad_finite
opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
opt.step(); opt.zero_grad(set_to_none=True)

ck=OUT/'smoke_checkpoint.pt'
torch.save({'state_dict':model.state_dict(),'architecture':[4480,1024,512,1024,4480]},ck)
reload_model=NativeTVX0Denoiser(4480,cond_dim=64,hidden=1024,time_dim=64).to(device)
state=torch.load(ck,map_location=device,weights_only=False)
reload_model.load_state_dict(state['state_dict'],strict=True)
reload_model.eval()
pt,pv=purify_indices(reload_model,text,visual,conds['0.5'],ids,t_edit=1,guidance=1.0,seeds=(20261001,),batch=8,device=device)
assert pt.shape==(8,384) and pv.shape==(8,4096)
assert np.isfinite(pt).all() and np.isfinite(pv).all()

raw_t=np.asarray(text)
raw_v=np.asarray(visual)
idt,idv=blend_native(raw_t,raw_v,np.zeros_like(raw_t,dtype=np.float32),np.zeros_like(raw_v,dtype=np.float32),0.0,0.0)
assert np.array_equal(idt,raw_t)
assert np.array_equal(idv,raw_v)
np.save(OUT/'rho0_text.npy',idt,allow_pickle=False)
np.save(OUT/'rho0_visual.npy',idv,allow_pickle=False)

summary={
 'phase':'PHASE3_DIFFUSION_INTEGRATION_SMOKE',
 'architecture':[4480,1024,512,1024,4480],
 'prediction':'x0','schedule':'cosine','diffusion_steps':50,'time_embedding_dim':64,
 'lambda_ctr':0.1,'p_uncond':0.15,'optimizer':'AdamW','lr':1e-3,'weight_decay':1e-4,
 'condition_formula':'Norm[c_collab + beta*(c_final-c_collab)]',
 'condition_shapes':{b:list(c.shape) for b,c in conds.items()},
 'condition_beta_range':[0.0,1.0],
 'train_item_count':int(len(train_ids)),
 'TRAIN_ONLY_ITEM_AUDIT':'PASS',
 'small_batch_forward':'PASS','small_batch_backward':'PASS',
 'loss':float(loss.detach().cpu()),'loss_detail':detail,
 'checkpoint_save_load':'PASS',
 'purified_text_shape':list(pt.shape),'purified_visual_shape':list(pv.shape),
 'finite_check':'PASS','rho0_feature_identity':'PASS',
 'DIFFUSION_FORMAL_TRAINING':'NOT_STARTED'
}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,sort_keys=True))
