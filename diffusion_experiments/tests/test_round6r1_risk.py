import numpy as np,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from diffusion_experiments.modules.round6r1_risk import degree_match_mask,raw_dot_risk_for_users,inverse_standardized

def run():
    items=np.array([37,205,900],np.int32); degree=np.zeros(1000,np.int64); degree[37]=10; degree[205]=12; degree[900]=100; target=37
    m=degree_match_mask(items,np.array([1,2]),target,degree,0.5); assert m.tolist()==[True,False]
    perm=np.array([2,0,1]); items2=items[perm]; negpos=np.array([int(np.flatnonzero(items2==205)[0]),int(np.flatnonzero(items2==900)[0])]); m2=degree_match_mask(items2,negpos,target,degree,0.5); assert m2.tolist()==[True,False]
    mu=np.array([1.,2.],np.float32); sd=np.array([2.,4.],np.float32); valid=np.array([True,True]); z=np.array([[[1.,-1.],[3.,.5]]],np.float32); raw=inverse_standardized(z,mu,sd,valid); assert np.allclose(raw,np.array([[[3.,-2.],[7.,4.]]],np.float32))
    raw_cf=np.array([[1.,0.],[0.,1.],[1.,1.]],np.float32); ids=np.array([[0,1,2]],np.int32); A=np.ones((1,3),bool); r=raw_dot_risk_for_users(z,np.array([0]),ids,A,raw_cf,mu,sd,valid); assert r['valid_query'][0]; assert np.allclose(r['weights'][0],(1-r['rho'][0])**2); assert r['weights'][0,np.nanargmax(r['rho'][0])]==0
    r2=raw_dot_risk_for_users(z,np.array([0]),ids,np.array([[1,0,0]],bool),raw_cf,mu,sd,valid); assert not r2['valid_query'][0] and np.isnan(r2['weights']).all()
    print('ROUND6R1_RISK_TEST_PASS')
if __name__=='__main__': run()
