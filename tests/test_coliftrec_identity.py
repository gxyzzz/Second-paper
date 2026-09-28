from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

from modules.coliftrec import CoLiftConfig, score_coliftrec
from modules.ranking import rank_by_score

class CoLiftIdentityTest(unittest.TestCase):
    def test_alpha_zero_preserves_ranking(self):
        scores=np.array([[4.,3.,2.,1.],[1.,4.,3.,2.]],dtype=np.float32)
        items=np.array([[0,1,2,3],[4,5,6,7]],dtype=np.int32)
        z=np.arange(8,dtype=np.float32).reshape(2,4)
        bg={m:{"shrunk_mean":np.zeros(8,dtype=np.float32),"count":np.zeros(8,dtype=np.int64),"global_mean":0.0}
            for m in ("text","attribute","visual")}
        cfg=CoLiftConfig(1.0,.75,.25,0.0,0.0,0.0)
        out,_=score_coliftrec(scores,items,z,z,z,bg,cfg,enabled={"text":True,"attribute":True,"visual":True})
        self.assertTrue(np.array_equal(rank_by_score(items,out),rank_by_score(items,scores)))

if __name__=="__main__":
    unittest.main()
