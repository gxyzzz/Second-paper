from pathlib import Path
import math
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from modules.ranking import metrics_at, rank_by_score

class RankingMetricsTest(unittest.TestCase):
    def test_stable_ranking_and_metrics(self):
        items=np.array([[10,11,12],[20,21,22]],dtype=np.int32)
        scores=np.array([[.2,.9,.1],[.5,.5,.1]],dtype=np.float32)
        ranked=rank_by_score(items,scores)
        self.assertTrue(np.array_equal(ranked,np.array([[11,10,12],[20,21,22]],dtype=np.int32)))
        users=np.array([0,1],dtype=np.int64)
        sets={0:{11},1:{21}}
        m=metrics_at(ranked,users,sets,ks=(1,2))
        self.assertEqual(m["R1"],.5)
        self.assertEqual(m["R2"],1.0)
        self.assertAlmostEqual(m["N1"],.5,places=12)
        self.assertAlmostEqual(m["N2"],(.5*(1.0+1.0/math.log2(3))),places=12)

if __name__=="__main__":
    unittest.main()
