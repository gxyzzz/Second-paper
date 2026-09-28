from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from modules.diffusion import blend_block

class DiffusionIdentityTest(unittest.TestCase):
    def test_rho_zero_is_exact_raw(self):
        raw=np.array([[3.,4.],[1.,2.]],dtype=np.float32)
        diff=np.array([[9.,8.],[7.,6.]],dtype=np.float32)
        out=blend_block(raw,diff,0.0)
        self.assertTrue(np.array_equal(out,raw))

if __name__=="__main__":
    unittest.main()
