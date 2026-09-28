from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
sys.path.insert(0,str(ROOT/"scripts"))

from pipelines.dataset_config import load_dataset_config
from pipelines.publication_eval import evaluate_test, evaluate_validation, generate_fixed_purified
import reproduce

class PipelineSmokeTest(unittest.TestCase):
    def test_three_domain_runtime_config(self):
        for ds in ("baby","sports","elec"):
            cfg=load_dataset_config(ds)
            self.assertEqual(cfg["backbone"]["seed"],999)
            self.assertTrue(Path(cfg["_method_config_path"]).is_file())
            self.assertNotIn("experiments/archive",cfg["_method_config_path"])
            plan=reproduce.workspace_paths(ds)
            self.assertEqual(plan["base"].name,ds)

    def test_runtime_functions_import(self):
        self.assertTrue(callable(evaluate_test))
        self.assertTrue(callable(evaluate_validation))
        self.assertTrue(callable(generate_fixed_purified))

if __name__=="__main__":
    unittest.main()
