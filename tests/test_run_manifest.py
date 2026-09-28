from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pipelines.runner import run_pipeline


class RunManifestTest(unittest.TestCase):
    def test_dry_run_creates_manifest_and_resolved_config(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / 'manifest_run'
            result = run_pipeline(
                model='MSCA', dataset='baby', stage='full', gpu_id=0,
                run_dir=run_dir, dry_run=True,
            )
            manifest = json.loads((run_dir / 'run_manifest.json').read_text())
            resolved = yaml.safe_load((run_dir / 'resolved_config.yaml').read_text())
            self.assertEqual(manifest['status'], 'DRY_RUN')
            self.assertFalse(manifest['TEST_ACCESSED'])
            self.assertEqual(resolved['stage'], 'full')
            self.assertEqual(resolved['publication_method']['backbone']['seed'], 999)
            self.assertEqual(
                resolved['publication_method']['datasets']['baby']['diffusion']['rho_text'],
                0.25,
            )
            self.assertTrue(Path(result['log_path']).is_file())

    def test_method_yaml_value_controls_resolved_runtime(self):
        import pipelines.dataset_config as dataset_config
        source = yaml.safe_load(dataset_config.METHOD_CONFIG_PATH.read_text())
        source['datasets']['baby']['diffusion']['rho_text'] = 0.50
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            method_copy = td / 'method.yaml'
            method_copy.write_text(yaml.safe_dump(source, sort_keys=False))
            run_dir = td / 'yaml_control_run'
            with patch.object(dataset_config, 'METHOD_CONFIG_PATH', method_copy):
                result = run_pipeline(
                    model='MSCA', dataset='baby', stage='full', gpu_id=0,
                    run_dir=run_dir, dry_run=True,
                )
            resolved = yaml.safe_load((run_dir / 'resolved_config.yaml').read_text())
            self.assertEqual(
                resolved['publication_method']['datasets']['baby']['diffusion']['rho_text'],
                0.50,
            )
            log_text = Path(result['log_path']).read_text(encoding='utf-8')
            self.assertIn('rho_T=0.5', log_text)


if __name__ == '__main__':
    unittest.main()
