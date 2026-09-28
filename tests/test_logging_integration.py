from pathlib import Path
import logging
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from utils.logger import init_logger


class LoggingIntegrationTest(unittest.TestCase):
    def test_one_console_one_file_handler(self):
        cfg = {'model': 'MSCA', 'dataset': 'baby', 'state': None}
        with tempfile.TemporaryDirectory() as td:
            path = init_logger(cfg, log_name='unit-log', log_dir=td, reset=True)
            init_logger(cfg, log_name='unit-log-second', log_dir=td, reset=True)
            root = logging.getLogger()
            self.assertEqual(len(root.handlers), 2)
            self.assertEqual(sum(isinstance(h, logging.FileHandler) for h in root.handlers), 1)
            root.info('LOGGER_INTEGRATION_PASS')
            for h in root.handlers:
                h.flush()
            active = getattr(root, '_second_paper_log_path')
            self.assertTrue(Path(active).is_file())
            self.assertIn('LOGGER_INTEGRATION_PASS', Path(active).read_text(encoding='utf-8'))
            self.assertTrue(Path(path).parent.samefile(td))


if __name__ == '__main__':
    unittest.main()
