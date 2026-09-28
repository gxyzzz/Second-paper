from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import main


class MainStageCliTest(unittest.TestCase):
    def test_default_stage_is_msca(self):
        args, _ = main.build_parser().parse_known_args(['-m', 'MSCA', '-d', 'baby'])
        self.assertEqual(args.stage, 'msca')

    def test_stage_choices(self):
        for stage in ('msca', 'coliftrec', 'full'):
            args, _ = main.build_parser().parse_known_args(['-m', 'MSCA', '-d', 'baby', '--stage', stage])
            self.assertEqual(args.stage, stage)


if __name__ == '__main__':
    unittest.main()
