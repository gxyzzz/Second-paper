#!/usr/bin/env python3
from pathlib import Path
import ast
root = Path(__file__).resolve().parents[1]
trainer_path = root / 'src/common/trainer.py'
quick_path = root / 'src/utils/quick_start.py'
trainer_src = trainer_path.read_text(encoding='utf-8')
quick_src = quick_path.read_text(encoding='utf-8')
tree = ast.parse(trainer_src)
trainer_cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Trainer')
fit = next(n for n in trainer_cls.body if isinstance(n, ast.FunctionDef) and n.name == 'fit')
args = [a.arg for a in fit.args.args]
fit_text = ast.get_source_segment(trainer_src, fit) or ''
assert 'test_data' not in args
assert 'test_data' not in fit_text
assert 'best_test' not in fit_text
assert 'best_test' not in quick_src
assert 'test_data' not in quick_src
assert '_valid_epoch(test' not in quick_src
print('TRAINING_TEST_ISOLATION = PASS')
