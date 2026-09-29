from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipelines.search_test_eval import evaluate_frozen_recommendation_test


def main():
    parser = argparse.ArgumentParser(
        description="One-time Test evaluation of a Validation-frozen search recommendation"
    )
    parser.add_argument("--search-dir", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--out-dir")
    parser.add_argument("--skip-if-no-stable", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%d %b %H:%M:%S",
    )
    result = evaluate_frozen_recommendation_test(
        Path(args.search_dir),
        gpu=args.gpu,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        logger=logging.getLogger("search_final_test"),
        skip_if_no_stable=args.skip_if_no_stable,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
