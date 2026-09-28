# coding: utf-8
# @email: enoche.chow@gmail.com

"""Unified MMRec/MSCA-style training, Validation selection, and formal Test entrypoint."""

import argparse
import os

from pipelines.runner import run_pipeline

os.environ['NUMEXPR_MAX_THREADS'] = '48'


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='MSCA', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='baby', help='name of datasets')
    parser.add_argument(
        '--stage', choices=['msca', 'coliftrec', 'full'], default='msca',
        help='pipeline depth; default preserves the upstream MSCA training entrypoint',
    )
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--stopping-step', type=int)
    parser.add_argument('--train-batch-size', type=int)
    parser.add_argument('--run-dir')
    parser.add_argument('--checkpoint')
    parser.add_argument(
        '--smoke', action='store_true',
        help='engineering smoke only; never treated as publication evidence',
    )
    parser.add_argument('--dry-run', action='store_true')
    return parser


def main(argv=None):
    args, _ = build_parser().parse_known_args(argv)
    overrides = {}
    if args.epochs is not None:
        overrides['epochs'] = args.epochs
    if args.stopping_step is not None:
        overrides['stopping_step'] = args.stopping_step
    if args.train_batch_size is not None:
        overrides['train_batch_size'] = args.train_batch_size

    return run_pipeline(
        model=args.model,
        dataset=args.dataset,
        stage=args.stage,
        gpu_id=args.gpu_id,
        run_dir=args.run_dir,
        checkpoint=args.checkpoint,
        msca_overrides=overrides,
        smoke=args.smoke,
        dry_run=args.dry_run,
    )


if __name__ == '__main__':
    main()
