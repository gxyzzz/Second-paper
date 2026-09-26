# coding: utf-8
# @email: enoche.chow@gmail.com

"""
Main entry
# UPDATED: 2022-Feb-15
##########################
"""

import os
import argparse
from utils.quick_start import quick_start
os.environ['NUMEXPR_MAX_THREADS'] = '48'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='MSCA', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='baby', help='name of datasets')
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--stopping-step', type=int)
    parser.add_argument('--train-batch-size', type=int)

    args, _ = parser.parse_known_args()
    config_dict = {'gpu_id': args.gpu_id}
    if args.epochs is not None:
        config_dict['epochs'] = args.epochs
    if args.stopping_step is not None:
        config_dict['stopping_step'] = args.stopping_step
    if args.train_batch_size is not None:
        config_dict['train_batch_size'] = args.train_batch_size

    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True)