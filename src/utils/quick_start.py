# coding: utf-8
# @email: enoche.chow@gmail.com

"""Run application with validation-only model selection."""
from logging import getLogger
from itertools import product
from utils.dataset import RecDataset
from utils.dataloader import TrainDataLoader, EvalDataLoader
from utils.logger import init_logger
from utils.configurator import Config
from utils.utils import init_seed, get_model, get_trainer, dict2str
import platform
import torch
import os


def quick_start(
    model,
    dataset,
    config_dict,
    save_model=True,
    mg=False,
    init_logging=True,
    log_name=None,
    log_dir=None,
    config_obj=None,
):
    """Train a backbone and select its checkpoint using Validation only.

    Default behavior is backward-compatible with the upstream entrypoint.
    Publication pipeline callers pass init_logging=False and a pre-built
    config_obj so one run uses one logger from start to finish.
    """
    config = config_obj or Config(model, dataset, config_dict, mg)
    log_path = None
    if init_logging:
        log_path = init_logger(config, log_name=log_name, log_dir=log_dir)
    logger = getLogger()
    logger.info('Server: \t' + platform.node())
    logger.info('Dir: \t' + os.getcwd() + '\n')
    logger.info(config)

    dataset_obj = RecDataset(config)
    logger.info(str(dataset_obj))
    train_dataset, valid_dataset, _ = dataset_obj.split()
    logger.info('\n====Training====\n' + str(train_dataset))
    logger.info('\n====Validation====\n' + str(valid_dataset))

    train_data = TrainDataLoader(
        config, train_dataset, batch_size=config['train_batch_size'], shuffle=True
    )
    valid_data = EvalDataLoader(
        config, valid_dataset, additional_dataset=train_dataset,
        batch_size=config['eval_batch_size']
    )

    hyper_ret = []
    idx = 0
    logger.info('\n\n=================================\n\n')

    hyper_ls = []
    if "seed" not in config['hyper_parameters']:
        config['hyper_parameters'] = ['seed'] + config['hyper_parameters']
    for name in config['hyper_parameters']:
        hyper_ls.append(config[name] or [None])

    combinators = list(product(*hyper_ls))
    total_loops = len(combinators)
    for hyper_tuple in combinators:
        for name, value in zip(config['hyper_parameters'], hyper_tuple):
            config[name] = value
        init_seed(config['seed'])
        logger.info('========={}/{}: Parameters:{}={}======='.format(
            idx + 1, total_loops, config['hyper_parameters'], hyper_tuple
        ))

        train_data.pretrain_setup()
        model_obj = get_model(config['model'])(config, train_data).to(config['device'])
        logger.info(model_obj)
        trainer = get_trainer()(config, model_obj, mg)

        best_valid_score, best_valid_result = trainer.fit(
            train_data, valid_data=valid_data, saved=save_model
        )
        checkpoint_state = torch.load(
            trainer.saved_model_file, map_location='cpu', weights_only=False
        )
        hyper_ret.append({
            'params': hyper_tuple,
            'valid_score': best_valid_score,
            'valid_result': best_valid_result,
            'checkpoint': trainer.saved_model_file,
            'best_epoch': int(checkpoint_state['epoch']),
            'checkpoint_best_valid_score': float(checkpoint_state['best_valid_score']),
        })
        idx += 1

        logger.info('best valid result: {}'.format(dict2str(best_valid_result)))
        logger.info('checkpoint: {}'.format(trainer.saved_model_file))
        logger.info('best epoch: {}'.format(hyper_ret[-1]['best_epoch']))
        best_idx = max(range(len(hyper_ret)), key=lambda i: hyper_ret[i]['valid_score'])
        best_entry = hyper_ret[best_idx]
        logger.info(
            'Current BEST by Validation:\nParameters: {}={},\nValid: {},\nCheckpoint: {}\n'.format(
                config['hyper_parameters'], best_entry['params'],
                dict2str(best_entry['valid_result']), best_entry['checkpoint']
            )
        )

    logger.info('\n============All Over=====================')
    for entry in hyper_ret:
        logger.info(
            'Parameters: {}={},\n best valid: {},\n checkpoint: {}'.format(
                config['hyper_parameters'], entry['params'],
                dict2str(entry['valid_result']), entry['checkpoint']
            )
        )

    best_idx = max(range(len(hyper_ret)), key=lambda i: hyper_ret[i]['valid_score'])
    best_entry = hyper_ret[best_idx]
    logger.info('\n\n BEST BY VALIDATION ')
    logger.info(
        '\tParameters: {}={},\nValid: {},\nCheckpoint: {}\n'.format(
            config['hyper_parameters'], best_entry['params'],
            dict2str(best_entry['valid_result']), best_entry['checkpoint']
        )
    )
    if log_path:
        best_entry['log_path'] = log_path
    return best_entry
