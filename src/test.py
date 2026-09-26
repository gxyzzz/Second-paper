# coding: utf-8

import os
import argparse
import torch
from utils.dataset import RecDataset
from utils.topk_evaluator import TopKEvaluator
from utils.dataloader import TrainDataLoader, EvalDataLoader
from utils.utils import get_model, dict2str
os.environ['NUMEXPR_MAX_THREADS'] = '48'


def quick_test(model_name, dataset_name, checkpoint_path):
    trained_checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    config = trained_checkpoint['config']
    assert model_name == config['model'] and dataset_name == config['dataset'], (
        "Loading wrong checkpoint!"
    )

    dataset_obj = RecDataset(config)
    train_dataset, _, test_dataset = dataset_obj.split()
    print('\n====Training reference====\n' + str(train_dataset))
    print('\n====Testing====\n' + str(test_dataset))

    train_data = TrainDataLoader(
        config, train_dataset, batch_size=config['train_batch_size'], shuffle=True
    )
    test_data = EvalDataLoader(
        config, test_dataset, additional_dataset=train_dataset,
        batch_size=config['eval_batch_size']
    )

    train_data.pretrain_setup()
    model = get_model(config['model'])(config, train_data).to(config['device'])
    incompatible = model.load_state_dict(
        trained_checkpoint["state_dict"], strict=True
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Strict checkpoint load reported incompatible keys: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )

    evaluator = TopKEvaluator(config)
    model.eval()
    batch_matrix_list = []
    with torch.no_grad():
        for batched_data in test_data:
            scores = model.full_sort_predict(batched_data)
            masked_items = batched_data[1]
            scores[masked_items[0], masked_items[1]] = -1e10
            _, topk_index = torch.topk(
                scores, max(config['topk']), dim=-1
            )
            batch_matrix_list.append(topk_index)

    test_result = evaluator.evaluate(
        batch_matrix_list, test_data, is_test=True, idx=0
    )
    print('checkpoint epoch: {}'.format(trained_checkpoint['epoch']))
    print('checkpoint best_valid_score: {}'.format(
        trained_checkpoint['best_valid_score']
    ))
    print('test result: {}'.format(dict2str(test_result)))
    return test_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='MSCA')
    parser.add_argument('--dataset', '-d', type=str, default='baby')
    parser.add_argument('--checkpoint_path', '-c', type=str, required=True)
    args, _ = parser.parse_known_args()
    quick_test(
        model_name=args.model,
        dataset_name=args.dataset,
        checkpoint_path=args.checkpoint_path,
    )
