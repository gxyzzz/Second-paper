# coding: utf-8

import os
import argparse
import torch
from utils.dataset import RecDataset
from utils.topk_evaluator import TopKEvaluator
from utils.dataloader import TrainDataLoader, EvalDataLoader
from utils.utils import get_model, dict2str
os.environ['NUMEXPR_MAX_THREADS'] = '48'


# Modified from `quick_start.py` and `trainer.py`
def quick_test(model_name, dataset_name, checkpoint_path):
    trained_checkpoint = torch.load(checkpoint_path, map_location="cpu") 
    config = trained_checkpoint['config']
    assert model_name == config['model'] and dataset_name == config['dataset'], "Loading wrong checkpoint!"
    # load data
    dataset = RecDataset(config)
    train_dataset, valid_dataset, test_dataset = dataset.split()
    print('\n====Training====\n' + str(train_dataset))
    print('\n====Validation====\n' + str(valid_dataset))
    print('\n====Testing====\n' + str(test_dataset))
    # wrap into dataloader
    train_data = TrainDataLoader(config, train_dataset, batch_size=config['train_batch_size'], shuffle=True)
    (valid_data, test_data) = (
        EvalDataLoader(config, valid_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']),
        EvalDataLoader(config, test_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']))
    # set random state of dataloader
    train_data.pretrain_setup()
    # model loading and initialization
    model = get_model(config['model'])(config, train_data).to(config['device'])
    model.load_state_dict(trained_checkpoint["state_dict"], strict=False)

    evaluator = TopKEvaluator(config)
    model.eval()
    # batch full users
    batch_matrix_list = []
    for batch_idx, batched_data in enumerate(test_data):
        # predict: interaction without item ids
        scores = model.full_sort_predict(batched_data)
        masked_items = batched_data[1]
        # mask out pos items
        scores[masked_items[0], masked_items[1]] = -1e10
        # rank and get top-k
        _, topk_index = torch.topk(scores, max(config['topk']), dim=-1)  # nusers x topk
        batch_matrix_list.append(topk_index)
    test_result = evaluator.evaluate(batch_matrix_list, test_data, is_test=True, idx=0)
    print('test result: {}'.format(dict2str(test_result)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='MSCA', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='baby', help='name of datasets')
    parser.add_argument('--checkpoint_path', '-c', type=str, default='saved/MSCA-baby.pth', help='path of checkpoints')

    args, _ = parser.parse_known_args()
    
    quick_test(model_name=args.model, dataset_name=args.dataset, checkpoint_path=args.checkpoint_path)