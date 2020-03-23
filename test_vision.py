#!/usr/bin/env python3

import os
import json
import torch

import learn2learn as l2l

from utils import *
from maml_vision import evaluate

cuda = True

base_path = "results/maml_min_23_03_16h08_1_1215"


def run(path):
    # Initialize
    with open(path + "/logger.json", "r") as f:
        params = json.load(f)['config']

    device = torch.device('cpu')
    if cuda and torch.cuda.device_count():
        torch.cuda.manual_seed(params['seed'])
        device = torch.device('cuda')

    if "min" in path:
        _, _, test_tasks = get_mini_imagenet(params['ways'], params['shots'])
        model = l2l.vision.models.MiniImagenetCNN(params['ways'])
    else:
        _, _, test_tasks = get_omniglot(params['ways'], params['shots'])
        if params['model_type'] == 'omni_CNN':
            model = l2l.vision.models.OmniglotCNN(params['ways'])
        else:
            model = l2l.vision.models.OmniglotFC(28 ** 2, params['ways'])

    ckpnt = base_path + "/model_checkpoints/"
    model_ckpnt_results = {}

    for model_ckpnt in os.scandir(ckpnt):
        if model_ckpnt.path.endswith(".pt"):
            print(f'Testing {model_ckpnt.path}')
            res = evaluate_model(params, model, test_tasks, device, model_ckpnt.path)
            model_ckpnt_results[model_ckpnt.path] = res

    with open(base_path + '/ckpnt_results.json', 'w') as fp:
        json.dump(model_ckpnt_results, fp, sort_keys=True, indent=4)


def evaluate_model(params, model, test_tasks, device, path):

    model.load_state_dict(torch.load(path))
    model.to(device)
    maml = l2l.algorithms.MAML(model, lr=params['fast_lr'], first_order=False)
    loss = torch.nn.CrossEntropyLoss(reduction='mean')

    return evaluate(params, test_tasks, maml, loss, device)


if __name__ == '__main__':
    run(base_path)

