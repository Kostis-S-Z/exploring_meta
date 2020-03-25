#!/usr/bin/env python3

import os
import json
import torch

import learn2learn as l2l

from utils import *
from core_funtions.vision import evaluate

cuda = True

base_path = "results/reprod_results/maml_min_04_03_16h43_42_2373"

eval_iters = False
cl_exp = False
rep_exp = True

cl_params = {
    "adapt_steps": 1,
    "inner_lr": 0.1,
    "n_tasks": 10
}

rep_params = {
    "adapt_steps": 1,
    "inner_lr": 0.1,
    "n_tasks": 2,
    "layers": [3, 4]
}


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

    # Evaluate the model at every checkpoint
    if eval_iters:
        ckpnt = base_path + "/model_checkpoints/"
        model_ckpnt_results = {}
        for model_ckpnt in os.scandir(ckpnt):
            if model_ckpnt.path.endswith(".pt"):
                print(f'Testing {model_ckpnt.path}')
                res = evaluate_model(params, model, test_tasks, device, model_ckpnt.path)
                model_ckpnt_results[model_ckpnt.path] = res

        with open(base_path + '/ckpnt_results.json', 'w') as fp:
            json.dump(model_ckpnt_results, fp, sort_keys=True, indent=4)

    final_model = base_path + '/model.pt'
    # Run a Continual Learning experiment
    if cl_exp:
        print("Running Continual Learning experiment...")
        model.load_state_dict(torch.load(final_model))
        model.to(device)
        maml = l2l.algorithms.MAML(model, lr=cl_params['inner_lr'], first_order=False)
        loss = torch.nn.CrossEntropyLoss(reduction='mean')

        run_cl_exp(base_path, maml, loss, test_tasks, device,
                   params['ways'], params['shots'], cl_params=cl_params)

    # Run a Representation change experiment
    if rep_exp:
        model.load_state_dict(torch.load(final_model))
        model.to(device)
        maml = l2l.algorithms.MAML(model, lr=rep_params['inner_lr'], first_order=False)
        loss = torch.nn.CrossEntropyLoss(reduction='mean')

        print("Running Representation experiment...")
        run_rep_exp(base_path, maml, loss, test_tasks, device,
                    params['ways'], params['shots'], rep_params=rep_params)


def evaluate_model(params, model, test_tasks, device, path):
    model.load_state_dict(torch.load(path))
    model.to(device)
    maml = l2l.algorithms.MAML(model, lr=params['inner_lr'], first_order=False)
    loss = torch.nn.CrossEntropyLoss(reduction='mean')

    return evaluate(params, test_tasks, maml, loss, device)


if __name__ == '__main__':
    run(base_path)
