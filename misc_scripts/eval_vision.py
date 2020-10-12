#!/usr/bin/env python3

import os
import json
import torch

import learn2learn as l2l

from utils import get_mini_imagenet, get_omniglot
from core_functions.vision import evaluate, OmniglotCNN, MiniImagenetCNN, ConvBase
from misc_scripts import run_cl_exp, run_rep_exp

cuda = True

base_path = "/home/kosz/Projects/KTH/Thesis/exploring_meta/vision/results/anil_20w1s_omni_06_09_11h17_3_4772"
# base_path = "/home/kosz/Projects/KTH/Thesis/exploring_meta/vision/test/maml_5w1s_min_09_10_17h15_42_7327"
# base_path = "/home/kosz/Projects/KTH/Thesis/exploring_meta/vision/test/maml_5w1s_min_09_10_17h00_42_1871"
base_path = "/home/kosz/Projects/KTH/Thesis/models/vision/mini_imagenet/5w5s/anil_5w5s_min_11_09_00h36_1_6461"

meta_test = True
eval_iters = False
cl_exp = False
rep_exp = True

cl_params = {
    "adapt_steps": 1,
    "inner_lr": 0.5,
    "n_tasks": 10
}

rep_params = {
    "adapt_steps": 1,
    "inner_lr": 0.5,
    "n_tasks": 1,
    "layers": [1, 6, 11, -1],
    # "layers": [2, 4]
}


class Lambda(torch.nn.Module):

    def __init__(self, fn):
        super(Lambda, self).__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


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
    else:
        _, _, test_tasks = get_omniglot(params['ways'], params['shots'])

    if "maml" in path:
        run_maml(params, test_tasks, device)
    else:
        run_anil(params, test_tasks, device)


def run_maml(params, test_tasks, device):
    if 'min' == params['dataset']:
        print('Loading Mini-ImageNet model')
        model = MiniImagenetCNN(params['ways'])
    else:
        print('Loading Omniglot model')
        model = OmniglotCNN(params['ways'])

    # Evaluate the model at every checkpoint
    if eval_iters:
        ckpnt = base_path + "/model_checkpoints/"
        model_ckpnt_results = {}
        for model_ckpnt in os.scandir(ckpnt):
            if model_ckpnt.path.endswith(".pt"):
                print(f'Testing {model_ckpnt.path}')
                res = evaluate_maml(params, model, test_tasks, device, model_ckpnt.path)
                model_ckpnt_results[model_ckpnt.path] = res

        with open(base_path + '/ckpnt_results.json', 'w') as fp:
            json.dump(model_ckpnt_results, fp, sort_keys=True, indent=4)

    final_model = base_path + '/model.pt'
    if meta_test:
        evaluate_maml(params, model, test_tasks, device, final_model)

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


def run_anil(params, test_tasks, device):
    # ANIL
    if 'omni' == params['dataset']:
        fc_neurons = 128
        features = ConvBase(output_size=64, hidden=32, channels=1, max_pool=False)
    else:
        fc_neurons = 1600
        features = ConvBase(output_size=64, channels=3, max_pool=True)
    features = torch.nn.Sequential(features, Lambda(lambda x: x.view(-1, fc_neurons)))
    head = torch.nn.Linear(fc_neurons, params['ways'])
    head = l2l.algorithms.MAML(head, lr=params['inner_lr'])

    # Evaluate the model at every checkpoint
    if eval_iters:
        ckpnt = base_path + "/model_checkpoints/"
        model_ckpnt_results = {}
        for model_ckpnt in os.scandir(ckpnt):
            if model_ckpnt.path.endswith(".pt"):

                if "features" in model_ckpnt.path:
                    features_path = model_ckpnt.path
                    head_path = str.replace(features_path, "features", "head")

                    print(f'Testing {model_ckpnt.path}')
                    res = evaluate_anil(params, features, head, test_tasks, device, features_path, head_path)
                    model_ckpnt_results[model_ckpnt.path] = res

        with open(base_path + '/ckpnt_results.json', 'w') as fp:
            json.dump(model_ckpnt_results, fp, sort_keys=True, indent=4)

    final_features = base_path + '/features.pt'
    final_head = base_path + '/head.pt'

    if meta_test:
        evaluate_anil(params, features, head, test_tasks, device, final_features, final_head)

    if cl_exp:
        print("Running Continual Learning experiment...")
        features.load_state_dict(torch.load(final_features))
        features.to(device)

        head.load_state_dict(torch.load(final_head))
        head.to(device)

        loss = torch.nn.CrossEntropyLoss(reduction='mean')

        run_cl_exp(base_path, head, loss, test_tasks, device,
                   params['ways'], params['shots'], cl_params=cl_params, features=features)

    if rep_exp:
        features.load_state_dict(torch.load(final_features))
        features.to(device)

        head.load_state_dict(torch.load(final_head))
        head.to(device)

        loss = torch.nn.CrossEntropyLoss(reduction='mean')

        # Only check head change
        rep_params['layers'] = [-1]

        print("Running Representation experiment...")
        run_rep_exp(base_path, head, loss, test_tasks, device,
                    params['ways'], params['shots'], rep_params=rep_params, features=features)


def evaluate_maml(params, model, test_tasks, device, path):
    model.load_state_dict(torch.load(path))
    model.to(device)

    maml = l2l.algorithms.MAML(model, lr=params['inner_lr'], first_order=False)

    loss = torch.nn.CrossEntropyLoss(reduction='mean')

    return evaluate(params, test_tasks, maml, loss, device)


def evaluate_anil(params, features, head, test_tasks, device, features_path, head_path):
    features.load_state_dict(torch.load(features_path))
    features.to(device)

    head.load_state_dict(torch.load(head_path))
    head = l2l.algorithms.MAML(head, lr=params['inner_lr'])
    head.to(device)

    loss = torch.nn.CrossEntropyLoss(reduction='mean')

    return evaluate(params, test_tasks, head, loss, device, features=features)


if __name__ == '__main__':
    run(base_path)
    exit()
    # MIN

    # ANIL 5w1s
    base_path = "/home/kosz/Projects/KTH/Thesis/models/vision/mini_imagenet/5w1s/anil_5w1s_min_10_09_10h08_3_8815"
    base_path = "/home/kosz/Projects/KTH/Thesis/models/vision/mini_imagenet/5w1s/anil_5w1s_min_10_09_11h06_2_2906"
    base_path = "/home/kosz/Projects/KTH/Thesis/models/vision/mini_imagenet/5w1s/anil_5w1s_min_10_09_11h59_1_1374"

    # MAML 5w1s
    base_path = "/home/kosz/Projects/KTH/Thesis/models/vision/mini_imagenet/5w1s/maml_5w1s_min_10_09_12h58_3_2722"
    base_path = "/home/kosz/Projects/KTH/Thesis/models/vision/mini_imagenet/5w1s/maml_5w1s_min_10_09_15h12_1_9323"
    base_path = "/home/kosz/Projects/KTH/Thesis/models/vision/mini_imagenet/5w1s/maml_5w1s_min_10_09_17h09_2_6302"

    # ANIL 5w5s
    base_path = "/home/kosz/Projects/KTH/Thesis/models/vision/mini_imagenet/5w5s/anil_5w5s_min_11_09_00h36_1_6461"
    base_path = "/home/kosz/Projects/KTH/Thesis/models/vision/mini_imagenet/5w5s/anil_5w5s_min_11_09_03h38_2_8655"
    base_path = "/home/kosz/Projects/KTH/Thesis/models/vision/mini_imagenet/5w5s/anil_5w5s_min_11_09_05h56_3_6285"

    # MAML 5w5s
    base_path = "/home/kosz/Projects/KTH/Thesis/models/vision/mini_imagenet/5w5s/maml_5w5s_min_31_03_12h53_1_1434"
    base_path = "/home/kosz/Projects/KTH/Thesis/models/vision/mini_imagenet/5w5s/maml_5w5s_min_31_03_12h54_2_1671"
    base_path = "/home/kosz/Projects/KTH/Thesis/models/vision/mini_imagenet/5w5s/maml_5w5s_min_31_03_12h54_3_2104"

    # Omni

    # ANIL 20w1s
    base_path = "/home/kosz/Projects/KTH/Thesis/models/vision/omniglot/20w1s/anil_20w1s_omni_06_09_11h17_1_4305"
    base_path = "/home/kosz/Projects/KTH/Thesis/models/vision/omniglot/20w1s/anil_20w1s_omni_06_09_11h17_2_8126"
    base_path = "/home/kosz/Projects/KTH/Thesis/models/vision/omniglot/20w1s/anil_20w1s_omni_06_09_11h17_3_4772"
    # MAML 20w1s
    base_path = "/home/kosz/Projects/KTH/Thesis/models/vision/omniglot/20w1s/maml_20w1s_omni_31_03_10h18_1_9247"
    base_path = "/home/kosz/Projects/KTH/Thesis/models/vision/omniglot/20w1s/maml_20w1s_omni_31_03_10h21_2_302"
    base_path = "/home/kosz/Projects/KTH/Thesis/models/vision/omniglot/20w1s/maml_20w1s_omni_31_03_10h22_3_7628"

    # ANIL 20w5s
    base_path = "/home/kosz/Projects/KTH/Thesis/models/vision/omniglot/20w5s/anil/anil_20w5s_omni_09_09_13h23_2_4977"
    base_path = "/home/kosz/Projects/KTH/Thesis/models/vision/omniglot/20w5s/anil/anil_20w5s_omni_09_09_13h24_1_775"
    base_path = "/home/kosz/Projects/KTH/Thesis/models/vision/omniglot/20w5s/anil/anil_20w5s_omni_09_09_14h31_3_5663"
    # MAML 20w5s
    base_path = "/home/kosz/Projects/KTH/Thesis/models/vision/omniglot/20w5s/maml/maml_20w5s_omni_31_03_10h23_1_6864"
    base_path = "/home/kosz/Projects/KTH/Thesis/models/vision/omniglot/20w5s/maml/maml_20w5s_omni_31_03_10h24_2_1576"
    base_path = "/home/kosz/Projects/KTH/Thesis/models/vision/omniglot/20w5s/maml/maml_20w5s_omni_31_03_10h24_3_8259"
