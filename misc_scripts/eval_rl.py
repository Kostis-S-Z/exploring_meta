#!/usr/bin/env python3

import random
import numpy as np
import json

import torch
import cherry as ch
from learn2learn.algorithms import MAML

from misc_scripts import run_cl_rl_exp, run_rep_rl_exp, measure_change_through_time
from core_functions.rl import evaluate
from core_functions.policies import DiagNormalPolicy, DiagNormalPolicyANIL

from utils.plotter import bar_plot_ml10, bar_plot_ml10_one_task

# BASE PATH
# base = '/home/kosz/Projects/KTH/Thesis/models/rl/Particles2D/'
# base = '/home/kosz/Projects/KTH/Thesis/models/rl/ML1_Push/'

# MODEL PATH
base = '/home/kosz/Projects/KTH/Thesis/models/rl/ML10/final/'
# model_path = 'anil_trpo_ML10_30_06_16h37_42_363'
model_path = 'maml_trpo_ML10_30_06_16h36_42_2714'
# base = '/home/kosz/Projects/KTH/Thesis/models/rl/ML10/'
# model_path = 'ppo_ML10_04_06_18h37_42_6537'

path = base + model_path
checkpoint = None  # or choose a number

save_res = True
test_on_train = False  # Meta-testing on the train tasks of ML10
each3 = True  # Sample 3 of each task

workers = 1
render = False  # Rendering doesn't work with parallel async envs, use 1 worker

EVALUATE = False  # Standard meta-testing
eval_params = {
    'adapt_steps': 1,  # 1, 5
    'adapt_batch_size': 10,  # 10, 300
    'inner_lr': 0.001,  # 0.001 or 0.05
    'gamma': 0.99,  # 0.99, 0.95
    'tau': 1.0,
    'max_path_length': 150,
    # n_tasks: 20 or 'reach' Number of different tasks to evaluate on or explicit state which tasks
    # When you explicitly state a task to evaluate on, make sure that each3 = False and that the task belongs to the
    # correct set. e.g 'door-close' is on test set, so you should have test_on_train=False
    'n_tasks': 20,
    'ppo_epochs': 3,
    'ppo_clip_ratio': 0.1
}

RUN_CL = False  # Continual Learning experiment
cl_params = {
    'max_path_length': 150,
    'normalize_rewards': False,
    'adapt_steps': 1,  # 1, 5
    'adapt_batch_size': 10,  # 10, 300
    'eval_batch_size': 10,  # 10, 300
    'inner_lr': 0.001,  # 0.001 or 0.05
    'gamma': 0.99,   # 0.99, 0.95
    'tau': 1.0,
}

RUN_RC = True  # Representation Change experiment
rep_params = {
    'metrics': ['CCA'],  # CCA, CKA_L, CKA_K
    'max_path_length': 150,
    'adapt_steps': 1,
    'adapt_batch_size': 10,
    'inner_lr': 0.001,
    'gamma': 0.99,
    'tau': 1.0,
    'n_tasks': 1,
    'eval_each_task': True,  # If true ignore, n_tasks
    'layers': [2, 4, -1],  # 1/3: Linear output, 2/4: ReLU output
}


def run():
    try:
        with open(path + '/logger.json', 'r') as f:
            params = json.load(f)['config']
    except FileNotFoundError:
        print('WARNING CONFIG NOT FOUND. Using default parameters')
        params = dict()
        params['inner_lr'] = 0.1
        params['ppo_epochs'] = 3
        params['ppo_clip_ratio'] = 0.1
        params['tau'] = 1.0
        params['gamma'] = 0.99
        params['seed'] = 42

    eval_params['seed'] = params['seed']
    cl_params['seed'] = params['seed']
    rep_params['seed'] = params['seed']
    algo = params['algo']
    env_name = params['dataset']

    anil = True if 'anil' in algo else False

    if 'maml' in algo or 'anil' in algo:
        ml_algo = params['algo'].split('_')[0]
        rl_algo = params['algo'].split('_')[1]
    elif 'ppo' == algo or 'random' == algo:
        ml_algo = ''
        rl_algo = 'ppo'
    else:
        ml_algo = ''
        rl_algo = params['algo'].split('_')[1]

    cl_params['algo'] = rl_algo
    rep_params['algo'] = rl_algo
    cl_params['anil'] = anil
    rep_params['anil'] = anil
    if 'ML' in env_name:
        state_size = 9
        action_size = 4
        rep_params['extra_info'], cl_params['extra_info'] = True, True
    else:
        state_size = 2
        action_size = 2
        rep_params['extra_info'], cl_params['extra_info'] = False, False

    if checkpoint is None:
        baseline_path = path + '/baseline.pt'
        if ml_algo == 'anil':
            head_path = path + '/head.pt'
            body_path = path + '/body.pt'
        else:
            policy_path = path + '/model.pt'
    else:
        baseline_path = path + f'/model_checkpoints/model_baseline_{checkpoint}.pt'
        if ml_algo == 'maml':
            policy_path = path + f'/model_checkpoints/model_{checkpoint}.pt'
        else:
            head_path = path + f'/model_checkpoints/model_head_{checkpoint}.pt'
            body_path = path + f'/model_checkpoints/model_body_{checkpoint}.pt'

    device = torch.device('cpu')
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])

    baseline = ch.models.robotics.LinearValue(state_size, action_size)
    baseline.load_state_dict(torch.load(baseline_path))
    baseline.to(device)

    if ml_algo == 'anil':
        policy = DiagNormalPolicyANIL(state_size, action_size, params['fc_neurons'])
        policy.head.load_state_dict(torch.load(head_path))
        policy.body.load_state_dict(torch.load(body_path))
    else:
        policy = DiagNormalPolicy(state_size, action_size)
        policy.load_state_dict(torch.load(policy_path))

    policy = MAML(policy, lr=eval_params['inner_lr'])
    policy.to(device)

    print(f'Testing {ml_algo}-{rl_algo} on {env_name}')
    if EVALUATE:
        t_test = 'train' if test_on_train else 'test'
        test_rewards, av_test_rew, av_test_suc, res_per_task = evaluate(rl_algo, env_name, policy, baseline,
                                                                        eval_params, anil=anil, render=render,
                                                                        test_on_train=test_on_train, each3=each3)
        print(f'Average meta-testing reward: {av_test_rew}')
        print(f'Average meta-testing success rate: {av_test_suc * 100}%')

        if save_res:
            with open(f"{params['algo']}_{t_test}_{params['seed']}.json", 'w') as f:
                f.write(json.dumps(res_per_task))
        # with open(f"maml_trpo_test_{i}.json") as f:
        #     res_per_task = json.loads(f.read())

        for key, val in res_per_task.items():
            print(f'{key}: \n\tRewards: {val[::2]}\n\tSuccess: {val[1::2]}\n')

        bar_plot_ml10(res_per_task, f"{params['algo']}_{t_test}_{params['seed']}.png")

    if RUN_CL:
        print('Running Continual Learning experiment...')
        run_cl_rl_exp(path, env_name, policy, baseline, cl_params, workers, test_on_train=test_on_train)
    if RUN_RC:
        print('Running Rep Change experiment...')
        run_rep_rl_exp(path, env_name, policy, baseline, rep_params)


if __name__ == '__main__':
    run()
