#!/usr/bin/env python3

import random
import numpy as np
import json

import torch
import cherry as ch
from learn2learn.algorithms import MAML

from utils import make_env
from misc_scripts import run_cl_rl_exp, run_rep_rl_exp
from core_functions.rl import evaluate
from core_functions.policies import DiagNormalPolicy

# BASE PATH
base = '../final_models/rl/ML1_Push/'
# base = '../final_models/rl/ML10_Push/'
# base = '../final_models/rl/Particles2D/'

# MODEL PATH
# model_path = 'random_ML1_push-v1_12_06_17h24_1_9000'
# model_path = 'ppo_ML1_push-v1_16_06_13h08_1_686'
model_path = 'maml_trpo_ML1_push-v1_21_05_13h34_42_9086'


checkpoint = None  # or choose a number
path = base + model_path

workers = 2
render = False  # Rendering doesn't work with parallel async envs, use 1 worker

evaluate_model = True
cl_exp = False
rep_exp = False

# An episode can have either a finite number of steps, e.g 100 for Particles 2D or until done
eval_params = {
    'adapt_steps': 1,  # Number of steps to adapt to a new task
    'adapt_batch_size': 20,  # Number of shots per task
    'inner_lr': 0.1,
    'gamma': 0.99,
    'tau': 1.0,
    'max_path_length': 100,
    'n_tasks': 10,  # Number of different tasks to evaluate on
}

cl_params = {
    'max_path_length': 100,
    'normalize_rewards': False,
    'adapt_steps': 1,
    'adapt_batch_size': 20,
    'eval_batch_size': 50,
    'inner_lr': 0.1,
    'gamma': 0.99,
    'tau': 1.0,
    'n_tasks': 5,
    # PPO
    'ppo_epochs': 3,
    'ppo_clip_ratio': 0.1,
    'extra_info': False,  # if env is metaworld, log success metric
    'seed': 42,
}

rep_params = {
    'metrics': ['CCA', 'CKA_L'],  # CCA, CKA_L, CKA_K
    'max_path_length': 100,
    'adapt_steps': 3,
    'adapt_batch_size': 5,
    'inner_lr': 0.1,
    'gamma': 0.99,
    'tau': 1.0,
    'n_tasks': 1,
    'layers': [2, 4],
    # PPO
    'ppo_epochs': 3,
    'ppo_clip_ratio': 0.1,
    'extra_info': False,  # if env is metaworld, log success metric
    'seed': 42,
}


# Layer 1/3: Linear output
# Layer 2/4: ReLU output


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

    algo = params['algo']
    dataset = params['dataset']

    anil = True if 'anil' in algo else False

    if 'maml' in algo or 'anil' in algo:
        ml_algo = params['algo'].split('_')[0]
        rl_algo = params['algo'].split('_')[1]
    elif 'random' == algo:
        ml_algo = ''
        rl_algo = 'ppo'
    else:
        ml_algo = ''
        rl_algo = params['algo'].split('_')[1]

    cl_params['algo'] = rl_algo
    rep_params['algo'] = rl_algo
    cl_params['anil'] = anil
    rep_params['anil'] = anil
    if 'ML' in dataset:
        rep_params['extra_info'] = True
        cl_params['extra_info'] = True

    device = torch.device('cpu')
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])

    env = make_env(dataset, workers, params['seed'], test=True, max_path_length=eval_params['max_path_length'])

    eval_params.update(params)

    if checkpoint is None:
        policy_path = path + '/model.pt'
        baseline_path = path + '/baseline.pt'
    else:
        policy_path = path + f'/model_checkpoints/model_{checkpoint}.pt'
        baseline_path = path + f'/model_checkpoints/model_baseline_{checkpoint}.pt'

    baseline = ch.models.robotics.LinearValue(env.state_size, env.action_size)
    baseline.load_state_dict(torch.load(baseline_path))
    baseline.to(device)

    policy = DiagNormalPolicy(env.state_size, env.action_size)
    policy.load_state_dict(torch.load(policy_path))
    policy = MAML(policy, lr=eval_params['inner_lr'])
    policy.to(device)

    print(f'Testing {ml_algo}-{rl_algo} on {dataset}')
    if evaluate_model:
        test_rewards, av_test_rew, av_test_suc = evaluate(rl_algo, dataset, policy, baseline, eval_params, anil=anil,
                                                          render=render)
        print(f'Average meta-testing reward: {av_test_rew}')
        print(f'Average meta-testing success rate: {av_test_suc * 100}%')

    if cl_exp:
        cl(dataset, policy, baseline)
    if rep_exp:
        rep(dataset, policy, baseline)


def cl(dataset, policy, baseline):
    # Continual Learning experiment
    print('Running Continual Learning experiment...')
    run_cl_rl_exp(path, dataset, policy, baseline, cl_params, workers)

    # adapt_bsz_list = [10, 25, 50, 100]
    # for adapt_bsz in adapt_bsz_list:
    #     cl_params['adapt_batch_size'] = adapt_bsz
    #     run_cl_rl_exp(path, env, policy, baseline, cl_params)


def rep(dataset, policy, baseline):
    # Run a Representation change experiment
    print('Running Rep Change experiment...')
    run_rep_rl_exp(path, dataset, policy, baseline, rep_params)


if __name__ == '__main__':
    run()
