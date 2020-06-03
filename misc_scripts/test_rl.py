#!/usr/bin/env python3

import os
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

base = '/home/kosz/Projects/KTH/Thesis/exploring_meta/rl/results/'
model_path = 'maml_trpo_ML1_push-v1_21_05_13h30_42_2120'
checkpoint = 1351  # or choose a number
path = base + model_path
ML_ALGO = model_path.split('_')[0]
RL_ALGO = model_path.split('_')[1]
DATASET = model_path.split('_')[2] + '_' + model_path.split('_')[3]
anil = False if ML_ALGO == 'maml' else True

workers = 1
cuda = False

render = False  # Rendering doesn't work with parallel async envs, use 1 worker

evaluate_model = False
cl_exp = False
rep_exp = True

# An episode can have either a finite number of steps, e.g 100 for Particles 2D or until done
eval_params = {
    'adapt_steps': 1,  # Number of steps to adapt to a new task
    'adapt_batch_size': 10,  # Number of shots per task
    'max_path_length': 150,
    'n_eval_tasks': 10,  # Number of different tasks to evaluate on
}

cl_params = {
    'max_path_length': 150,
    'normalize_rewards': False,
    'adapt_steps': 3,
    'adapt_batch_size': 10,
    'inner_lr': 0.1,
    'gamma': 0.99,
    'tau': 1.0,
    'n_tasks': 5,
    # PPO
    'ppo_epochs': 3,
    'ppo_clip_ratio': 0.1,
}

rep_params = {
    'metrics': ['CCA', 'CKA_L'],  # CCA, CKA_L, CKA_K
    'max_path_length': 150,
    'adapt_steps': 3,
    'adapt_batch_size': 5,
    'inner_lr': 0.1,
    'gamma': 0.99,
    'tau': 1.0,
    'n_tasks': 1,
    'layers': [2, 4]
}
# Layer 1/3: Linear output
# Layer 2/4: ReLU output


def run():
    cl_params['algo'] = RL_ALGO
    rep_params['algo'] = RL_ALGO
    cl_params['anil'] = anil
    rep_params['anil'] = anil

    print(f'Testing {ML_ALGO}-{RL_ALGO} on {DATASET}')
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

    device = torch.device('cpu')
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])

    env = make_env(DATASET, workers, params['seed'], test=True)

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

    if evaluate_model:
        test_rewards, av_test_rew, av_test_suc = evaluate(RL_ALGO, env, policy, baseline, eval_params, anil=anil, render=render)
        print(f'Average meta-testing reward: {av_test_rew}')
        print(f'Average meta-testing success rate: {av_test_suc * 100}%')

    if cl_exp:
        cl(env, policy, baseline)
    if rep_exp:
        rep(env, policy, baseline)


def cl(env, policy, baseline):
    # Continual Learning experiment
    print('Running Continual Learning experiment...')
    run_cl_rl_exp(path, env, policy, baseline, cl_params)

    adapt_bsz_list = range(1, 10, 1)
    # for adapt_bsz in adapt_bsz_list:
    #     cl_params['adapt_batch_size'] = adapt_bsz
    #     run_cl_rl_exp(path, env, policy, baseline, cl_params)


def rep(env, policy, baseline):
    # Run a Representation change experiment
    print('Running Continual Learning experiment...')
    run_rep_rl_exp(path, env, policy, baseline, rep_params)


if __name__ == '__main__':
    run()
