#!/usr/bin/env python3

import os
import random
import numpy as np
import json

import torch
import cherry as ch
from learn2learn.algorithms import MAML

from utils import make_env
from misc_scripts import run_cl_rl_exp
from core_functions.rl import evaluate
from core_functions.policies import DiagNormalPolicy

base = '/home/kosz/Projects/KTH/Thesis/exploring_meta/rl/results/'
model_path = 'maml_trpo_ML1_push-v1_18_05_16h39_42_3002'
checkpoint = None  # or choose a number
path = base + model_path
ML_ALGO = model_path.split('_')[0]
RL_ALGO = model_path.split('_')[1]
DATASET = model_path.split('_')[2] + '_' + model_path.split('_')[3]

workers = 1
cuda = False

render = False  # if you want to render you need to select just 1 worker

evaluate_model = False
cl_exp = True
rep_exp = False

# An episode can have either a finite number of steps, e.g 100 for Particles 2D or until done
eval_params = {
    'adapt_steps': 2,  # Number of steps to adapt to a new task
    'adapt_batch_size': 1,  # Number of shots per task
    'n_eval_tasks': 10,  # Number of different tasks to evaluate on
}

cl_params = {
    'normalize_rewards': False,
    'adapt_steps': 3,
    'adapt_batch_size': 1,  # shots
    'inner_lr': 0.1,
    'gamma': 0.99,
    'tau': 1.0,
    'n_tasks': 5
}
anil = False if ML_ALGO == 'maml' else True


def run():
    # Initialize
    print(f'Testing {ML_ALGO}-{RL_ALGO} on {DATASET}')
    try:
        with open(path + '/logger.json', 'r') as f:
            params = json.load(f)['config']
    except FileNotFoundError:
        print('WARNING CONFIG NOT FOUND. Using default parameters')
        params = dict()
        params['inner_lr'] = 0.1
        params['ppo_epochs'] = 2
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
        test_rewards, av_test_rew = evaluate(RL_ALGO, env, policy, baseline, eval_params, anil=anil, render=render)
        print('Average meta-testing reward:', av_test_rew)

    # Run a Continual Learning experiment
    if cl_exp:
        print('Running Continual Learning experiment...')
        run_cl_rl_exp(path, env, policy, baseline, cl_params=cl_params)


if __name__ == '__main__':
    run()
