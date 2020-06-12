#!/usr/bin/env python3

import random
import numpy as np
import json

import torch
import cherry as ch
from learn2learn.algorithms import MAML

from utils import make_env
from core_functions.rl import evaluate
from core_functions.policies import DiagNormalPolicy

base = './trained_policies/'
model_path = 'maml_trpo_ML10_25_05_09h38_1_2259'

checkpoint = None  # None or choose a number

path = base + model_path
ML_ALGO = model_path.split('_')[0]
RL_ALGO = model_path.split('_')[1]
DATASET = model_path.split('_')[2]
# In case of ML1 also get which task
if DATASET == 'ML1':
    DATASET += '_' + model_path.split('_')[3]
anil = False if ML_ALGO == 'maml' else True

workers = 5
render = False  # Rendering doesn't work with parallel async envs, use 1 worker

# An episode can have either a finite number of steps, e.g 100 for Particles 2D or until done
eval_params = {
    'n_eval_tasks': 10,  # Number of different tasks to evaluate on,
    'adapt_batch_size': 20,  # Number of shots per task
    'max_path_length': 150,
    'adapt_steps': 3,  # Number of steps to adapt to a new task
    'inner_lr': 0.1,
    'gamma': 0.99,
    'tau': 1.0,
}


def run():

    print(f'Testing {ML_ALGO}-{RL_ALGO} on {DATASET}')
    with open(path + '/logger.json', 'r') as f:
        params = json.load(f)['config']
    eval_params['seed'] = params['seed']

    device = torch.device('cpu')
    random.seed(eval_params['seed'])
    np.random.seed(eval_params['seed'])
    torch.manual_seed(eval_params['seed'])

    env = make_env(DATASET, workers, eval_params['seed'], test=True)

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

    test_rewards, av_test_rew, av_test_suc = evaluate(RL_ALGO, env, policy, baseline, eval_params, anil=anil, render=render)
    print(f'Average meta-testing reward: {av_test_rew}')
    print(f'Average meta-testing success rate: {av_test_suc * 100}%')


if __name__ == '__main__':
    run()
