#!/usr/bin/env python3

import os
import random
import numpy as np
import json

import torch
import cherry as ch
import learn2learn as l2l

import utils
from misc_scripts import run_cl_rl_exp
from core_functions.rl import evaluate
from core_functions.policies import DiagNormalPolicy

base = '/home/kosz/Projects/KTH/Thesis/exploring_meta/rl/results/'
model_path = 'maml_ppo_ML1_push-v1_15_05_10h31_42_1472'
path = base + model_path
ANIL = False
ALGO = model_path.split('_')[1]

render = True

evaluate_model = True
cl_exp = False
rep_exp = False

workers = 1
cuda = False

# An episode can have either a finite number of steps, e.g 100 for Particles 2D or until done

eval_params = {
    'adapt_steps': 2,  # Number of steps to adapt to a new task
    'adapt_batch_size': 5,  # Number of shots per task
    'n_eval_tasks': 10,  # Number of different tasks to evaluate on
}

cl_params = {
    'normalize_rewards': False,
    'adapt_steps': 3,
    'adapt_batch_size': 10,  # shots
    'inner_lr': 0.05,
    'gamma': 0.99,
    'tau': 1.0,
    'n_tasks': 5
}


def run():
    # Initialize
    try:
        with open(path + '/logger.json', 'r') as f:
            params = json.load(f)['config']
    except FileNotFoundError:
        print('WARNING CONFIG NOT FOUND. Using default parameters')
        params = dict()
        params['inner_lr'] = 0.1
        params['tau'] = 1.0
        params['gamma'] = 0.99
        params['seed'] = 42

    device = torch.device('cpu')

    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])

    env = make_env('ML1', 'pick-place-v1', params['seed'])

    if cuda and torch.cuda.device_count():
        print(f'Running on {torch.cuda.get_device_name(0)}')
        torch.cuda.manual_seed(params['seed'])
        device = torch.device('cuda')

    eval_params['inner_lr'] = params['inner_lr']
    eval_params['tau'] = params['tau']
    eval_params['gamma'] = params['gamma']

    policy_path = path + '/model_checkpoints/model_101.pt'
    baseline_path = path + '/model_checkpoints/model_baseline_101.pt'

    baseline = ch.models.robotics.LinearValue(env.state_size, env.action_size)
    baseline.load_state_dict(torch.load(baseline_path))
    baseline.to(device)

    policy = DiagNormalPolicy(env.state_size, env.action_size)
    policy.load_state_dict(torch.load(policy_path))
    policy.to(device)

    if evaluate_model:
        eval_reward = evaluate(ALGO, env, policy, baseline, eval_params, anil=ANIL, render=render)
        print(eval_reward)

    # Run a Continual Learning experiment
    if cl_exp:
        print('Running Continual Learning experiment...')
        run_cl_rl_exp(path, env, policy, baseline, cl_params=cl_params)


def make_env(benchmark, task, seed, test=False):
    benchmark_env = getattr(utils, f'MetaWorld{benchmark}')  # Use modded MetaWorld env

    def init_env():
        if test:
            env = benchmark_env.get_test_tasks(task)
        else:
            env = benchmark_env.get_train_tasks(task)

        env = ch.envs.ActionSpaceScaler(env)
        return env

    env = l2l.gym.AsyncVectorEnv([init_env for _ in range(workers)])

    env.seed(seed)
    env.set_task(env.sample_tasks(1)[0])
    env = ch.envs.Torch(env)
    return env


if __name__ == '__main__':
    run()
