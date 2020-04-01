#!/usr/bin/env python3

import os
import json
import torch

from misc_scripts import run_cl_rl_exp
from core_functions.rl import evaluate
from core_functions.policies import DiagNormalPolicy
import cherry as ch

import random
import numpy as np
import gym


env_name = "Particles2D-v1"
base_path = "/home/kosz/Projects/KTH/Thesis/exploring_meta/rl_results/maml_Particles2D-v1_26_03_09h50_42_7376"

evaluate_model = True
cl_exp = False
rep_exp = False

cuda = False

# An episode can have either a finite number of steps, e.g 100 for Particles 2D or until done

eval_params = {
    'n_eval_adapt_steps': 1,  # Number of steps to adapt to a new task
    'n_eval_episodes': 5,  # Number of shots per task
    'n_eval_tasks': 20,  # Number of different tasks to evaluate on
}

cl_params = {
    "adapt_steps": 10,
    "adapt_batch_size": 10,  # shots
    "inner_lr": 0.3,
    "gamma": 0.99,
    "tau": 1.0,
    "n_tasks": 5
}


def run(path):
    # Initialize
    with open(path + "/logger.json", "r") as f:
        params = json.load(f)['config']

    device = torch.device('cpu')
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])

    env = gym.make(env_name)
    env.seed(params['seed'])
    env = ch.envs.Torch(env)

    eval_params['inner_lr'] = params['inner_lr']
    eval_params['tau'] = params['tau']
    eval_params['gamma'] = params['gamma']

    obs_size = env.observation_space.shape[0]
    act_size = env.action_space.shape[0]

    final_model = base_path + '/model.pt'

    baseline = ch.models.robotics.LinearValue(obs_size, act_size)
    policy = DiagNormalPolicy(obs_size, act_size)

    # policy.load_state_dict(torch.load(final_model))
    policy.to(device)

    if evaluate_model:
        eval_reward = evaluate(env, policy, baseline, eval_params)
        print(eval_reward)

    # Run a Continual Learning experiment
    if cl_exp:
        print("Running Continual Learning experiment...")

        run_cl_rl_exp(base_path, env, policy, baseline, cl_params=cl_params)


if __name__ == '__main__':
    run(base_path)
