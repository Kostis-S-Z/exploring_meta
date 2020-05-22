#!/usr/bin/env python3

import argparse
import random
import torch
import numpy as np
from copy import deepcopy

import cherry as ch
from learn2learn.algorithms import MAML

from utils import *
from core_functions.policies import DiagNormalPolicyANIL
from core_functions.rl import fast_adapt_ppo, evaluate_ppo, set_device

#   - ML1_reach-v1, ML1_pick-place-v1, ML1_push-v1
#   - ML10, ML45
env_name = 'ML1_push-v1'
workers = 5

params = {
    # Inner loop parameters
    'ppo_epochs': 2,
    'ppo_clip_ratio': 0.1,
    'inner_lr': 0.05,
    'adapt_steps': 1,
    'adapt_batch_size': 10,  # 'shots' (will be *evenly* distributed across workers)
    # Outer loop parameters
    'meta_batch_size': 20,  # 'ways'
    'outer_lr': 0.1,
    # Common parameters
    'activation': 'tanh',  # for MetaWorld use tanh, others relu
    'tau': 1.0,
    'gamma': 0.99,
    # Other parameters
    'num_iterations': 1000,
    'save_every': 25,
    'seed': 42}


def main():
    # Set seed
    device = torch.device('cpu')
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    set_device(device)

    env = make_env(env_name, workers, params['seed'])

    task_list = env.sample_tasks(params['meta_batch_size'])
    env.set_task(task_list[0])
    env.reset()
    task = ch.envs.Runner(env)

    baseline = ch.models.robotics.LinearValue(env.state_size, env.action_size)
    policy = DiagNormalPolicyANIL(env.state_size, env.action_size, fc_neurons=100)
    policy = MAML(policy, lr=params['inner_lr'])
    meta_optimizer = torch.optim.Adam(policy.parameters(), lr=params['outer_lr'])


if __name__ == '__main__':
    main()


