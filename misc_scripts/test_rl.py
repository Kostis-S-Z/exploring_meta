#!/usr/bin/env python3

import os
import json
import torch

from core_functions.policies import DiagNormalPolicy
import cherry as ch

from utils import run_cl_rl_exp
import random
import numpy as np
import gym


env_name = "Particles2D-v1"
base_path = "/home/kosz/Projects/KTH/Thesis/exploring_meta/rl_results/maml_Particles2D-v1_26_03_09h50_42_7376"

cl_exp = True
rep_exp = False

cuda = False

# An episode can have either a finite number of steps, e.g 100 for Particles 2D or until done

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

    obs_size = env.observation_space.shape[0]
    act_size = env.action_space.shape[0]

    baseline = ch.models.robotics.LinearValue(obs_size, act_size)
    policy = DiagNormalPolicy(obs_size, act_size)

    final_model = base_path + '/model.pt'
    # Run a Continual Learning experiment
    if cl_exp:
        print("Running Continual Learning experiment...")
        policy.load_state_dict(torch.load(final_model))
        policy.to(device)

        run_cl_rl_exp(base_path, env, policy, baseline, cl_params=cl_params)


if __name__ == '__main__':
    run(base_path)
