#!/usr/bin/env python3

import argparse
import random
import torch
import numpy as np
from copy import deepcopy

from tqdm import trange

import gym
import cherry as ch

from utils import *
from core_functions.policies import DiagNormalPolicy
from core_functions.rl import fast_adapt_a2c, meta_optimize

params = {
    "outer_lr": 0.1,  # ?
    "inner_lr": 0.1,  # ?
    "tau": 1.0,
    "gamma": 0.99,
    "backtrack_factor": 0.5,  # Meta-optimizer
    "ls_max_steps": 15,       # Meta-optimizer
    "max_kl": 0.01,           # Meta-optimizer
    "adapt_batch_size": 20,
    "meta_batch_size": 32,
    "adapt_steps": 1,
    "num_iterations": 100,
    "save_every": 1000,
    "seed": 42}

# Environments:
#   - Particles2D-v1
#   - AntDirection-v1
#   - procgen:procgen-coinrun-v0
# env_name = "procgen:procgen-coinrun-v0"
env_name = "Particles2D-v1"

workers = 4

distribution_mode = "easy"
num_levels = 0
start_level = 0
test_worker_interval = 0

cl_test = False
rep_test = False

cuda = False

wandb = False


class MamlRL(Experiment):

    def __init__(self):
        super(MamlRL, self).__init__("maml", env_name, params, path="rl_results/", use_wandb=wandb)

        device = torch.device('cpu')

        random.seed(self.params['seed'])
        np.random.seed(self.params['seed'])
        torch.manual_seed(self.params['seed'])

        env = gym.make(env_name)
        env.seed(self.params['seed'])

        if cuda and torch.cuda.device_count():
            torch.cuda.manual_seed(self.params['seed'])
            device = torch.device('cuda')

        self.run(env, device)

    def run(self, env, device):

        if env_name == "procgen:procgen-coinrun-v0":
            obs_size = np.prod(env.observation_space.shape)
            act_size = env.action_space.n
        else:
            obs_size = env.observation_space.shape[0]
            act_size = env.action_space.shape[0]

        print(obs_size)
        print(act_size)

        baseline = ch.models.robotics.LinearValue(obs_size, act_size)
        policy = DiagNormalPolicy(obs_size, act_size)

        if cuda:
            policy.to('cuda')

        self.log_model(policy, device, input_shape=(1, obs_size))  # Input shape is specific to dataset

        t = trange(self.params['num_iterations'])
        try:

            for iteration in t:
                # opt.zero_grad()

                iter_reward = 0
                iter_replays = []
                iter_policies = []

                for task_i in range(self.params['meta_batch_size']):

                    clone = deepcopy(policy)
                    env.reset()

                    task = ch.envs.Runner(env, meta_env=False)
                    task_replay = []

                    # Adapt
                    for step in range(self.params['adapt_steps']):
                        train_episodes = task.run(clone, episodes=self.params['adapt_batch_size'])
                        task_replay.append(train_episodes)
                        clone = fast_adapt_a2c(clone, train_episodes, baseline,
                                               self.params['inner_lr'], self.params['gamma'], self.params['tau'],
                                               first_order=True)

                    # Compute validation Loss
                    valid_episodes = task.run(clone, episodes=self.params['adapt_batch_size'])
                    task_replay.append(valid_episodes)

                    iter_reward += valid_episodes.reward().sum().item() / self.params['adapt_batch_size']
                    iter_replays.append(task_replay)
                    iter_policies.append(clone)

                adapt_reward = iter_reward / self.params['meta_batch_size']
                metrics = {'adapt_reward': adapt_reward}

                t.set_postfix(metrics)
                self.log_metrics(metrics)

                meta_optimize(self.params, policy, baseline, iter_replays, iter_policies, cuda)

                if iteration % self.params['save_every'] == 0:
                    self.save_model_checkpoint(policy, str(iteration))

        # Support safely manually interrupt training
        except KeyboardInterrupt:
            print('\nManually stopped training! Start evaluation & saving...\n')
            self.logger['manually_stopped'] = True
            self.params['num_iterations'] = iteration

        self.save_model(policy)

        self.logger['elapsed_time'] = str(round(t.format_dict['elapsed'], 2)) + ' sec'

        # self.logger['test_acc'] = self.evaluate(test_tasks, maml, loss, device)

        self.save_logs_to_file()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MAML on RL tasks')

    parser.add_argument('--env', type=str, default=env_name, help='Pick an environment')

    args = parser.parse_args()

    MamlRL()
