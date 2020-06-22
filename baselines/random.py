#!/usr/bin/env python3

import argparse
import random
import torch
import numpy as np

from tqdm import trange

import cherry as ch
from learn2learn.algorithms import MAML

from utils import *
from core_functions.policies import DiagNormalPolicy
from core_functions.rl import evaluate_ppo
from core_functions.runner import Runner

params = {
    # Common parameters
    'batch_size': 1,
    'n_episodes': 1,
    'max_path_length': 150,
    'activation': 'tanh',  # for MetaWorld use tanh, others relu
    # Other parameters
    'num_iterations': 1000,
    'save_every': 10000,
    'seed': 3,
    # For evaluation
    'n_tasks': 10,
    'adapt_steps': 3,
    'adapt_batch_size': 20,
    'gamma': 0.99,
    'tau': 1.0,
    'ppo_epochs': 3,
    'ppo_clip_ratio': 0.1,
    'inner_lr': 0.1}

# Environments:
#   - Particles2D-v1
#   - AntDirection-v1
#   - ML1_reach-v1, ML1_pick-place-v1, ML1_push-v1
#   - ML10, ML45

env_name = 'ML10'

workers = 1

wandb = False


class Random(Experiment):

    def __init__(self):
        super(Random, self).__init__('random', env_name, params, path='random_results/', use_wandb=wandb)

        # Set seed
        device = torch.device('cpu')
        random.seed(self.params['seed'])
        np.random.seed(self.params['seed'])
        torch.manual_seed(self.params['seed'])

        env = make_env(env_name, workers, params['seed'])
        self.run(env, device)

    def run(self, env, device):

        baseline = ch.models.robotics.LinearValue(env.state_size, env.action_size)
        policy = DiagNormalPolicy(env.state_size, env.action_size)

        self.log_model(policy, device, input_shape=(1, env.state_size))

        t = trange(self.params['num_iterations'], desc='Iteration', position=0)
        try:
            for iteration in t:

                iter_reward = 0.0

                task_list = env.sample_tasks(self.params['batch_size'])
                for task_i in trange(len(task_list), leave=False, desc='Task', position=0):
                    task = task_list[task_i]
                    env.set_task(task)
                    env.reset()
                    task = Runner(env)

                    episodes = task.run(policy, episodes=params['n_episodes'])
                    task_reward = episodes.reward().sum().item() / params['n_episodes']

                    iter_reward += task_reward

                # Log
                average_return = iter_reward / self.params['batch_size']
                metrics = {'average_return': average_return}

                t.set_postfix(metrics)
                self.log_metrics(metrics)

                if iteration % self.params['save_every'] == 0:
                    self.save_model_checkpoint(policy, str(iteration + 1))
                    self.save_model_checkpoint(baseline, 'baseline_' + str(iteration + 1))

        # Support safely manually interrupt training
        except KeyboardInterrupt:
            print('\nManually stopped training! Start evaluation & saving...\n')
            self.logger['manually_stopped'] = True
            self.params['num_iterations'] = iteration

        self.save_model(policy)
        self.save_model(baseline, name='baseline')

        self.logger['elapsed_time'] = str(round(t.format_dict['elapsed'], 2)) + ' sec'
        # Evaluate on new test tasks
        policy = MAML(policy, lr=self.params['inner_lr'])
        self.logger['test_reward'] = evaluate_ppo(env_name, policy, baseline, params)
        self.log_metrics({'test_reward': self.logger['test_reward']})
        self.save_logs_to_file()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A random policy on RL tasks')

    parser.add_argument('--env', type=str, default=env_name, help='Pick an environment')
    parser.add_argument('--batch_size', type=int, default=params['batch_size'], help='Batch size')
    parser.add_argument('--num_iterations', type=int, default=params['num_iterations'], help='Number of epochs')
    parser.add_argument('--save_every', type=int, default=params['save_every'], help='Interval to save model')
    parser.add_argument('--seed', type=int, default=params['seed'], help='Seed')

    args = parser.parse_args()

    params['batch_size'] = args.batch_size
    params['num_iterations'] = args.num_iterations
    params['save_every'] = args.save_every
    params['seed'] = args.seed

    Random()
