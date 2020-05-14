#!/usr/bin/env python3

import argparse
import random
import torch
import numpy as np
from copy import deepcopy

from tqdm import trange, tqdm

import cherry as ch
from learn2learn.algorithms import MAML

from utils import *
from core_functions.policies import DiagNormalPolicy
from core_functions.rl import fast_adapt_trpo, meta_optimize_trpo, evaluate_trpo
from misc_scripts import run_cl_rl_exp


params = {
    # Inner loop parameters
    'inner_lr': 0.1,
    'adapt_steps': 1,
    'adapt_batch_size': 10,  # 'shots' (will be *evenly* distributed across workers)
    # Outer loop parameters
    'meta_batch_size': 20,  # 'ways'
    'outer_lr': 0.1,
    'backtrack_factor': 0.5,
    'ls_max_steps': 15,
    'max_kl': 0.01,
    # Common parameters
    'activation': 'tanh',  # for MetaWorld use tanh, others relu
    'tau': 1.0,
    'gamma': 0.99,
    # Other parameters
    'num_iterations': 1000,
    'save_every': 25,
    'seed': 42}


eval_params = {
    'adapt_steps': 5,  # Number of steps to adapt to a new task
    'adapt_batch_size': 10,  # Number of shots per task
    'n_eval_tasks': 10,  # Number of different tasks to evaluate on
    'inner_lr': params['inner_lr'],  # Just use the default parameters for evaluating
    'tau': params['tau'],
    'gamma': params['gamma'],
}

cl_params = {
    'adapt_steps': 10,
    'adapt_batch_size': 10,  # shots
    'inner_lr': 0.3,
    'gamma': 0.99,
    'tau': 1.0,
    'n_tasks': 5
}


# Environments:
#   - Particles2D-v1
#   - AntDirection-v1
#   - ML1_reach-v1, ML1_pick-place-v1, ML1_push-v1
#   - ML10, ML45

env_name = 'ML1_pick-place-v1'

workers = 2

wandb = False

cl_test = False
rep_test = False


class MamlTRPO(Experiment):

    def __init__(self):
        super(MamlTRPO, self).__init__('maml_trpo', env_name, params, path='results/', use_wandb=wandb)

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
        policy = MAML(policy, lr=self.params['inner_lr'])

        self.log_model(policy, device, input_shape=(1, env.state_size))

        t = trange(self.params['num_iterations'], desc='Iteration', position=0)
        try:
            for iteration in t:

                iter_reward = 0.0
                iter_replays = []
                iter_policies = []

                task_list = env.sample_tasks(self.params['meta_batch_size'])

                for task_i in trange(len(task_list), leave=False, desc='Task', position=0):
                    task = task_list[task_i]

                    learner = deepcopy(policy)
                    env.set_task(task)
                    env.reset()
                    task = ch.envs.Runner(env)

                    # Adapt
                    learner, task_replay, task_rew = fast_adapt_trpo(task, learner, baseline, self.params,
                                                                     first_order=True, device=device)

                    iter_reward += task_rew
                    iter_replays.append(task_replay)
                    iter_policies.append(learner)

                # Log
                average_return = iter_reward / self.params['meta_batch_size']
                metrics = {'average_return': average_return}
                t.set_postfix(metrics)
                self.log_metrics(metrics)

                # Meta-optimize
                meta_optimize_trpo(self.params, policy, baseline, iter_replays, iter_policies, device)

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
        env = make_env(env_name, workers, params['seed'], test=True)
        self.logger['test_reward'] = evaluate_trpo(env, policy, baseline, eval_params)
        self.log_metrics({'test_reward': self.logger['test_reward']})
        self.save_logs_to_file()

        if cl_test:
            print('Running Continual Learning experiment...')
            run_cl_rl_exp(self.model_path, env, policy, baseline, cl_params=cl_params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MAML-TRPO on RL tasks')

    parser.add_argument('--env', type=str, default=env_name, help='Pick an environment')

    parser.add_argument('--outer_lr', type=float, default=params['outer_lr'], help='Outer lr')
    parser.add_argument('--inner_lr', type=float, default=params['inner_lr'], help='Inner lr')
    parser.add_argument('--adapt_steps', type=int, default=params['adapt_steps'], help='Adaptation steps in inner loop')
    parser.add_argument('--meta_batch_size', type=int, default=params['meta_batch_size'], help='Batch size')
    parser.add_argument('--adapt_batch_size', type=int, default=params['adapt_batch_size'], help='Adapt batch size')

    parser.add_argument('--num_iterations', type=int, default=params['num_iterations'], help='Number of epochs')
    parser.add_argument('--save_every', type=int, default=params['save_every'], help='Interval to save model')

    parser.add_argument('--seed', type=int, default=params['seed'], help='Seed')

    args = parser.parse_args()

    params['outer_lr'] = args.outer_lr
    params['inner_lr'] = args.inner_lr
    params['adapt_steps'] = args.adapt_steps
    params['meta_batch_size'] = args.meta_batch_size
    params['adapt_batch_size'] = args.adapt_batch_size

    params['num_iterations'] = args.num_iterations
    params['save_every'] = args.save_every

    params['seed'] = args.seed

    MamlTRPO()
