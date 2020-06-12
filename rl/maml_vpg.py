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
from core_functions.rl import fast_adapt_vpg, evaluate_vpg, set_device
from misc_scripts import run_cl_rl_exp

params = {
    # Inner loop parameters
    'inner_lr': 0.05,
    'max_path_length': 150,
    'adapt_steps': 1,
    'adapt_batch_size': 10,  # 'shots' (will be *evenly* distributed across workers)
    # Outer loop parameters
    'meta_batch_size': 20,  # 'ways'
    'outer_lr': 0.05,
    # Common parameters
    'dice': False,
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
    'max_path_length': params['max_path_length'],
    'tau': params['tau'],
    'gamma': params['gamma'],
}


# Environments:
#   - Particles2D-v1
#   - AntDirection-v1
#   - ML1_reach-v1, ML1_pick-place-v1, ML1_push-v1
#   - ML10, ML45

env_name = 'Particles2D-v1'

workers = 5
wandb = False


class MamlVPG(Experiment):

    def __init__(self):
        super(MamlVPG, self).__init__('maml_vpg', env_name, params, path='results/', use_wandb=wandb)

        # Set seed
        device = torch.device('cpu')
        random.seed(self.params['seed'])
        np.random.seed(self.params['seed'])
        torch.manual_seed(self.params['seed'])

        env = make_env(env_name, workers, params['seed'])
        self.run(env, device)

    def run(self, env, device):

        set_device(device)
        baseline = ch.models.robotics.LinearValue(env.state_size, env.action_size)
        policy = DiagNormalPolicy(env.state_size, env.action_size)
        policy = MAML(policy, lr=self.params['inner_lr'])

        meta_optimizer = torch.optim.Adam(policy.parameters(), lr=self.params['outer_lr'])

        self.log_model(policy, device, input_shape=(1, env.state_size))

        t = trange(self.params['num_iterations'], desc='Iteration', position=0)
        try:
            for iteration in t:
                meta_optimizer.zero_grad()

                iter_reward = 0.0
                iter_loss = 0.0

                task_list = env.sample_tasks(self.params['meta_batch_size'])

                for task_i in trange(len(task_list), leave=False, desc='Task', position=0):
                    task = task_list[task_i]

                    learner = policy.clone()
                    env.set_task(task)
                    env.reset()
                    task = ch.envs.Runner(env)

                    # Adapt
                    loss, task_rew, task_suc = fast_adapt_vpg(task, learner, baseline, self.params, first_order=False)

                    iter_reward += task_rew
                    iter_loss += loss

                # Log
                average_return = iter_reward / self.params['meta_batch_size']
                av_loss = iter_loss / self.params['meta_batch_size']
                metrics = {'average_return': average_return,
                           'loss': av_loss.item()}

                t.set_postfix(metrics)
                self.log_metrics(metrics)

                # Meta-optimize: Back-propagate through the accumulated gradients and optimize
                av_loss.backward()
                meta_optimizer.step()

                if iteration % self.params['save_every'] == 0:
                    self.save_model_checkpoint(policy.module, str(iteration + 1))
                    self.save_model_checkpoint(baseline, 'baseline_' + str(iteration + 1))

        # Support safely manually interrupt training
        except KeyboardInterrupt:
            print('\nManually stopped training! Start evaluation & saving...\n')
            self.logger['manually_stopped'] = True
            self.params['num_iterations'] = iteration

        self.save_model(policy.module)
        self.save_model(baseline, name='baseline')

        self.logger['elapsed_time'] = str(round(t.format_dict['elapsed'], 2)) + ' sec'
        # Evaluate on new test tasks
        env = make_env(env_name, workers, params['seed'], test=True)
        self.logger['test_reward'] = evaluate_vpg(env, policy, baseline, eval_params)
        self.log_metrics({'test_reward': self.logger['test_reward']})
        self.save_logs_to_file()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MAML-VPG on RL tasks')

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

    MamlVPG()
