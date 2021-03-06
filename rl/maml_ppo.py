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
from core_functions.rl import fast_adapt_ppo, evaluate_ppo, set_device
from core_functions.runner import Runner


params = {
    # Inner loop parameters
    'ppo_epochs': 3,
    'ppo_clip_ratio': 0.1,
    'inner_lr': 0.01,
    'max_path_length': 150,  # [100, 150] or None=use the maximum length (None currently WIP)
    'adapt_steps': 1,
    'adapt_batch_size': 20,  # 'shots' (will be *evenly* distributed across workers)
    # Outer loop parameters
    'meta_batch_size': 40,  # 'ways'
    'outer_lr': 0.01,
    # Common parameters
    'activation': 'tanh',  # for MetaWorld use tanh, others relu
    'tau': 1.0,
    'gamma': 0.99,
    # Other parameters
    'num_iterations': 1000,
    'save_every': 100,
    'seed': 42}


eval_params = {
    'adapt_steps': 5,  # Number of steps to adapt to a new task
    'adapt_batch_size': 10,  # Number of shots per task
    'n_tasks': 10,  # Number of different tasks to evaluate on
    'inner_lr': params['inner_lr'],  # Just use the default parameters for evaluating
    'max_path_length': params['max_path_length'],
    'tau': params['tau'],
    'gamma': params['gamma'],
    'ppo_epochs': params['ppo_epochs'],
    'ppo_clip_ratio': params['ppo_clip_ratio'],
    'seed': params['seed']
}

# Environments:
#   - Particles2D-v1
#   - AntDirection-v1
#   - ML1_reach-v1, ML1_pick-place-v1, ML1_push-v1
#   - ML10, ML45

env_name = 'ML10'

workers = 5
wandb = False

extra_info = True if 'ML' in env_name else False


class MamlPPO(Experiment):

    def __init__(self):
        super(MamlPPO, self).__init__('maml_ppo', env_name, params, path='results/', use_wandb=wandb)

        # Set seed
        device = torch.device('cpu')
        random.seed(self.params['seed'])
        np.random.seed(self.params['seed'])
        torch.manual_seed(self.params['seed'])

        env = make_env(env_name, workers, params['seed'], max_path_length=params['max_path_length'])
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
                    task = Runner(env, extra_info=extra_info)

                    # Adapt
                    eval_loss, task_rew, task_suc = fast_adapt_ppo(task, learner, baseline, self.params)

                    iter_reward += task_rew
                    iter_loss += eval_loss
                    # print(f'\tTask {task_i} reward: {task_rew} | Loss : {eval_loss.item()}')

                # Log
                average_return = iter_reward / self.params['meta_batch_size']
                av_loss = iter_loss / self.params['meta_batch_size']

                # print(f'Iter {iteration} average reward: {average_return} | Loss : {av_loss}')
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
        self.logger['test_reward'] = evaluate_ppo(env_name, policy, baseline, eval_params)
        self.log_metrics({'test_reward': self.logger['test_reward']})
        self.save_logs_to_file()


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

    MamlPPO()
