#!/usr/bin/env python3

import argparse
import random
import torch
import numpy as np
from copy import deepcopy

from tqdm import trange

import cherry as ch

from utils import *
from core_functions.policies import DiagNormalPolicy
from core_functions.rl import fast_adapt_trpo, meta_optimize_trpo, evaluate_trpo, set_device
from core_functions.runner import Runner


params = {
    # Inner loop parameters
    'inner_lr': 0.001,
    'max_path_length': 150,  # [100, 150] or None=use the maximum length (None currently WIP)
    'adapt_steps': 1,
    'adapt_batch_size': 20,  # 'shots' (will be *evenly* distributed across workers)
    # Outer loop parameters
    'meta_batch_size': 40,  # 'ways'
    'outer_lr': 0.3,
    'backtrack_factor': 0.5,
    'ls_max_steps': 15,
    'max_kl': 0.01,
    # Common parameters
    'activation': 'tanh',  # for MetaWorld use tanh, others relu
    'tau': 1.0,
    'gamma': 0.99,
    # Other parameters
    'num_iterations': 250,
    'save_every': 250,
    'seed': 42
    # For evaluation
    }

eval_params = {
    'adapt_steps': 5,  # Number of steps to adapt to a new task
    'adapt_batch_size': 10,  # Number of shots per task
    'n_tasks': 10,  # Number of different tasks to evaluate on
    'inner_lr': params['inner_lr'],  # Just use the default parameters for evaluating
    'max_path_length': params['max_path_length'],
    'tau': params['tau'],
    'gamma': params['gamma'],
    'seed': params['seed']
}

# Environments:
#   - Particles2D-v1
#   - AntDirection-v1
#   - ML1_reach-v1, ML1_pick-place-v1, ML1_push-v1
#   - ML10, ML45

env_name = 'ML1_push-v1'

workers = 5

wandb = True

extra_info = True if 'ML' in env_name else False


class MamlTRPO(Experiment):

    def __init__(self):
        super(MamlTRPO, self).__init__('maml_trpo', env_name, params, path='results/', use_wandb=wandb)

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

        self.log_model(policy, device, input_shape=(1, env.state_size))

        t = trange(self.params['num_iterations'], desc='Iteration', position=0)
        try:
            for iteration in t:

                iter_loss = 0.0
                iter_reward = 0.0
                # iter_success_per_task = {}
                iter_replays = []
                iter_policies = []

                task_list = env.sample_tasks(self.params['meta_batch_size'])

                for task_i in trange(len(task_list), leave=False, desc='Task', position=0):
                    task = task_list[task_i]
                    # task_id = f'task_{task["task"]}'

                    learner = deepcopy(policy)
                    env.set_task(task)
                    env.reset()
                    task = Runner(env, extra_info=extra_info)

                    # Adapt
                    learner, eval_loss, task_replay, task_rew, task_suc = fast_adapt_trpo(task, learner, baseline,
                                                                                          self.params, first_order=True)

                    # Calculate average success rate of support episodes
                    # task_adapt_suc = get_ep_successes(task_replay[0]) / self.params['adapt_batch_size']
                    # iter_success_per_task[task_id + '_adapt'] = task_adapt_suc
                    # iter_success_per_task[task_id] = task_suc
                    iter_reward += task_rew
                    iter_loss += eval_loss.item()
                    iter_replays.append(task_replay)
                    iter_policies.append(learner)

                # Log
                average_return = iter_reward / self.params['meta_batch_size']
                average_loss = iter_loss / self.params['meta_batch_size']
                metrics = {'average_return': average_return,
                           'loss': average_loss}
                t.set_postfix(metrics)
                # metrics.update(iter_success_per_task)
                self.log_metrics(metrics)

                # Meta-optimize
                meta_optimize_trpo(self.params, policy, baseline, iter_replays, iter_policies)

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
        self.logger['test_reward'] = evaluate_trpo(env_name, policy, baseline, eval_params)
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

    MamlTRPO()
