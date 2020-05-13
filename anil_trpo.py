#!/usr/bin/env python3

import argparse
import random
import torch
import numpy as np
from copy import deepcopy

from tqdm import trange

import gym
import cherry as ch
import learn2learn as l2l

from utils import *
from core_functions.policies import DiagNormalPolicyANIL
from core_functions.rl import fast_adapt_trpo, meta_optimize_trpo, evaluate_trpo

# ANIL Defaults: meta_batch_size: 40, adapt_steps: 1, adapt_batch_size: 20, inner_lr: 0.1
# Train for 500 epochs then evaluate on a new set of tasks.

params = {
    "outer_lr": 0.1,  # ?
    "inner_lr": 0.1,
    "tau": 1.0,
    "gamma": 0.99,
    "backtrack_factor": 0.5,  # Meta-optimizer
    "ls_max_steps": 15,  # Meta-optimizer
    "max_kl": 0.01,  # Meta-optimizer
    "adapt_batch_size": 20,  # "shots"
    "meta_batch_size": 40,  # "ways"
    "adapt_steps": 1,
    "num_iterations": 500,
    "save_every": 50,
    "seed": 42}

# Adapt steps: how many times you will replay & learn a specific number of episodes (=adapt_batch_size)
# Meta_batch_size (=ways): how many tasks an epoch has. (a task can have one or many episodes)
# Adapt_batch_size (=shots): number of episodes (not steps!) during adaptation

eval_params = {
    'n_eval_adapt_steps': 5,  # Number of steps to adapt to a new task
    'n_eval_episodes': 10,  # Number of shots per task
    'n_eval_tasks': 10,  # Number of different tasks to evaluate on
    'inner_lr': params['inner_lr'],  # Just use the default parameters for evaluating
    'tau': params['tau'],
    'gamma': params['gamma'],
}

cl_params = {
    "adapt_steps": 10,
    "adapt_batch_size": 10,  # shots
    "inner_lr": 0.3,
    "gamma": 0.99,
    "tau": 1.0,
    "n_tasks": 5
}

network = [100, 100]
fc_neurons = 100

# Environments:
#   - Particles2D-v1
#   - AntDirection-v1
env_name = "Particles2D-v1"
workers = 4

cl_test = False
rep_test = False

cuda = False

wandb = False


class AnilTRPO(Experiment):

    def __init__(self):
        super(AnilTRPO, self).__init__("anil", env_name, params, path="rl/results/", use_wandb=wandb)

        def make_env():
            env = gym.make(env_name)
            env = ch.envs.ActionSpaceScaler(env)
            return env

        device = torch.device('cpu')

        random.seed(self.params['seed'])
        np.random.seed(self.params['seed'])
        torch.manual_seed(self.params['seed'])

        env = l2l.gym.AsyncVectorEnv([make_env for _ in range(workers)])
        env.seed(self.params['seed'])
        env.set_task(env.sample_tasks(1)[0])
        env = ch.envs.Torch(env)

        if cuda and torch.cuda.device_count():
            torch.cuda.manual_seed(self.params['seed'])
            device = torch.device('cuda')

        self.run(env, device)

    def run(self, env, device):

        baseline = ch.models.robotics.LinearValue(env.state_size, env.action_size)

        policy = DiagNormalPolicyANIL(env.state_size, env.action_size, fc_neurons, network)
        policy = l2l.algorithms.MAML(policy, lr=self.params['inner_lr'])

        self.log_model(policy.body, device, input_shape=(1, env.state_size), name='body')
        self.log_model(policy.head, device, input_shape=(env.action_size, fc_neurons), name='head')

        t = trange(self.params['num_iterations'])
        try:
            for iteration in t:

                iter_reward = 0.0
                iter_replays = []
                iter_policies = []

                task_list = env.sample_tasks(self.params['meta_batch_size'])
                for task_i in trange(len(task_list), leave=False, desc="Task", position=0):
                    task = task_list[task_i]

                    clone = deepcopy(policy)
                    env.set_task(task)
                    env.reset()
                    task = ch.envs.Runner(env)

                    # Fast adapt
                    learner, task_replay, task_rew = fast_adapt_trpo(task, clone, baseline, self.params, anil=True,
                                                                     first_order=True, device=device)

                    iter_reward += task_rew
                    iter_replays.append(task_replay)
                    iter_policies.append(learner)

                # Meta-optimize
                meta_optimize_trpo(self.params, policy, baseline, iter_replays, iter_policies, device)

                # Log
                adapt_reward = iter_reward / self.params['meta_batch_size']
                metrics = {'adapt_reward': adapt_reward}

                t.set_postfix(metrics)
                self.log_metrics(metrics)

                if iteration % self.params['save_every'] == 0:
                    self.save_model_checkpoint(policy.body, 'body_' + str(iteration + 1))
                    self.save_model_checkpoint(policy.head, 'head_' + str(iteration + 1))
                    self.save_model_checkpoint(baseline, 'baseline_' + str(iteration + 1))

        # Support safely manually interrupt training
        except KeyboardInterrupt:
            print('\nManually stopped training! Start evaluation & saving...\n')
            self.logger['manually_stopped'] = True
            self.params['num_iterations'] = iteration

        self.save_model(policy.body, name="body")
        self.save_model(policy.head, name="head")
        self.save_model(baseline, name="baseline")

        self.logger['elapsed_time'] = str(round(t.format_dict['elapsed'], 2)) + ' sec'
        # Evaluate on new test tasks
        self.logger['test_reward'] = evaluate_trpo(env, policy, baseline, eval_params)
        self.log_metrics({'test_reward': self.logger['test_reward']})
        self.save_logs_to_file()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ANIL-TRPO on RL-MuJoCo tasks')

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

    AnilTRPO()
