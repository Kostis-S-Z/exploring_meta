#!/usr/bin/env python3

import argparse
import random
import torch
import numpy as np
from copy import deepcopy

from tqdm import trange, tqdm

import metaworld.benchmarks as mtwrld
import cherry as ch
import learn2learn as l2l

from utils import *
from core_functions.policies import DiagNormalPolicy
from core_functions.rl import fast_adapt_trpo_a2c, meta_optimize
from misc_scripts import run_cl_rl_exp

params = {
    # Inner loop parameters
    "inner_lr": 0.1,
    "adapt_steps": 3,
    "adapt_batch_size": 5,  # "shots"
    # Outer loop parameters
    "meta_batch_size": 10,  # "ways"
    "outer_lr": 0.05,
    "backtrack_factor": 0.5,
    "ls_max_steps": 15,
    "max_kl": 0.01,
    # Common parameters
    "tau": 1.0,
    "gamma": 0.99,
    # Other parameters
    "num_iterations": 50000,
    "save_every": 25,
    "seed": 42}

eval_params = {
    'adapt_steps': 5,  # Number of steps to adapt to a new task
    'n_eval_episodes': 10,  # Number of shots per task
    'n_eval_tasks': 10,  # Number of different tasks to evaluate on
    'inner_lr': params['inner_lr'],  # Just use the default parameters for evaluating
    'tau': params['tau'],
    'gamma': params['gamma'],
}

benchmark = "ML1"  # Choose between ML1, ML10, ML45
workers = 1

cuda = False

wandb = False


def make_env(test=False):
    # Set a specific task or left empty to train on all available tasks
    task = 'pick-place-v1' if benchmark == "ML1" else ""

    # Fetch one of the ML benchmarks from metaworld
    benchmark_env = getattr(mtwrld, benchmark)

    if test:
        env = benchmark_env.get_test_tasks(task)
    else:
        env = benchmark_env.get_train_tasks(task)

    env = ch.envs.ActionSpaceScaler(env)
    env.seed(params['seed'])
    env.set_task(env.sample_tasks(1)[0])
    env = ch.envs.Torch(env)
    return env


def collect_episodes(model, task, n_episodes, n_steps=None):
    # If user doesn't provide predefined horizon length, use maximum horizon set by meta-world
    if n_steps is None:
        n_steps = task.active_env.max_path_length

    episodes = ch.ExperienceReplay()
    # Collect multiple episodes per task
    for episode in range(n_episodes):
        # We manually define the horizon for 150 steps following the metaworld paper
        episode = task.run(model, steps=n_steps)
        episodes.__iadd__(episode)
        # Due to the current meta-world build, when an episode reaches the end of the horizon it doesn't
        # automatically reset the environment so we have to manually reset it
        # (see https://github.com/rlworkgroup/metaworld/issues/60)
        task.env.reset()
    return episodes


class MamlRL(Experiment):

    def __init__(self):
        super(MamlRL, self).__init__("maml", "metaworld", params, path="rl_results/", use_wandb=wandb)

        device = torch.device('cpu')

        random.seed(self.params['seed'])
        np.random.seed(self.params['seed'])
        torch.manual_seed(self.params['seed'])

        env = make_env()

        if cuda and torch.cuda.device_count():
            print(f"Running on {torch.cuda.get_device_name(0)}")
            torch.cuda.manual_seed(self.params['seed'])
            device = torch.device('cuda')

        self.run(env, device)

    def run(self, env, device):

        baseline = ch.models.robotics.LinearValue(env.state_size, env.action_size)
        # baseline.to(device)
        policy = DiagNormalPolicy(env.state_size, env.action_size)
        policy.to(device)

        self.log_model(policy, device, input_shape=(1, env.state_size))  # Input shape is specific to dataset

        t = trange(self.params['num_iterations'], desc="Iteration", position=0)
        try:
            for iteration in t:

                iter_reward = 0
                iter_replays = []
                iter_policies = []

                task_list = env.sample_tasks(self.params['meta_batch_size'])

                for task_i in trange(len(task_list), leave=False, desc="Task", position=0):
                    task = task_list[task_i]

                    clone = deepcopy(policy)
                    env.set_task(task)
                    env.reset()

                    task = ch.envs.Runner(env)
                    task_replay = []

                    # Adapt
                    for step in range(self.params['adapt_steps']):
                        train_episodes = collect_episodes(clone, task, self.params['adapt_batch_size'])
                        task_replay.append(train_episodes)
                        clone = fast_adapt_trpo_a2c(clone, train_episodes, baseline,
                                                    self.params['inner_lr'], self.params['gamma'], self.params['tau'],
                                                    first_order=True, device=device)

                    # Compute validation Loss
                    valid_episodes = collect_episodes(clone, task, self.params['adapt_batch_size'])
                    task_replay.append(valid_episodes)

                    iter_reward += valid_episodes.reward().sum().item() / self.params['adapt_batch_size']
                    iter_replays.append(task_replay)
                    iter_policies.append(clone)

                validation_reward = iter_reward / self.params['meta_batch_size']
                metrics = {'validation_reward': validation_reward}

                t.set_postfix(metrics)
                self.log_metrics(metrics)

                meta_optimize(self.params, policy, baseline, iter_replays, iter_policies, device)

                if iteration % self.params['save_every'] == 0:
                    self.save_model_checkpoint(policy, str(iteration))

        # Support safely manually interrupt training
        except KeyboardInterrupt:
            print('\nManually stopped training! Start evaluation & saving...\n')
            self.logger['manually_stopped'] = True
            self.params['num_iterations'] = iteration

        self.save_model(policy)

        self.logger['elapsed_time'] = str(round(t.format_dict['elapsed'], 2)) + ' sec'
        # Evaluate on new test tasks
        self.logger['test_reward'] = evaluate(policy, baseline, device)
        self.log_metrics({'test_reward': self.logger['test_reward']})
        self.save_logs_to_file()


def evaluate(policy, baseline, device):
    env = make_env(test=True)
    eval_task_list = env.sample_tasks(eval_params['n_eval_tasks'])

    tasks_reward = 0.0

    for i, task in enumerate(eval_task_list):
        clone = deepcopy(policy)
        env.set_task(task)
        env.reset()
        task = ch.envs.Runner(env)

        # Adapt
        for step in range(eval_params['adapt_steps']):
            adapt_episodes = task.run(clone, steps=150)
            clone = fast_adapt_trpo_a2c(clone, adapt_episodes, baseline, eval_params['adapt_lr'],
                                        eval_params['gamma'], eval_params['tau'], first_order=True, device=device)
            task.env.reset()

        eval_episodes = task.run(clone, steps=150)

        task_reward = eval_episodes.reward().sum().item()
        print(f"Reward for task {i} : {task_reward}")
        tasks_reward += task_reward

    final_eval_reward = tasks_reward / eval_params['n_eval_tasks']
    return final_eval_reward


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MAML on RL tasks')

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

    MamlRL()
