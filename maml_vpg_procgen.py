#!/usr/bin/env python3

import argparse
import random
import numpy as np
from tqdm import trange
from copy import deepcopy

import learn2learn as l2l
import torch
import cherry as ch

from mpi4py import MPI
from procgen import ProcgenEnv
from baselines import logger
from baselines.common.mpi_util import setup_mpi_gpus
from baselines.common.vec_env import (VecExtractDictObs, VecMonitor, VecNormalize)

from utils import *
from core_functions.policies import DiagNormalPolicyCNN
from core_functions.rl import maml_vpg_a2c_loss, evaluate
from misc_scripts import run_cl_rl_exp

from sampler import Sampler

# updates = total timesteps / batch
# 1.000.000 serial timesteps takes around 3hours
# 25.000.000 timesteps for easy difficulty
# 200.000.000 timesteps for hard difficulty

params = {
    # Inner loop parameters
    "n_adapt_steps": 3,  # Number of inner loop iterations
    "inner_lr": 0.1,  # Default: 0.1
    # Outer loop parameters
    "outer_lr": 0.1,
    "backtrack_factor": 0.5,
    "ls_max_steps": 15,
    "max_kl": 0.01,
    # Common parameters
    "tau": 0.95,
    "gamma": 0.99,

    # Environment params

    # Total timesteps:
    # nbatch = nenvs * nsteps
    # nbatch_train = nbatch // nminibatches
    # nupdates = total_timesteps//nbatch
    #
    # batch_size = n_envs * n_steps

    # easy or hard: only affects the visual variance between levels
    "distribution_mode": "easy",
    # Number of environments OF THE SAME LEVEL to run in parallel -> 32envs ~7gb RAM (Original was 64)
    "n_envs": 32,
    # 0-unlimited, 1-debug. For generalization: 200-easy, 500-hard
    "n_levels": 0,
    # iters = outer updates = epochs PPO: 64envs, 25M -> 1.525, 200M-> 12.207
    # We could have more epochs in one iterations, but for simplicity now we make it the same
    "n_iters": 500,
    # Number of different levels the agent should train on in an iteration (="ways") prev. meta_batch_size
    "n_tasks_per_iter": 1,
    # Number of runs on the same level for one inner iteration (="shots") prev. adapt_batch_size
    # "n_episodes_per_task": 100,  # Currently not in use. So just one episode per environment
    # Rollout length of each of the above runs
    "n_steps_per_episode": 256,
    # One of the workers will test, define how often (train & test in parallel)
    "test_worker_interval": 0,
    # Model params
    "save_every": 25,
    "seed": 42}

network = [32, 64, 64]

eval_params = {
    'n_eval_adapt_steps': 5,  # Number of steps to adapt to a new task
    'n_eval_episodes': 10,  # Number of shots per task
    'n_eval_tasks': 10,  # Number of different tasks to evaluate on
    'inner_lr': params['inner_lr'],  # Just use the default parameters for evaluating
    'tau': params['tau'],
    'gamma': params['gamma'],
}
cl_test = True
cl_params = {
    "adapt_steps": 10,
    "adapt_batch_size": 10,  # shots
    "inner_lr": 0.3,
    "gamma": 0.99,
    "tau": 1.0,
    "n_tasks": 5
}

# Potential games:
#   caveflyer
#   coinrun
#   dodgeball
#   maze: fast rewards
#   starpilot: fast rewards
#   bigfish: fast rewards

env_name = "starpilot"
start_level = 0  # ???

cuda = True

wandb = False


class MamlRL(Experiment):

    def __init__(self):
        super(MamlRL, self).__init__("maml", env_name, params, path="rl_results/", use_wandb=wandb)

        device = torch.device('cpu')
        random.seed(self.params['seed'])
        np.random.seed(self.params['seed'])
        torch.manual_seed(self.params['seed'])

        if cuda and torch.cuda.device_count():
            print(f"Running on {torch.cuda.get_device_name(0)}")
            torch.cuda.manual_seed(self.params['seed'])
            device = torch.device('cuda')

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        is_test_worker = False
        if params['test_worker_interval'] > 0:
            is_test_worker = comm.Get_rank() % params['test_worker_interval'] == (params['test_worker_interval'] - 1)

        mpi_rank_weight = 0 if is_test_worker else 1
        n_levels = 0 if is_test_worker else self.params['n_levels']

        log_comm = comm.Split(1 if is_test_worker else 0, 0)
        format_strs = ['csv', 'stdout'] if log_comm.Get_rank() == 0 else []

        logger.configure(dir=self.model_path, format_strs=format_strs)
        logger.info(f"Creating {self.params['n_envs']} {env_name} environments")

        venv = ProcgenEnv(num_envs=self.params['n_envs'], env_name=env_name, num_levels=n_levels,
                          start_level=start_level, distribution_mode=self.params['distribution_mode'])

        venv = VecExtractDictObs(venv, "rgb")

        venv = VecMonitor(venv=venv, filename=None, keep_buf=100, )

        venv = VecNormalize(venv=venv, ob=False)

        setup_mpi_gpus()

        self.run(venv, device)

    def run(self, env, device):

        observ_space = env.observation_space.shape[::-1]
        observ_size = len(observ_space)
        observ_space_flat = observ_space[0] * observ_space[1] * observ_space[2]
        action_space = env.action_space.n + 1

        final_pixel_dim = int(64 / (np.power(2, len(network))))
        fc_neurons = network[-1] * final_pixel_dim * final_pixel_dim

        samples_across_workers = self.params['n_steps_per_episode'] * self.params['n_envs']

        baseline = ch.models.robotics.LinearValue(observ_space_flat, action_space)
        baseline.to(device)
        policy = DiagNormalPolicyCNN(observ_size, action_space, network=network)
        policy.to(device)
        meta_learner = l2l.algorithms.MAML(policy, lr=self.params['inner_lr'])
        opt = torch.optim.Adam(policy.parameters(), lr=self.params['inner_lr'])

        self.log_model(policy, device, input_shape=observ_space)  # Input shape is specific to dataset
        print("FC Neurons: ", fc_neurons)

        # Sampler uses policy.eval() which turns off training to sample the actions
        sampler = Sampler(env=env, model=policy, num_steps=self.params['n_steps_per_episode'],
                          gamma_coef=params['gamma'], lambda_coef=params['tau'],
                          device=device, num_envs=self.params['n_envs'])

        t_iter = trange(self.params['n_iters'], desc="Iteration", position=0)
        try:
            for iteration in t_iter:

                tr_iter_reward = 0.0
                tr_iter_loss = 0.0
                val_iter_reward = 0.0
                val_iter_loss = 0.0

                # Pick a new level out of
                t_task = trange(self.params['n_tasks_per_iter'], leave=False, desc="Task", position=0)
                for task in t_task:

                    tr_task_reward = 0.0
                    tr_task_loss = 0.0
                    learner = meta_learner.clone()

                    for step in range(self.params['n_adapt_steps']):
                        # Sample training episodes (64envs, 256 length takes less than 1GB)
                        tr_ep_samples, tr_ep_infos = sampler.run()

                        # Adapt
                        loss = maml_vpg_a2c_loss(tr_ep_samples, learner, baseline,
                                                 self.params['gamma'], self.params['tau'], device)
                        learner.adapt(loss)

                        # Metrics
                        tr_task_loss += loss
                        tr_task_reward += tr_ep_samples["rewards"].sum().item() / samples_across_workers
                        print(f"Train reward of task {task} is {tr_task_reward}")

                    # Compute validation Loss
                    val_ep_samples, val_ep_info = sampler.run()
                    loss = maml_vpg_a2c_loss(val_ep_samples, learner, baseline,
                                             self.params['gamma'], self.params['tau'], device)

                    # Validation reward & loss for task i
                    val_task_reward = val_ep_samples["rewards"].sum().item() / samples_across_workers

                    # Average train reward / loss across tasks
                    tr_iter_reward += tr_task_reward / self.params['n_adapt_steps']
                    tr_iter_loss = tr_task_loss / self.params['n_adapt_steps']

                    # Average valid reward / loss across tasks
                    val_iter_reward += val_task_reward
                    val_iter_loss += loss

                    print(f"Train reward of task {task} is {tr_task_reward}")
                    print(f"Valid reward of task {task} is {val_task_reward}")

                    task_metrics = {f'av_train_task_{task}': tr_task_reward,
                                    f'av_valid_task_{task}':val_task_reward}
                    t_task.set_postfix(task_metrics)

                tr_iter_reward = tr_iter_reward / self.params['n_tasks_per_iter']
                tr_iter_loss = tr_iter_loss / self.params['n_tasks_per_iter']
                val_iter_reward = val_iter_reward / self.params['n_tasks_per_iter']
                val_iter_loss = val_iter_loss / self.params['n_tasks_per_iter']

                metrics = {'tr_iter_reward': tr_iter_reward,
                           'tr_iter_loss': tr_iter_loss,
                           'val_iter_reward': val_iter_reward,
                           'val_iter_loss': val_iter_loss}

                t_iter.set_postfix(metrics)
                self.log_metrics(metrics)

                opt.zero_grad()
                val_iter_loss.backward()
                opt.step()

                if iteration % self.params['save_every'] == 0:
                    self.save_model_checkpoint(policy, str(iteration))

        # Support safely manually interrupt training
        except KeyboardInterrupt:
            print('\nManually stopped training! Start evaluation & saving...\n')
            self.logger['manually_stopped'] = True
            self.params['num_iterations'] = iteration

        self.save_model(policy)

        self.logger['elapsed_time'] = str(round(t_iter.format_dict['elapsed'], 2)) + ' sec'
        # Evaluate on new test tasks
        # self.logger['test_reward'] = evaluate(env, policy, baseline, eval_params)
        # self.log_metrics({'test_reward': self.logger['test_reward']})
        self.save_logs_to_file()

        if cl_test:
            print("Running Continual Learning experiment...")
            run_cl_rl_exp(self.model_path, env, policy, baseline, cl_params=cl_params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MAML on RL tasks')

    parser.add_argument('--env', type=str, default=env_name, help='Pick an environment')

    parser.add_argument('--outer_lr', type=float, default=params['outer_lr'], help='Outer lr')
    parser.add_argument('--inner_lr', type=float, default=params['inner_lr'], help='Inner lr')

    parser.add_argument('--n_iters', type=int, default=params['n_iters'], help='Number of epochs')
    parser.add_argument('--save_every', type=int, default=params['save_every'], help='Interval to save model')

    parser.add_argument('--seed', type=int, default=params['seed'], help='Seed')

    args = parser.parse_args()

    params['outer_lr'] = args.outer_lr
    params['inner_lr'] = args.inner_lr

    params['n_iters'] = args.n_iters
    params['save_every'] = args.save_every

    params['seed'] = args.seed

    MamlRL()
