#!/usr/bin/env python3

import argparse
import random
import numpy as np
from tqdm import trange

import torch
import learn2learn as l2l

from procgen import ProcgenEnv
from baselines.common.vec_env import (VecExtractDictObs, VecMonitor, VecNormalize)

from utils import *
from core_functions.policies import ActorCritic
from core_functions.a2c_ppo2 import adapt_a2c, compute_a2c_loss
from misc_scripts import run_cl_rl_exp

from sampler_ppo2 import Sampler

# updates = total timesteps / batch
# 1.000.000 serial timesteps takes around 3hours
# 25.000.000 timesteps for easy difficulty
# 200.000.000 timesteps for hard difficulty

params = {
    # Inner loop parameters
    "n_adapt_steps": 3,  # Number of inner loop iterations
    "inner_lr": 0.005,  # Default: 0.1
    # Outer loop parameters
    "outer_lr": 0.0005,
    "backtrack_factor": 0.5,
    "ls_max_steps": 15,
    "max_kl": 0.01,
    # Common parameters
    "tau": 0.95,
    "gamma": 0.99,

    # Environment params

    # easy or hard: only affects the visual variance between levels
    "distribution_mode": "easy",
    # Number of environments OF THE SAME LEVEL to run in parallel -> 32envs ~7gb RAM (Original was 64)
    "n_envs": 1,
    # 0-unlimited, 1-debug. For generalization: 200-easy, 500-hard
    "n_levels": 1,
    # Number of different levels the agent should train on in an iteration (="ways") prev. meta_batch_size
    # If n_levels = 1, then it a matter of how often to use inner loop adaptation
    "n_tasks_per_iter": 3,
    # Number of total timesteps performed
    "n_timesteps": 5_000_000,
    # Number of runs on the same level for one inner iteration (="shots") prev. adapt_batch_size
    # "n_episodes_per_task": 100,  # Currently not in use. So just one episode per environment
    # Rollout length of each of the above runs
    "n_steps_per_episode": 256,
    # Split the batch in mini batches for faster adaptation
    # "n_steps_per_mini_batch": 256,
    # Model params
    "save_every": 50,
    "seed": 42}

# Timesteps performed per task in each iteration
params['steps_per_task'] = int(params['n_steps_per_episode'] * params['n_envs'])
# Split the episode in mini batches
# params['n_mini_batches'] = int(params['steps_per_task'] / params['n_steps_per_mini_batch'])
# iters = outer updates: 64envs, 25M -> 1.525, 200M-> 12.207
params['n_iters'] = int(params['n_timesteps'] // params['steps_per_task'])
# Total timesteps performed per task (if task==1, then total timesteps==total steps per task)
params['total_steps_per_task'] = int(params['steps_per_task'] * params['n_iters'])

network = [64, 128, 128]

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
start_level = params['seed']

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

        venv = ProcgenEnv(num_envs=self.params['n_envs'], env_name=env_name, num_levels=self.params['n_levels'],
                          start_level=start_level, distribution_mode=self.params['distribution_mode'],
                          paint_vel_info=True)

        venv = VecExtractDictObs(venv, "rgb")
        venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
        venv = VecNormalize(venv=venv, ob=False)

        self.run(venv, device)

    def run(self, env, device):

        observ_space = env.observation_space.shape[::-1]
        observ_size = len(observ_space)
        action_space = env.action_space.n

        final_pixel_dim = int(64 / (np.power(2, len(network))))
        fc_neurons = network[-1] * final_pixel_dim * final_pixel_dim

        # Initialize models
        policy = ActorCritic(observ_size, action_space, network)
        policy.to(device)
        meta_actor = l2l.algorithms.MAML(policy.actor, lr=self.params['inner_lr'])
        meta_critic = l2l.algorithms.MAML(policy.critic, lr=self.params['inner_lr'])

        meta_learner = l2l.algorithms.MAML(policy, lr=self.params['inner_lr'])

        actor_optimiser = torch.optim.Adam(policy.actor.parameters(), lr=self.params['outer_lr'])
        critic_optimiser = torch.optim.Adam(policy.critic.parameters(), lr=self.params['outer_lr'])

        self.log_model(policy.actor, device, input_shape=observ_space)  # Input shape is specific to dataset

        # Sampler uses policy.eval() which turns off training to sample the actions (same as torch.no_grad?)
        sampler = Sampler(env=env, model=policy, num_steps=self.params['n_steps_per_episode'],
                          gamma_coef=params['gamma'], lambda_coef=params['tau'],
                          device=device, num_envs=self.params['n_envs'])

        t_iter = trange(self.params['n_iters'], desc="Iteration", position=0)
        try:
            for iteration in t_iter:

                tr_iter_reward = 0.0
                val_iter_reward = 0.0

                # Pick a level out of "n_levels"
                t_task = trange(self.params['n_tasks_per_iter'], leave=False, desc="Task", position=0)
                for task in t_task:
                    a2c_learner = meta_learner.clone()
                    actor_learner = meta_actor.clone()
                    critic_learner = meta_critic.clone()

                    # Sample training episodes (32envs, 256 length takes less than 1GB)
                    tr_ep_samples, tr_ep_infos = sampler.run()

                    # Adapt
                    policy_loss_tr, value_loss_tr = adapt_a2c(tr_ep_samples, a2c_learner, actor_learner, critic_learner,
                                                              self.params['n_adapt_steps'], self.params['inner_lr'],
                                                              device)

                    # Compute validation Loss
                    val_ep_samples, val_ep_info = sampler.run()
                    policy_loss_v, value_loss_v = compute_a2c_loss(val_ep_samples, device)

                    # Average reward across tasks
                    tr_task_reward = tr_ep_samples["rewards"].sum().item() / self.params['n_envs']
                    tr_iter_reward += tr_task_reward

                    val_task_reward = val_ep_samples["rewards"].sum().item() / self.params['n_envs']
                    val_iter_reward += val_task_reward

                    policy_loss_v += policy_loss_v
                    value_loss_v += value_loss_v

                    # print(f"\nTrain reward of task {task} is {tr_task_reward}")
                    # print(f"Valid reward of task {task} is {val_task_reward}")

                    task_metrics = {f'av_train_task_{task}': tr_task_reward,
                                    f'av_valid_task_{task}': val_task_reward}
                    t_task.set_postfix(task_metrics)

                # Tasks rewards
                tr_iter_reward = tr_iter_reward / self.params['n_tasks_per_iter']
                val_iter_reward = val_iter_reward / self.params['n_tasks_per_iter']

                # Tasks loss
                policy_loss_tr = policy_loss_tr / self.params['n_tasks_per_iter']
                value_loss_tr = value_loss_tr / self.params['n_tasks_per_iter']
                policy_loss_v = policy_loss_v / self.params['n_tasks_per_iter']
                value_loss_v = value_loss_v / self.params['n_tasks_per_iter']

                # print(f"\nAverage train reward: {tr_iter_reward}")
                # print(f"Average valid reward: {val_iter_reward}")

                # Optimize actor by updating the PPO loss
                actor_optimiser.zero_grad()
                policy_loss_v.requires_grad = True
                policy_loss_v.backward()
                actor_optimiser.step()

                # Fit the value function by regression on MSE
                critic_optimiser.zero_grad()
                value_loss_v.requires_grad = True
                value_loss_v.backward()
                critic_optimiser.step()

                step = iteration * params['n_steps_per_episode'] * params['n_tasks_per_iter'] * params['n_envs']
                metrics = {'step': step,
                           'tr_iter_reward': tr_iter_reward,
                           'tr_actor_loss': policy_loss_tr,
                           'tr_critic_loss': value_loss_tr,
                           'val_iter_reward': val_iter_reward,
                           'val_critic_loss': value_loss_v.item(),
                           'val_actor_loss': policy_loss_v.item()}
                # 'val_iter_actor_loss': policy_loss_v}

                t_iter.set_postfix({'step': step,
                                    'tr_iter_reward': tr_iter_reward,
                                    'tr_actor_loss': policy_loss_tr,
                                    'tr_critic_loss': value_loss_tr})
                self.log_metrics(metrics, step)

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
            run_cl_rl_exp(self.model_path, env, policy, None, cl_params=cl_params)


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
