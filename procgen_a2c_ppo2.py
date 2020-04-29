#!/usr/bin/env python3

import argparse
import random
import numpy as np
from tqdm import trange

import torch
import cherry as ch
import learn2learn as l2l

from procgen import ProcgenEnv
from baselines.common.vec_env import (VecExtractDictObs, VecMonitor, VecNormalize)

from utils import *
from core_functions.policies import ActorCritic
from core_functions.a2c_ppo2 import compute_a2c_loss, compute_adv_ret

from sampler_ppo2 import Sampler

# updates = total timesteps / batch
# 1.000.000 serial timesteps takes around 3hours
# 25.000.000 timesteps for easy difficulty
# 200.000.000 timesteps for hard difficulty

params = {
    "ppo_epochs": 3,
    "lr": 0.0005,  # Default: 0.1
    "backtrack_factor": 0.5,
    "ls_max_steps": 15,
    "max_kl": 0.01,
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
    # "n_tasks_per_iter": 20,
    # Number of total timesteps performed
    "n_timesteps": 5_000_000,
    # Number of runs on the same level for one inner iteration (="shots") prev. adapt_batch_size
    # "n_episodes_per_task": 100,  # Currently not in use. So just one episode per environment
    # Rollout length of each of the above runs
    "n_steps_per_episode": 256,
    # Split the batch in mini batches for faster adaptation
    # "n_steps_per_mini_batch": 256,
    # Model params
    "save_every": 100,
    "seed": 42}

# Timesteps performed per task in each iteration
params['steps_per_task'] = int(params['n_steps_per_episode'] * params['n_envs'])
# Split the episode in mini batches
# params['n_mini_batches'] = int(params['steps_per_task'] / params['n_steps_per_mini_batch'])
# iters = outer updates: 64envs, 25M -> 1.525, 200M-> 12.207
params['n_iters'] = int(params['n_timesteps'] // params['steps_per_task'])
# Total timesteps performed per task (if task==1, then total timesteps==total steps per task)
params['total_steps_per_task'] = int(params['steps_per_task'] * params['n_iters'])

network = [64, 128, 256, 256]

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
log_validation = False

wandb = False


class PPO2Procgen(Experiment):

    def __init__(self):
        super(PPO2Procgen, self).__init__("ppo2", env_name, params, path="rl_results/", use_wandb=wandb)

        device = torch.device('cpu')
        random.seed(self.params['seed'])
        np.random.seed(self.params['seed'])
        torch.manual_seed(self.params['seed'])

        if cuda and torch.cuda.device_count():
            print(f"Running on {torch.cuda.get_device_name(0)}")
            torch.cuda.manual_seed(self.params['seed'])
            device = torch.device('cuda')

        self.run(device)

    def make_procgen_env(self):
        venv = ProcgenEnv(num_envs=self.params['n_envs'], env_name=env_name, num_levels=self.params['n_levels'],
                          start_level=start_level, distribution_mode=self.params['distribution_mode'],
                          paint_vel_info=True)

        venv = VecExtractDictObs(venv, "rgb")
        venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
        venv = VecNormalize(venv=venv, ob=False)
        return venv

    def run(self, device):

        env = self.make_procgen_env()

        observ_space = env.observation_space.shape[::-1]
        observ_size = len(observ_space)
        action_space = env.action_space.n

        final_pixel_dim = int(64 / (np.power(2, len(network))))
        fc_neurons = network[-1] * final_pixel_dim * final_pixel_dim

        # Initialize models
        policy = ActorCritic(observ_size, action_space, network)
        policy.to(device)

        """Single optimiser"""
        # policy_optimiser = torch.optim.Adam(policy.parameters(), lr=self.params['lr'])
        """Separate optimisers"""
        actor_optimiser = torch.optim.Adam(policy.actor.parameters(), lr=self.params['lr'])
        critic_optimiser = torch.optim.Adam(policy.critic.parameters(), lr=self.params['lr'])

        self.log_model(policy.actor, device, input_shape=observ_space)  # Input shape is specific to dataset

        t_iter = trange(self.params['n_iters'], desc="Iteration", position=0)
        try:

            for iteration in t_iter:

                env = self.make_procgen_env()
                sampler = Sampler(env=env, model=policy, num_steps=self.params['n_steps_per_episode'],
                                  gamma_coef=params['gamma'], lambda_coef=params['tau'],
                                  device=device, num_envs=self.params['n_envs'])

                # Sample training episodes (32envs, 256 length takes less than 1GB)
                tr_ep_samples, tr_ep_infos = sampler.run(no_grad=False, with_adv_ret=False)
                tr_rewards = tr_ep_samples['rewards'].sum().item() / self.params['n_envs']

                # n_envs = params['n_envs']
                # tr_rewards_per_env = [tr_ep_samples['rewards'].reshape(-1, n_envs)[e] for e in range(n_envs)]

                advantages, returns = compute_adv_ret(tr_ep_samples, self.params['gamma'], self.params['tau'], device)
                tr_ep_samples["advantages"] = advantages.cpu().detach().numpy()
                tr_ep_samples["returns"] = returns.cpu().detach().numpy()

                policy_loss, value_loss = compute_a2c_loss(tr_ep_samples, device)

                loss = policy_loss + (value_l_weight * value_loss)

                # Optimize
                policy_optimiser.zero_grad()
                loss.requires_grad = True
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                policy_optimiser.step()

                # Average reward across tasks
                tr_iter_reward = tr_rewards
                tr_actor_loss = policy_loss.item()
                tr_critic_loss = value_loss.item()

                # Compute validation Loss
                val_ep_samples, val_ep_info = sampler.run()
                val_iter_reward = val_ep_samples["rewards"].sum().item() / self.params['n_envs']
                val_actor_loss, val_critic_loss = compute_a2c_loss(val_ep_samples, device)

                step = iteration * params['n_steps_per_episode'] * params['n_envs']
                metrics = {'tr_iter_reward': tr_iter_reward,
                           'tr_actor_loss': tr_actor_loss,
                           'tr_critic_loss': tr_critic_loss}

                if log_validation:
                    # Compute validation loss without storing gradients & calculate advantages needed for the loss
                    val_ep_samples, val_ep_info = sampler.run(no_grad=True, with_adv_ret=True)
                    val_actor_loss, val_critic_loss = compute_a2c_loss(val_ep_samples, device)

                    # Update metrics with validation data
                    val_iter_reward = val_ep_samples["rewards"].sum().item() / self.params['n_envs']
                    metrics.update({'val_iter_reward': val_iter_reward,
                                    'val_actor_loss': val_actor_loss.item(),
                                    'val_critic_loss': val_critic_loss.item()})

                self.log_metrics(metrics, step)
                t_iter.set_postfix({"Steps": step,
                                    "tr_iter_reward": tr_iter_reward,
                                    'tr_actor_loss': tr_actor_loss,
                                    'tr_critic_loss': tr_critic_loss})

                if iteration % self.params['save_every'] == 0:
                    self.save_model_checkpoint(policy, str(iteration))

        # Support safely manually interrupt training
        except KeyboardInterrupt:
            print('\nManually stopped training! Start evaluation & saving...\n')
            self.logger['manually_stopped'] = True
            self.params['num_iterations'] = iteration

        self.save_model(policy)

        self.logger['elapsed_time'] = str(round(t_iter.format_dict['elapsed'], 2)) + ' sec'
        self.save_logs_to_file()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PPO2 on a Procgen env')

    parser.add_argument('--env', type=str, default=env_name, help='Pick an environment')
    parser.add_argument('--lr', type=float, default=params['lr'], help='lr')
    parser.add_argument('--n_iters', type=int, default=params['n_iters'], help='Number of epochs')
    parser.add_argument('--save_every', type=int, default=params['save_every'], help='Interval to save model')
    parser.add_argument('--seed', type=int, default=params['seed'], help='Seed')

    args = parser.parse_args()

    params['lr'] = args.lr
    params['n_iters'] = args.n_iters
    params['save_every'] = args.save_every
    params['seed'] = args.seed

    PPO2Procgen()
