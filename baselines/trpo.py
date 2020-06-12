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
from core_functions.rl import evaluate_trpo


params = {
    # TRPO parameters
    'backtrack_factor': 0.5,
    'ls_max_steps': 15,
    'trpo_steps': 10,
    'max_kl': 0.01,
    # Common parameters
    'batch_size': 20,
    'n_episodes': 10,
    'lr': 0.05,
    'max_path_length': 150,
    'activation': 'tanh',  # for MetaWorld use tanh, others relu
    'tau': 1.0,
    'gamma': 0.99,
    # Other parameters
    'num_iterations': 1000,
    'save_every': 25,
    'seed': 42,
    # For evaluation
    'n_tasks': 10,
    'adapt_steps': 3,
    'adapt_batch_size': 20,
    'inner_lr': 0.1}

# Environments:
#   - Particles2D-v1
#   - AntDirection-v1
#   - ML1_reach-v1, ML1_pick-place-v1, ML1_push-v1
#   - ML10, ML45

env_name = 'ML1_push-v1'

workers = 5

wandb = False


class TRPO(Experiment):

    def __init__(self):
        super(TRPO, self).__init__('trpo', env_name, params, path='trpo_results/', use_wandb=wandb)

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
                iter_loss = 0.0

                task_list = env.sample_tasks(self.params['batch_size'])
                for task_i in trange(len(task_list), leave=False, desc='Task', position=0):
                    task = task_list[task_i]
                    env.set_task(task)
                    env.reset()
                    task = ch.envs.Runner(env)

                    episodes = task.run(policy, episodes=params['n_episodes'])
                    task_reward = episodes.reward().sum().item() / params['n_episodes']

                    # Calculate loss & fit value function & update policy
                    loss = trpo_update(episodes, policy, baseline, self.params)

                    iter_reward += task_reward
                    iter_loss += loss.item()
                    print(f'Task {task_i}: Rew: {task_reward} | loss: {loss.item()}')

                # Log
                average_return = iter_reward / self.params['batch_size']
                av_loss = iter_loss / self.params['batch_size']
                metrics = {'average_return': average_return,
                           'loss': av_loss}

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
        self.logger['test_reward'] = evaluate_trpo(env_name, policy, baseline, params)
        self.log_metrics({'test_reward': self.logger['test_reward']})
        self.save_logs_to_file()


def trpo_update(episodes, policy, baseline, prms):
    """
    Inspired by cherry-rl/examples/bsuite/trpo_v_random.py
    """
    new_loss = 0.0
    old_policy = deepcopy(policy)
    for step in range(prms['trpo_steps']):
        states = episodes.state()
        actions = episodes.action()
        rewards = episodes.reward()
        dones = episodes.done()
        next_states = episodes.next_state()
        returns = ch.td.discount(prms['gamma'], rewards, dones)
        baseline.fit(states, returns)
        values = baseline(states)
        next_values = baseline(next_states)

        # Compute KL
        with torch.no_grad():
            old_density = old_policy.density(states)
        new_density = policy.density(states)
        kl = torch.distributions.kl_divergence(old_density, new_density).mean()

        # Compute surrogate loss
        old_log_probs = old_density.log_prob(actions).mean(dim=1, keepdim=True)
        new_log_probs = new_density.log_prob(actions).mean(dim=1, keepdim=True)
        bootstraps = values * (1.0 - dones) + next_values * dones
        advantages = ch.pg.generalized_advantage(prms['gamma'], prms['tau'], rewards,
                                                 dones, bootstraps, torch.zeros(1))
        advantages = ch.normalize(advantages).detach()
        surr_loss = ch.algorithms.trpo.policy_loss(new_log_probs, old_log_probs, advantages)

        # Compute the update
        grad = torch.autograd.grad(surr_loss, policy.parameters(), retain_graph=True)

        fvp = ch.algorithms.trpo.hessian_vector_product(kl, policy.parameters())
        grad = torch.nn.utils.parameters_to_vector(grad).detach()
        step = ch.algorithms.trpo.conjugate_gradient(fvp, grad)
        lagrange_mult = 0.5 * torch.dot(step, fvp(step)) / prms['max_kl']
        step = step / lagrange_mult
        step_ = [torch.zeros_like(p.data) for p in policy.parameters()]
        torch.nn.utils.vector_to_parameters(step, step_)
        step = step_

        #  Line-search
        for ls_step in range(prms['ls_max_steps']):
            stepsize = prms['backtrack_factor'] ** ls_step
            clone = deepcopy(policy)
            for c, u in zip(clone.parameters(), step):
                c.data.add_(u.data, alpha=-stepsize)
            new_density = clone.density(states)
            new_kl = torch.distributions.kl_divergence(old_density, new_density).mean()
            new_log_probs = new_density.log_prob(actions).mean(dim=1, keepdim=True)
            new_loss = ch.algorithms.trpo.policy_loss(new_log_probs, old_log_probs, advantages)
            if new_loss < surr_loss and new_kl < prms['max_kl']:
                for p, c in zip(policy.parameters(), clone.parameters()):
                    p.data[:] = c.data[:]
                break

    return new_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TRPO on RL tasks')

    parser.add_argument('--env', type=str, default=env_name, help='Pick an environment')
    parser.add_argument('--lr', type=float, default=params['lr'], help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=params['batch_size'], help='Batch size')
    parser.add_argument('--num_iterations', type=int, default=params['num_iterations'], help='Number of epochs')
    parser.add_argument('--save_every', type=int, default=params['save_every'], help='Interval to save model')
    parser.add_argument('--seed', type=int, default=params['seed'], help='Seed')

    args = parser.parse_args()

    params['lr'] = args.lr
    params['batch_size'] = args.batch_size
    params['num_iterations'] = args.num_iterations
    params['save_every'] = args.save_every
    params['seed'] = args.seed

    TRPO()
