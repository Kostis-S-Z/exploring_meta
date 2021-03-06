import cherry as ch
from cherry.algorithms import a2c, trpo, ppo
from cherry.pg import generalized_advantage

from learn2learn import clone_module, magic_box
from learn2learn.algorithms.maml import maml_update

import torch
from torch.distributions.kl import kl_divergence
from torch.nn.utils import parameters_to_vector, vector_to_parameters

import numpy as np
from copy import deepcopy
from collections import defaultdict

from utils import make_env
from core_functions.runner import Runner

device = torch.device('cpu')


ML10_train_task_names = {
    0: 'reach',
    1: 'push',
    2: 'pick-place',
    3: 'door-open',
    4: 'drawer-close',
    5: 'button-press',
    6: 'peg-insert-side',
    7: 'window-open',
    8: 'sweep',
    9: 'basketball',
}

ML10_eval_task_names = {
    0: 'drawer-open',
    1: 'door-close',
    2: 'shelf-place',
    3: 'sweep-into',
    4: 'lever-pull',
}


def set_device(dev):
    global device
    device = dev


def get_episode_values(episodes):
    states = episodes.state().to(device)
    actions = episodes.action().to(device)
    rewards = episodes.reward().to(device)
    dones = episodes.done().to(device)
    next_states = episodes.next_state().to(device)

    return states, actions, rewards, dones, next_states


def get_ep_successes(episodes, path_length):
    successes = 0
    # This works only if ExperienceReplay has a 'success' attribute!
    try:
        # Reshape ExperienceReplay so its easy to iterate
        success_matrix = episodes.success().reshape(path_length, -1).T
        for episode_suc in success_matrix:  # Iterate over each episode
            # if there was a success in that episode
            if 1. in episode_suc:  # Same as if True in [bool(s) for s in episode_suc]
                successes += 1
    except AttributeError:
        print('No success metric registered!')
        pass  # Returning 0! Implement success attribute if you want to count success of task
    return successes


def get_success_per_ep(episodes, path_length):
    # This works only if ExperienceReplay has a 'success' attribute!
    n_episodes = episodes.success().reshape(path_length, -1).shape[1]
    successes = {i: 0 for i in range(n_episodes)}
    success_step = {i: None for i in range(n_episodes)}
    try:
        # Reshape ExperienceReplay so its easy to iterate
        success_matrix = episodes.success().reshape(path_length, -1).T
        for i, episode_suc in enumerate(success_matrix):  # Iterate over each episode
            # if there was a success in that episode
            if 1. in episode_suc:  # Same as if True in [bool(s) for s in episode_suc]
                successes[i] = 1
                # Get the step it succeeded
                success_step[i] = np.argmax(episode_suc > 0.1)
    except AttributeError:
        print('No success metric registered!')
        pass  # Returning 0! Implement success attribute if you want to count success of task
    return successes, success_step


def compute_advantages(baseline, tau, gamma, rewards, dones, states, next_states, update_vf=True):
    returns = ch.td.discount(gamma, rewards, dones)

    if update_vf:
        baseline.fit(states, returns)

    values = baseline(states)
    next_values = baseline(next_states)
    bootstraps = values * (1.0 - dones) + next_values * dones
    next_value = torch.zeros(1, device=values.device)
    return generalized_advantage(tau=tau,
                                 gamma=gamma,
                                 rewards=rewards,
                                 dones=dones,
                                 values=bootstraps,
                                 next_value=next_value)


def sample_3_from_each_task(env):
    # Get a sufficient large enough pool of tasks (this is computationally negligible)
    task_list = env.sample_tasks(200)
    check = defaultdict(list)
    for i, k in enumerate(task_list):
        check[k['task']] += [i]

    final_task_list = []
    for key, val in check.items():
        for sample in val[:3]:
            final_task_list.append(task_list[sample])

    return final_task_list


def sample_explicit_task(env, task):
    # one liner to fetch the key from the dictionary based on the value
    if task in ML10_eval_task_names.values():
        task_index = list(ML10_eval_task_names.keys())[list(ML10_eval_task_names.values()).index(task)]
    else:
        task_index = list(ML10_train_task_names.keys())[list(ML10_train_task_names.values()).index(task)]
    # Get a sufficient large enough pool of tasks (this is computationally negligible)
    task_list = env.sample_tasks(100)
    for t in task_list:
        if t['task'] == task_index:
            return t
    return None


def evaluate(algo, env_name, policy, baseline, params, anil, render=False, test_on_train=False, each3=False):
    rewards_per_task = defaultdict(list)
    tasks_rewards = []
    tasks_success_rate = []

    if test_on_train:
        ml_task_names = ML10_train_task_names  # Meta-train tasks
    else:
        ml_task_names = ML10_eval_task_names  # Meta-testing tasks

    extra_info = True if 'ML' in env_name else False  # if env is metaworld, log success metric
    env = make_env(env_name, 1, params['seed'], test=(not test_on_train), max_path_length=params['max_path_length'])

    if each3:
        # Overwrite number of tasks and just sample 3 trials from each task
        eval_task_list = sample_3_from_each_task(env)
    elif isinstance(params['n_tasks'], str):
        eval_task_list = [sample_explicit_task(env, params['n_tasks'])]
    else:
        eval_task_list = env.sample_tasks(params['n_tasks'])

    for i, task in enumerate(eval_task_list):
        learner = deepcopy(policy)
        env.set_task(task)
        env.reset()
        env_task = Runner(env, extra_info=extra_info)

        # Adapt
        if algo == 'vpg':
            _, task_reward, task_suc = fast_adapt_vpg(env_task, learner, baseline, params, anil=anil, render=render)
        elif algo == 'ppo':
            _, task_reward, task_suc = fast_adapt_ppo(env_task, learner, baseline, params, render=render)
        else:
            learner, _, _, task_reward, task_suc = fast_adapt_trpo(env_task, learner, baseline, params, anil=anil,
                                                                   render=render)

        # Evaluate
        n_query_episodes = params['adapt_batch_size']
        query_episodes = env_task.run(learner, episodes=n_query_episodes, render=render)
        query_rew = query_episodes.reward().sum().item() / n_query_episodes
        query_success_rate = get_ep_successes(query_episodes, params['max_path_length']) / n_query_episodes

        tasks_rewards.append(query_rew)
        tasks_success_rate.append(query_success_rate)
        if extra_info:
            print(f'Task {i + 1} / {len(eval_task_list)}: {ml_task_names[task["task"]]} task'
                  f'\t {query_rew:.1f} rew | {query_success_rate * 100}% success rate')
            rewards_per_task[ml_task_names[task["task"]]] += [query_rew, query_success_rate]

    final_eval_reward = sum(tasks_rewards) / params['n_tasks']
    final_eval_suc = sum(tasks_success_rate) / params['n_tasks']

    if 'ML' in env_name:
        return tasks_rewards, final_eval_reward, final_eval_suc, rewards_per_task
    return tasks_rewards, final_eval_reward, final_eval_suc


""" VPG RELATED """


def weighted_cumsum(values, weights):
    for i in range(values.size(0)):
        values[i] += values[i - 1] * weights[i]
    return values


def vpg_a2c_loss(episodes, learner, baseline, gamma, tau, dice=False):
    # Get values to device
    states, actions, rewards, dones, next_states = get_episode_values(episodes)

    # Calculate loss between states and action in the network
    log_probs = learner.log_prob(states, actions)

    # Fit value function, compute advantages & normalize
    advantages = compute_advantages(baseline, tau, gamma, rewards, dones, states, next_states)

    # Calculate DiCE objective
    if dice:
        weights = torch.ones_like(dones)
        weights[1:].add_(dones[:-1], alpha=-1.0)
        weights /= dones.sum()
        cum_log_probs = weighted_cumsum(log_probs, weights)
        log_probs = magic_box(cum_log_probs)

    return a2c.policy_loss(log_probs, advantages)


def fast_adapt_vpg(task, learner, baseline, params, anil=False, first_order=False, render=False):
    # During inner loop adaptation we do not store gradients for the network body
    if anil:
        learner.module.turn_off_body_grads()

    for step in range(params['adapt_steps']):
        # Collect adaptation / support episodes
        support_episodes = task.run(learner, episodes=params['adapt_batch_size'], render=render)
        # Calculate loss & fit the value function
        adapt_loss = vpg_a2c_loss(support_episodes, learner, baseline, params['gamma'], params['tau'])
        # Adapt model based on the loss
        learner.adapt(adapt_loss, first_order=first_order, allow_unused=anil)  # In ANIL, not all parameters have grads

    # We need to include the body network parameters for the query set
    if anil:
        learner.module.turn_on_body_grads()

    # Collect evaluation / query episodes
    query_episodes = task.run(learner, episodes=params['adapt_batch_size'])
    # Calculate loss for the outer loop optimization
    valid_loss = vpg_a2c_loss(query_episodes, learner, baseline, params['gamma'], params['tau'])
    # Calculate the average reward of the evaluation episodes
    query_rew = query_episodes.reward().sum().item() / params['adapt_batch_size']
    query_success_rate = get_ep_successes(query_episodes, params['max_path_length']) / params['adapt_batch_size']

    return valid_loss, query_rew, query_success_rate


def evaluate_vpg(env, policy, baseline, eval_params, anil=False, render=False):
    return evaluate('vpg', env, policy, baseline, eval_params, anil, render=render)


""" PPO RELATED """


def fast_adapt_ppo(task, learner, baseline, params, anil=False, render=False):
    # During inner loop adaptation we do not store gradients for the network body
    if anil:
        learner.module.turn_off_body_grads()

    for step in range(params['adapt_steps']):
        # Collect adaptation / support episodes
        support_episodes = task.run(learner, episodes=params['adapt_batch_size'], render=render)

        # Get values to device
        states, actions, rewards, dones, next_states = get_episode_values(support_episodes)

        # Update value function & Compute advantages
        advantages = compute_advantages(baseline, params['tau'], params['gamma'], rewards, dones, states, next_states)
        advantages = ch.normalize(advantages, epsilon=1e-8).detach()
        # Calculate loss between states and action in the network
        with torch.no_grad():
            old_log_probs = learner.log_prob(states, actions)

        # Initialize inner loop PPO optimizer
        av_loss = 0.0
        for ppo_epoch in range(params['ppo_epochs']):
            new_log_probs = learner.log_prob(states, actions)
            # Compute the policy loss
            loss = ppo.policy_loss(new_log_probs, old_log_probs, advantages, clip=params['ppo_clip_ratio'])
            # Adapt model based on the loss
            learner.adapt(loss, allow_unused=anil)
            av_loss += loss

    # We need to include the body network parameters for the query set
    if anil:
        learner.module.turn_on_body_grads()

    # Collect evaluation / query episodes
    query_episodes = task.run(learner, episodes=params['adapt_batch_size'])
    # Get values to device
    states, actions, rewards, dones, next_states = get_episode_values(query_episodes)
    # Update value function & Compute advantages
    advantages = compute_advantages(baseline, params['tau'], params['gamma'], rewards, dones, states, next_states)
    advantages = ch.normalize(advantages, epsilon=1e-8).detach()
    # Calculate loss between states and action in the network
    with torch.no_grad():
        old_log_probs = learner.log_prob(states, actions)

    new_log_probs = learner.log_prob(states, actions)
    # Compute the policy loss
    valid_loss = ppo.policy_loss(new_log_probs, old_log_probs, advantages, clip=params['ppo_clip_ratio'])

    # Calculate the average reward of the evaluation episodes
    query_rew = query_episodes.reward().sum().item() / params['adapt_batch_size']
    query_success_rate = get_ep_successes(query_episodes, params['max_path_length']) / params['adapt_batch_size']

    return valid_loss, query_rew, query_success_rate


def single_ppo_update(episodes, learner, baseline, params, anil=False):
    # Get values to device
    states, actions, rewards, dones, next_states = get_episode_values(episodes)

    # Update value function & Compute advantages
    advantages = compute_advantages(baseline, params['tau'], params['gamma'], rewards, dones, states, next_states)
    advantages = ch.normalize(advantages, epsilon=1e-8).detach()
    # Calculate loss between states and action in the network
    with torch.no_grad():
        old_log_probs = learner.log_prob(states, actions)

    # Initialize inner loop PPO optimizer
    new_log_probs = learner.log_prob(states, actions)
    # Compute the policy loss
    loss = ppo.policy_loss(new_log_probs, old_log_probs, advantages, clip=params['ppo_clip_ratio'])
    # Adapt model based on the loss
    learner.adapt(loss, allow_unused=anil)
    return loss


def evaluate_ppo(env, policy, baseline, eval_params, anil=False, render=False):
    return evaluate('ppo', env, policy, baseline, eval_params, anil, render=render)


""" TRPO RELATED """


def trpo_a2c_loss(episodes, learner, baseline, gamma, tau, update_vf=True):
    # Get values to device
    states, actions, rewards, dones, next_states = get_episode_values(episodes)

    # Calculate loss between states and action in the network
    log_probs = learner.log_prob(states, actions)

    # Compute advantages & normalize
    advantages = compute_advantages(baseline, tau, gamma, rewards, dones, states, next_states, update_vf=update_vf)
    advantages = ch.normalize(advantages).detach()

    # Compute the policy loss
    return a2c.policy_loss(log_probs, advantages)


def trpo_update(episodes, learner, baseline, inner_lr, gamma, tau, anil=False, first_order=False):
    second_order = not first_order

    # Calculate loss & fit the value function
    loss = trpo_a2c_loss(episodes, learner, baseline, gamma, tau)

    # First or Second order derivatives
    gradients = torch.autograd.grad(loss, learner.parameters(),
                                    retain_graph=second_order,
                                    create_graph=second_order,
                                    allow_unused=anil)

    # Perform a MAML update of all the parameters in the model variable using the gradients above
    return maml_update(learner, inner_lr, gradients)


def fast_adapt_trpo(task, learner, baseline, params, anil=False, first_order=False, render=False):
    task_replay = []

    # During inner loop adaptation we do not store gradients for the network body
    if anil:
        learner.module.turn_off_body_grads()

    for step in range(params['adapt_steps']):
        # Collect adaptation / support episodes
        support_episodes = task.run(learner, episodes=params['adapt_batch_size'], render=render)
        task_replay.append(support_episodes)

        learner = trpo_update(support_episodes, learner, baseline,
                              params['inner_lr'], params['gamma'], params['tau'],
                              anil=anil, first_order=first_order)

    # We need to include the body network parameters for the query set
    if anil:
        learner.module.turn_on_body_grads()

    # Collect evaluation / query episodes
    query_episodes = task.run(learner, episodes=params['adapt_batch_size'])
    task_replay.append(query_episodes)
    # Calculate loss for the outer loop optimization WITHOUT adapting
    valid_loss = trpo_a2c_loss(query_episodes, learner, baseline, params['gamma'], params['tau'], update_vf=False)
    # Calculate the average reward of the evaluation episodes
    query_rew = query_episodes.reward().sum().item() / params['adapt_batch_size']
    query_success_rate = get_ep_successes(query_episodes, params['max_path_length']) / params['adapt_batch_size']

    return learner, valid_loss, task_replay, query_rew, query_success_rate


def meta_optimize_trpo(params, policy, baseline, iter_replays, iter_policies, anil=False):
    # Compute CG step direction
    old_loss, old_kl = meta_surrogate_loss(iter_replays, iter_policies, policy, baseline, params, anil)

    grad = torch.autograd.grad(old_loss,
                               policy.parameters(),
                               retain_graph=True)
    grad = parameters_to_vector([g.detach() for g in grad])
    Fvp = trpo.hessian_vector_product(old_kl, policy.parameters())
    step = trpo.conjugate_gradient(Fvp, grad)
    shs = 0.5 * torch.dot(step, Fvp(step))
    lagrange_multiplier = torch.sqrt(shs / params['max_kl'])
    step = step / lagrange_multiplier
    step_ = [torch.zeros_like(p.data) for p in policy.parameters()]
    vector_to_parameters(step, step_)
    step = step_
    del old_kl, Fvp, grad
    old_loss.detach_()

    # Line-search
    for ls_step in range(params['ls_max_steps']):
        stepsize = params['backtrack_factor'] ** ls_step * params['outer_lr']
        clone = deepcopy(policy)
        for p, u in zip(clone.parameters(), step):
            p.data.add_(u.data, alpha=-stepsize)  # same as p.data += u.data * (-stepsize)
        new_loss, kl = meta_surrogate_loss(iter_replays, iter_policies, clone, baseline, params, anil)
        if new_loss < old_loss and kl < params['max_kl']:
            for p, u in zip(policy.parameters(), step):
                p.data.add_(u.data, alpha=-stepsize)  # same as p.data += u.data * (-stepsize)
            break


def meta_surrogate_loss(iter_replays, iter_policies, policy, baseline, params, anil):
    mean_loss = 0.0
    mean_kl = 0.0
    for task_replays, old_policy in zip(iter_replays, iter_policies):
        train_replays = task_replays[:-1]
        valid_episodes = task_replays[-1]
        new_policy = clone_module(policy)

        # Fast Adapt to the training episodes
        for train_episodes in train_replays:
            new_policy = trpo_update(train_episodes, new_policy, baseline,
                                     params['inner_lr'], params['gamma'], params['tau'],
                                     anil=anil, first_order=False)

        # Calculate KL from the validation episodes
        states, actions, rewards, dones, next_states = get_episode_values(valid_episodes)

        # Compute KL
        old_densities = old_policy.density(states)
        new_densities = new_policy.density(states)
        kl = kl_divergence(new_densities, old_densities).mean()
        mean_kl += kl

        # Compute Surrogate Loss
        advantages = compute_advantages(baseline, params['tau'], params['gamma'], rewards, dones, states, next_states)
        advantages = ch.normalize(advantages).detach()
        old_log_probs = old_densities.log_prob(actions).mean(dim=1, keepdim=True).detach()
        new_log_probs = new_densities.log_prob(actions).mean(dim=1, keepdim=True)
        mean_loss += trpo.policy_loss(new_log_probs, old_log_probs, advantages)

    mean_kl /= len(iter_replays)
    mean_loss /= len(iter_replays)
    return mean_loss, mean_kl


def evaluate_trpo(env, policy, baseline, eval_params, anil=False, render=False):
    return evaluate('trpo', env, policy, baseline, eval_params, anil, render=render)
