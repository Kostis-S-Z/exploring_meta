import torch
import cherry as ch
import learn2learn as l2l
from copy import deepcopy

from torch.distributions.kl import kl_divergence
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from cherry.algorithms import a2c, trpo, ppo
from cherry.pg import generalized_advantage

""" COMMON """


def get_episode_values(episodes, device):
    states = episodes.state().to(device)
    actions = episodes.action().to(device)
    rewards = episodes.reward().to(device)
    dones = episodes.done().to(device)
    next_states = episodes.next_state().to(device)

    return states, actions, rewards, dones, next_states


def compute_advantages(baseline, tau, gamma, rewards, dones, states, next_states):
    returns = ch.td.discount(gamma, rewards, dones)
    # if baseline.linear.weight.dim() != states.dim():  # if dimensions are not equal, try to flatten
    #     states = states.flatten(1, -1)
    #     next_states = next_states.flatten(1, -1)
    #     dones = dones.reshape(-1, 1)

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


def evaluate(algo, env, policy, baseline, eval_params, anil, render=False):
    tasks_rewards = []
    eval_task_list = env.sample_tasks(eval_params['n_eval_tasks'])

    for i, task in enumerate(eval_task_list):
        learner = deepcopy(policy)
        env.set_task(task)
        env.reset()
        task = ch.envs.Runner(env)

        if algo == 'vpg':
            _, task_reward = fast_adapt_vpg(task, learner, baseline, eval_params, anil=anil, render=render)
        elif algo == 'ppo':
            _, task_reward = fast_adapt_ppo(task, learner, baseline, eval_params, render=render)
        else:
            _, _, task_reward = fast_adapt_trpo(task, learner, baseline, eval_params, anil=anil, render=render)

        tasks_rewards.append(task_reward)
        print(f"Reward for task {i} : {task_reward}")

    final_eval_reward = sum(tasks_rewards) / eval_params['n_eval_tasks']
    return tasks_rewards, final_eval_reward


""" VPG RELATED """


def weighted_cumsum(values, weights):
    for i in range(values.size(0)):
        values[i] += values[i - 1] * weights[i]
    return values


def vpg_a2c_loss(episodes, learner, baseline, gamma, tau, device='cpu'):
    # Get values to device
    states, actions, rewards, dones, next_states = get_episode_values(episodes, device)

    # Calculate loss between states and action in the network
    log_probs = learner.log_prob(states, actions)

    # Calculate DiCE objective
    weights = torch.ones_like(dones)
    weights[1:] = dones[:-1] - 1.0
    weights /= dones.sum()
    cum_log_probs = weighted_cumsum(log_probs, weights)

    # Fit value function, compute advantages & normalize
    advantages = compute_advantages(baseline, tau, gamma, rewards, dones, states, next_states)

    # Compute the policy loss
    return a2c.policy_loss(l2l.magic_box(cum_log_probs), advantages)


def fast_adapt_vpg(task, learner, baseline, params, anil=False, first_order=False, render=False, device='cpu'):
    # During inner loop adaptation we do not store gradients for the network body
    if anil:
        learner.module.turn_off_body_grads()

    for step in range(params['adapt_steps']):
        # Collect adaptation / support episodes
        support_episodes = task.run(learner, episodes=params['adapt_batch_size'], render=render)
        # Calculate loss & fit the value function
        inner_loss = vpg_a2c_loss(support_episodes, learner, baseline, params['gamma'], params['tau'], device)
        # Adapt model based on the loss
        learner.adapt(inner_loss, first_order=first_order, allow_unused=anil)  # In ANIL, not all parameters have grads

    # We need to include the body network parameters for the query set
    if anil:
        learner.module.turn_on_body_grads()

    # Collect evaluation / query episodes
    query_episodes = task.run(learner, episodes=params['adapt_batch_size'])
    # Calculate loss for the outer loop optimization
    outer_loss = vpg_a2c_loss(query_episodes, learner, baseline, params['gamma'], params['tau'], device)
    # Calculate the average reward of the evaluation episodes
    query_rew = query_episodes.reward().sum().item() / params['adapt_batch_size']

    return outer_loss, query_rew


def evaluate_vpg(env, policy, baseline, eval_params, anil=False, render=False):
    return evaluate('vpg', env, policy, baseline, eval_params, anil, render=render)


""" PPO RELATED """


def ppo_update(episodes, learner, baseline, params, anil=False, device='cpu'):
    """
    Inspired by cherry-rl/examples/spinning-up/cherry_ppo.py
    """

    # Get values to device
    states, actions, rewards, dones, next_states = get_episode_values(episodes, device)

    returns = ch.td.discount(params['gamma'], rewards, dones)

    # Update value function and get new values
    baseline.fit(states, returns)
    values = baseline(states)
    new_values = baseline(next_states)
    bootstraps = values * (1.0 - dones) + new_values * dones

    # Compute advantages
    with torch.no_grad():
        advantages = ch.pg.generalized_advantage(gamma=params['gamma'], tau=params['tau'],
                                                 rewards=rewards, dones=dones, values=bootstraps,
                                                 next_value=torch.zeros(1, device=values.device))
        advantages = ch.normalize(advantages, epsilon=1e-8).detach()
        # Calculate loss between states and action in the network
        old_log_probs = learner.log_prob(states, actions)

    # Initialize inner loop PPO optimizer
    av_loss = 0.0

    for ppo_epoch in range(params['ppo_epochs']):
        new_log_probs = learner.log_prob(states, actions)

        # Compute the policy loss
        loss = ppo.policy_loss(new_log_probs, old_log_probs, advantages, clip=params['ppo_clip_ratio'])

        # Adapt model based on the loss
        learner.adapt(loss, allow_unused=anil)

        # TODO: do we need to update the value function in every epoch? only once outside?
        # baseline.fit(states, returns)
        # print(f'\tPPO EPOCH {ppo_epoch}: {round(loss.item(), 3)}')
        av_loss += loss

    return av_loss / params['ppo_epochs']


def fast_adapt_ppo(task, learner, baseline, params, anil=False, render=False, device='cpu'):
    # During inner loop adaptation we do not store gradients for the network body
    if anil:
        learner.module.turn_off_body_grads()

    for step in range(params['adapt_steps']):
        # Collect adaptation / support episodes
        support_episodes = task.run(learner, episodes=params['adapt_batch_size'], render=render)

        # Calculate loss & fit the value function & update the policy
        inner_loss = ppo_update(support_episodes, learner, baseline, params, anil=anil, device=device)
        # print(f'Inner loss {step}: {round(inner_loss.item(), 3)}')

    # We need to include the body network parameters for the query set
    if anil:
        learner.module.turn_on_body_grads()

    # Collect evaluation / query episodes
    query_episodes = task.run(learner, episodes=params['adapt_batch_size'])
    # Calculate loss for the outer loop optimization
    outer_loss = ppo_update(query_episodes, learner, baseline, params, anil=anil, device=device)
    # Calculate the average reward of the evaluation episodes
    query_rew = query_episodes.reward().sum().item() / params['adapt_batch_size']

    return outer_loss, query_rew


def evaluate_ppo(env, policy, baseline, eval_params, anil=False, render=False):
    return evaluate('ppo', env, policy, baseline, eval_params, anil, render=render)


""" TRPO RELATED """


def trpo_a2c_loss(episodes, learner, baseline, gamma, tau, device):
    # Get values to device
    states, actions, rewards, dones, next_states = get_episode_values(episodes, device)

    # Calculate loss between states and action in the network
    log_probs = learner.log_prob(states, actions)

    # Compute advantages & normalize
    advantages = compute_advantages(baseline, tau, gamma, rewards, dones, states, next_states)
    advantages = ch.normalize(advantages).detach()

    # Compute the policy loss
    return a2c.policy_loss(log_probs, advantages)


def fast_adapt_trpo(task, learner, baseline, params, anil=False, first_order=True, render=False, device='cpu'):
    task_replay = []
    second_order = not first_order

    # During inner loop adaptation we do not store gradients for the network body
    if anil:
        learner.module.turn_off_body_grads()

    for step in range(params['adapt_steps']):
        # Collect adaptation / support episodes
        support_episodes = task.run(learner, episodes=params['adapt_batch_size'], render=render)
        task_replay.append(support_episodes)

        # Calculate loss & fit the value function
        loss = trpo_a2c_loss(support_episodes, learner, baseline, params['gamma'], params['tau'], device)

        # First or Second order derivatives
        gradients = torch.autograd.grad(loss, learner.parameters(),
                                        retain_graph=second_order,
                                        create_graph=second_order,
                                        allow_unused=anil)

        # Perform a MAML update of all the parameters in the model variable using the gradients above
        learner = l2l.algorithms.maml.maml_update(learner, params['inner_lr'], gradients)

    # We need to include the body network parameters for the query set
    if anil:
        learner.module.turn_on_body_grads()

    # Collect evaluation / query episodes
    query_episodes = task.run(learner, episodes=params['adapt_batch_size'])
    task_replay.append(query_episodes)
    # Calculate the average reward of the evaluation episodes
    query_rew = query_episodes.reward().sum().item() / params['adapt_batch_size']

    return learner, task_replay, query_rew


def meta_optimize_trpo(params, policy, baseline, iter_replays, iter_policies, device):
    # Compute CG step direction
    old_loss, old_kl = meta_surrogate_loss(iter_replays, iter_policies, policy, baseline, params, device)

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
            p.data = u.data + (-stepsize)
        new_loss, kl = meta_surrogate_loss(iter_replays, iter_policies, clone, baseline, params, device)
        if new_loss < old_loss and kl < params['max_kl']:
            for p, u in zip(policy.parameters(), step):
                p.data = u.data + (-stepsize)
            break


def meta_surrogate_loss(iter_replays, iter_policies, policy, baseline, params, device):
    mean_loss = 0.0
    mean_kl = 0.0
    for task_replays, old_policy in zip(iter_replays, iter_policies):
        train_replays = task_replays[:-1]
        valid_episodes = task_replays[-1]
        new_policy = l2l.clone_module(policy)

        # Fast Adapt to the training episodes
        for train_episodes in train_replays:
            # Calculate loss & fit the value function
            loss = trpo_a2c_loss(train_episodes, new_policy, baseline, params['gamma'], params['tau'], device)

            # First or Second order derivatives
            gradients = torch.autograd.grad(loss, new_policy.parameters(),
                                            retain_graph=True,  # First order = False
                                            create_graph=True)

            # Perform a MAML update of all the parameters in the model variable using the gradients above
            new_policy = l2l.algorithms.maml.maml_update(new_policy, params['inner_lr'], gradients)

        # Calculate KL from the validation episodes
        states, actions, rewards, dones, next_states = get_episode_values(valid_episodes, device)

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
