import torch
import cherry as ch
import learn2learn as l2l
from copy import deepcopy

from torch.distributions.kl import kl_divergence
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from cherry.algorithms import a2c, trpo
from cherry.pg import generalized_advantage


def get_episode_values(episodes, device):
    if isinstance(episodes, dict):
        states = torch.from_numpy(episodes["states"]).to(device)
        actions = torch.from_numpy(episodes["actions"]).to(device)
        rewards = torch.from_numpy(episodes["rewards"]).to(device)
        # Due to older pytorch version there is bug where all parameters should be floats and not integers
        dones = torch.from_numpy(episodes["dones"]).to(device).float()
        next_states = torch.from_numpy(episodes["next_states"]).to(device)
    else:
        states = episodes.state().to(device)
        actions = episodes.action().to(device)
        rewards = episodes.reward().to(device)
        dones = episodes.done().to(device)
        next_states = episodes.next_state().to(device)
    return states, actions, rewards, dones, next_states


def weighted_cumsum(values, weights):
    for i in range(values.size(0)):
        values[i] += values[i - 1] * weights[i]
    return values


def compute_advantages(baseline, tau, gamma, rewards, dones, states, next_states):
    # Linear value function approximator vs CNN
    if isinstance(baseline,  ch.models.robotics.LinearValue):
        return compute_advantages_linear(baseline, tau, gamma, rewards, dones, states, next_states)
    else:
        return compute_advantages_cnn(baseline, tau, gamma, rewards, dones, states, next_states)


def compute_advantages_linear(baseline, tau, gamma, rewards, dones, states, next_states):
    # Update baseline
    returns = ch.td.discount(gamma, rewards, dones)
    if baseline.linear.weight.dim() != states.dim():  # if dimensions are not equal, try to flatten
        states = states.flatten(1, -1)
        next_states = next_states.flatten(1, -1)
        dones = dones.reshape(-1, 1)

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


def compute_advantages_cnn(baseline, tau, gamma, rewards, dones, states, next_states):
    baseline_model = baseline[0]
    baseline_loss = baseline[1]
    baseline_optimizer = baseline[2]

    # Update baseline
    returns = ch.td.discount(gamma, rewards, dones)
    dones = dones.reshape(-1, 1)

    # Optimize for values
    baseline_optimizer.zero_grad()
    values = baseline_model(states)
    loss_1 = baseline_loss(values, returns)
    loss_1.backward()
    baseline_optimizer.step()

    baseline_optimizer.zero_grad()
    next_values = baseline_model(next_states)
    loss_2 = baseline_loss(next_values, returns)
    loss_2.backward()
    baseline_optimizer.step()

    with torch.no_grad():
        values = baseline_model(states)
        next_values = baseline_model(next_states)

    bootstraps = values * (1.0 - dones) + next_values * dones
    next_value = torch.zeros(1, device=values.device)
    return generalized_advantage(tau=tau,
                                 gamma=gamma,
                                 rewards=rewards,
                                 dones=dones,
                                 values=bootstraps,
                                 next_value=next_value)


def maml_vpg_a2c_loss(train_episodes, learner, baseline, gamma, tau, device='cpu'):
    # Update policy and baseline
    states, actions, rewards, dones, next_states = get_episode_values(train_episodes, device)

    log_probs = learner.log_prob(states, actions)

    weights = torch.ones_like(dones)
    weights[1:].add_(-1.0, dones[:-1])
    weights /= dones.sum()

    cum_log_probs = weighted_cumsum(log_probs, weights)

    advantages = compute_advantages(baseline, tau, gamma, rewards, dones, states, next_states)

    return a2c.policy_loss(l2l.magic_box(cum_log_probs), advantages)


def maml_trpo_a2c_loss(train_episodes, learner, baseline, gamma, tau, device):
    # Update policy and baseline
    states, actions, rewards, dones, next_states = get_episode_values(train_episodes, device)

    log_probs = learner.log_prob(states, actions)

    advantages = compute_advantages(baseline, tau, gamma, rewards, dones, states, next_states)

    advantages = ch.normalize(advantages).detach()
    return a2c.policy_loss(log_probs, advantages)


def fast_adapt_trpo_a2c(clone, train_episodes, baseline, fast_lr, gamma, tau, first_order=False, device='cpu'):
    second_order = not first_order
    loss = maml_trpo_a2c_loss(train_episodes, clone, baseline, gamma, tau, device)
    gradients = torch.autograd.grad(loss,
                                    clone.parameters(),
                                    retain_graph=second_order,
                                    create_graph=second_order)
    return l2l.algorithms.maml.maml_update(clone, fast_lr, gradients)


def meta_surrogate_loss(iter_replays, iter_policies, policy, baseline, tau, gamma, fast_lr, device):
    mean_loss = 0.0
    mean_kl = 0.0
    for task_replays, old_policy in zip(iter_replays, iter_policies):
        train_replays = task_replays[:-1]
        valid_episodes = task_replays[-1]
        new_policy = l2l.clone_module(policy)

        # Fast Adapt
        for train_episodes in train_replays:
            new_policy = fast_adapt_trpo_a2c(new_policy, train_episodes, baseline,
                                             fast_lr, gamma, tau, first_order=False, device=device)

        # Useful values
        states, actions, rewards, dones, next_states = get_episode_values(valid_episodes, device)

        # Compute KL
        old_densities = old_policy.density(states)
        new_densities = new_policy.density(states)
        kl = kl_divergence(new_densities, old_densities).mean()
        mean_kl += kl

        # Compute Surrogate Loss
        advantages = compute_advantages(baseline, tau, gamma, rewards, dones, states, next_states)
        advantages = ch.normalize(advantages).detach()
        old_log_probs = old_densities.log_prob(actions).mean(dim=1, keepdim=True).detach()
        new_log_probs = new_densities.log_prob(actions).mean(dim=1, keepdim=True)
        mean_loss += trpo.policy_loss(new_log_probs, old_log_probs, advantages)
    mean_kl /= len(iter_replays)
    mean_loss /= len(iter_replays)
    return mean_loss, mean_kl


def trpo_meta_optimization(params, policy, baseline, iter_replays, iter_policies, device):
    # TRPO meta-optimization

    if device == torch.device('cuda'):
        policy.to('cuda', non_blocking=True)
        baseline.to('cuda', non_blocking=True)
        iter_replays = [[r.to('cuda', non_blocking=True) for r in task_replays] for task_replays in
                        iter_replays]

    # Compute CG step direction
    old_loss, old_kl = meta_surrogate_loss(iter_replays, iter_policies, policy, baseline,
                                           params['tau'], params['gamma'], params['inner_lr'],
                                           device)  # TODO: maybe outer_lr

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
        stepsize = params['backtrack_factor'] ** ls_step * params['outer_lr']  # TODO: maybe inner_lr
        clone = deepcopy(policy)
        for p, u in zip(clone.parameters(), step):
            p.data.add_(-stepsize, u.data)
        new_loss, kl = meta_surrogate_loss(iter_replays, iter_policies, clone, baseline,
                                           params['tau'], params['gamma'], params['inner_lr'],
                                           device)  # TODO: maybe outer_lr
        if new_loss < old_loss and kl < params['max_kl']:
            for p, u in zip(policy.parameters(), step):
                p.data.add_(-stepsize, u.data)
            break


def evaluate(env, policy, baseline, eval_params):
    tasks_reward = 0
    eval_task_list = env.sample_tasks(eval_params['n_eval_tasks'])

    for i, task in enumerate(eval_task_list):
        clone = deepcopy(policy)
        env.set_task(task)
        env.reset()
        task = ch.envs.Runner(env)

        # Adapt
        for step in range(eval_params['n_eval_adapt_steps']):
            train_episodes = task.run(clone, episodes=eval_params['n_eval_episodes'])
            clone = fast_adapt_trpo_a2c(clone, train_episodes, baseline,
                                        eval_params['inner_lr'], eval_params['gamma'], eval_params['tau'],
                                        first_order=True)

        valid_episodes = task.run(clone, episodes=eval_params['n_eval_episodes'])

        task_reward = valid_episodes.reward().sum().item() / eval_params['n_eval_episodes']
        print(f"Reward for task {i} : {task_reward}")
        tasks_reward += task_reward

    final_eval_reward = tasks_reward / eval_params['n_eval_tasks']
    return final_eval_reward
