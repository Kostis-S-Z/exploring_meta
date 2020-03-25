import torch
import cherry as ch
import learn2learn as l2l
from copy import deepcopy

from torch.distributions.kl import kl_divergence
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from cherry.algorithms import a2c, trpo
from cherry.pg import generalized_advantage


def compute_advantages(baseline, tau, gamma, rewards, dones, states, next_states):
    # Update baseline
    returns = ch.td.discount(gamma, rewards, dones)
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


def maml_a2c_loss(train_episodes, learner, baseline, gamma, tau):
    # Update policy and baseline
    states = train_episodes.state()
    actions = train_episodes.action()
    rewards = train_episodes.reward()
    dones = train_episodes.done()
    next_states = train_episodes.next_state()

    log_probs = learner.log_prob(states, actions)

    advantages = compute_advantages(baseline, tau, gamma, rewards, dones, states, next_states)

    advantages = ch.normalize(advantages).detach()
    return a2c.policy_loss(log_probs, advantages)


def fast_adapt_a2c(clone, train_episodes, baseline, fast_lr, gamma, tau, first_order=False):
    second_order = not first_order
    loss = maml_a2c_loss(train_episodes, clone, baseline, gamma, tau)
    gradients = torch.autograd.grad(loss,
                                    clone.parameters(),
                                    retain_graph=second_order,
                                    create_graph=second_order)
    return l2l.algorithms.maml.maml_update(clone, fast_lr, gradients)


def meta_surrogate_loss(iter_replays, iter_policies, policy, baseline, tau, gamma, fast_lr):
    mean_loss = 0.0
    mean_kl = 0.0
    for task_replays, old_policy in zip(iter_replays, iter_policies):
        train_replays = task_replays[:-1]
        valid_episodes = task_replays[-1]
        new_policy = l2l.clone_module(policy)

        # Fast Adapt
        for train_episodes in train_replays:
            new_policy = fast_adapt_a2c(new_policy, train_episodes, baseline,
                                        fast_lr, gamma, tau, first_order=False)

        # Useful values
        states = valid_episodes.state()
        actions = valid_episodes.action()
        next_states = valid_episodes.next_state()
        rewards = valid_episodes.reward()
        dones = valid_episodes.done()

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


def meta_optimize(params, policy, baseline, iter_replays, iter_policies, cuda):
    # TRPO meta-optimization

    if cuda:
        policy.to('cuda', non_blocking=True)
        baseline.to('cuda', non_blocking=True)
        iter_replays = [[r.to('cuda', non_blocking=True) for r in task_replays] for task_replays in
                        iter_replays]

    # Compute CG step direction
    old_loss, old_kl = meta_surrogate_loss(iter_replays, iter_policies, policy, baseline,
                                           params['tau'], params['gamma'], params['inner_lr'])  # TODO: maybe outer_lr

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
                                           params['tau'], params['gamma'], params['inner_lr'])  # TODO: maybe outer_lr
        if new_loss < old_loss and kl < params['max_kl']:
            for p, u in zip(policy.parameters(), step):
                p.data.add_(-stepsize, u.data)
            break
