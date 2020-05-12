import torch
import cherry as ch
import learn2learn as l2l
from copy import deepcopy

from torch.distributions.kl import kl_divergence
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from cherry.algorithms import a2c, trpo
from cherry.pg import generalized_advantage


def weighted_cumsum(values, weights):
    for i in range(values.size(0)):
        values[i] += values[i - 1] * weights[i]
    return values


def get_episode_values(episodes, device):
    states = episodes.state().to(device)
    actions = episodes.action().to(device)
    rewards = episodes.reward().to(device)
    dones = episodes.done().to(device)
    next_states = episodes.next_state().to(device)

    return states, actions, rewards, dones, next_states


def compute_advantages(baseline, tau, gamma, rewards, dones, states, next_states):
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


def adapt_trpo_a2c(model, episodes, baseline, inner_lr, gamma, tau, anil=False, first_order=False, device='cpu'):

    # Calculate loss
    loss = trpo_a2c_loss(episodes, model, baseline, gamma, tau, device)

    # First or Second order derivatives
    second_order = not first_order

    # TODO: 1st METHOD!
    if anil:
        model = model.head

    gradients = torch.autograd.grad(loss,
                                    model.parameters(),
                                    retain_graph=second_order,
                                    create_graph=second_order)

    # Perform a MAML update of all the parameters in the model variable using the gradients above
    return l2l.algorithms.maml.maml_update(model, inner_lr, gradients)


def meta_surrogate_loss(iter_replays, iter_policies, policy, baseline, tau, gamma, fast_lr, device):
    mean_loss = 0.0
    mean_kl = 0.0
    for task_replays, old_policy in zip(iter_replays, iter_policies):
        train_replays = task_replays[:-1]
        valid_episodes = task_replays[-1]
        new_policy = l2l.clone_module(policy)

        # Fast Adapt
        # for train_episodes in train_replays:
        #     new_policy.head = adapt_trpo_a2c(new_policy, train_episodes, baseline,
        #                                 fast_lr, gamma, tau, anil=True, first_order=False, device=device)

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


def meta_optimize(params, policy, baseline, iter_replays, iter_policies, device):
    # TRPO meta-optimization

    if device == torch.device('cuda'):
        policy.to('cuda', non_blocking=True)
        baseline.to('cuda', non_blocking=True)
        iter_replays = [[r.to('cuda', non_blocking=True) for r in task_replays] for task_replays in
                        iter_replays]

    # Compute CG step direction
    old_loss, old_kl = meta_surrogate_loss(iter_replays, iter_policies, policy, baseline,
                                           params['tau'], params['gamma'], params['inner_lr'], device)  # TODO: maybe outer_lr

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
                                           params['tau'], params['gamma'], params['inner_lr'], device)  # TODO: maybe outer_lr
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
            clone = adapt_trpo_a2c(clone, train_episodes, baseline,
                                   eval_params['inner_lr'], eval_params['gamma'], eval_params['tau'],
                                   first_order=True)

        valid_episodes = task.run(clone, episodes=eval_params['n_eval_episodes'])

        task_reward = valid_episodes.reward().sum().item() / eval_params['n_eval_episodes']
        print(f"Reward for task {i} : {task_reward}")
        tasks_reward += task_reward

    final_eval_reward = tasks_reward / eval_params['n_eval_tasks']
    return final_eval_reward
