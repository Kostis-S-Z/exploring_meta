import numpy as np
import torch
from learn2learn.algorithms.maml import maml_update
import cherry as ch
from cherry.algorithms.a2c import policy_loss, state_value_loss


def compute_adv_ret(train_episodes, gamma, tau, device='cpu'):
    values = torch.from_numpy(train_episodes["values"]).to(device)
    rewards = torch.from_numpy(train_episodes["rewards"]).to(device)
    dones = torch.from_numpy(train_episodes["dones"]).to(device).float()

    with torch.no_grad():
        advantages = ch.pg.generalized_advantage(gamma,
                                                 tau,
                                                 rewards,
                                                 dones,
                                                 values,
                                                 torch.zeros(1, device=device))
        advantages = ch.normalize(advantages, epsilon=1e-8)
        returns = ch.td.discount(gamma, rewards, dones)

    return advantages, returns


def compute_a2c_loss(episode_samples, device='cpu'):
    log_prob = torch.from_numpy(episode_samples["log_prob"]).to(device)
    advantages = torch.from_numpy(episode_samples["advantages"]).to(device)
    values = torch.from_numpy(episode_samples["values"]).to(device)
    returns = torch.from_numpy(episode_samples["returns"]).to(device)

    return policy_loss(log_prob, advantages), state_value_loss(values, returns)


def adapt_a2c(train_episodes, a2c, actor, critic, adapt_steps, inner_lr, device):
    states = torch.from_numpy(train_episodes["states"]).to(device)
    actions = torch.from_numpy(train_episodes["actions"]).to(device)
    log_prob = torch.from_numpy(train_episodes["log_prob"]).to(device)
    values = torch.from_numpy(train_episodes["values"]).to(device)
    returns = torch.from_numpy(train_episodes["returns"]).to(device)
    advantages = torch.from_numpy(train_episodes["advantages"]).to(device)

    # Initialize just for the first epoch
    new_values = values
    new_log_probs = log_prob

    actor_loss = torch.zeros(adapt_steps, device=device, requires_grad=True)
    critic_loss = torch.zeros(adapt_steps, device=device, requires_grad=True)

    for step in range(adapt_steps):
        # Recalculate outputs for subsequent iterations
        if step > 0:
            with torch.no_grad():
                _, infos = a2c(states)
            masses = infos['mass']
            new_values = infos['value'].view(-1, 1)
            # TODO: Not sure if the following reshaping that way is correct!!!
            new_log_probs = masses.log_prob(actions.reshape(-1)).reshape(-1, 1)

        # Update the policy by maximising the PPO-Clip objective
        # TODO: different clip ratio?
        # policy_loss[0] = ch.algorithms.ppo.policy_loss(new_log_probs, log_prob, advantages, clip=0.1)
        # OR
        actor_loss[step] = ch.algorithms.a2c.policy_loss(new_log_probs, advantages)
        # Fit value function by regression on mean-squared error
        critic_loss[step] = ch.algorithms.a2c.state_value_loss(new_values, returns)

        # Update actor
        # TODO: allow unused???
        actor.adapt(actor_loss[step], allow_unused=True)

        # Update critic
        critic.adapt(critic_loss[step], allow_unused=True)

        # print(f"actor loss: {actor_loss[step]}\n"
        #       f"critic_loss: {critic_loss[step]}\n")

    # Average loss across adaptation steps
    actor_loss = safemean(actor_loss)
    critic_loss = safemean(critic_loss)

    return actor_loss, critic_loss


"""
second_order = True
# "adapt(loss, unused=True)" is the same as "grad(unused=True) + maml_update"
# Update actor
# actor_grads = torch.autograd.grad(actor_loss[step],
#                                   actor.parameters(),
#                                   retain_graph=second_order,
#                                   create_graph=second_order,
#                                   allow_unused=True)
#
# actor = maml_update(actor, inner_lr, actor_grads)
"""

# Avoid division error when calculate the mean (in our case if epinfo is empty returns np.nan, not return an error)
def safemean(xs):
    if isinstance(xs, torch.Tensor):
        xs = xs.cpu().detach().numpy()
    return np.nan if len(xs) == 0 else np.mean(xs)
