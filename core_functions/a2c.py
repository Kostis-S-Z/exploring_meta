import torch
import learn2learn as l2l
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
                                                 torch.zeros(1))
        advantages = ch.normalize(advantages, epsilon=1e-8)
        returns = ch.td.discount(gamma, rewards, dones)

    return advantages, returns


def compute_a2c_loss(episode_samples, device='cpu'):
    log_prob = torch.from_numpy(episode_samples["log_prob"]).to(device)
    advantages = torch.from_numpy(episode_samples["advantages"]).to(device)
    values = torch.from_numpy(episode_samples["values"]).to(device)
    returns = torch.from_numpy(episode_samples["returns"]).to(device)

    return policy_loss(log_prob, advantages), state_value_loss(values, returns)


def adapt_a2c(train_episodes, a2c, actor, critic, adapt_steps, inner_lr, device='cpu'):
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

        second_order = True
        # Update actor
        actor_grads = torch.autograd.grad(actor_loss[step],
                                          actor.parameters(),
                                          retain_graph=second_order,
                                          create_graph=second_order,
                                          allow_unused=True)

        l2l.algorithms.maml.maml_update(actor, inner_lr, actor_grads)

        # Update critic
        critic_grads = torch.autograd.grad(critic_loss[step],
                                           critic.parameters(),
                                           retain_graph=second_order,
                                           create_graph=second_order,
                                           allow_unused=True)

        l2l.algorithms.maml.maml_update(critic, inner_lr, critic_grads)

    return actor_loss, critic_loss
