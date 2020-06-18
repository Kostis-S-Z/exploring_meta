from collections import defaultdict

import numpy as np
import torch


def tensor(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)

    x = np.asarray(x, dtype=np.float)
    x = torch.tensor(x, device=device, dtype=torch.float32)
    return x


def input_preprocessing(x, device):
    x = np.transpose(x, (0, 3, 1, 2))
    x = tensor(x, device)
    x = x.float()
    x /= 255.0
    return x


def to_np(t):
    return t.cpu().detach().numpy()


class Sampler:
    def __init__(self, env, model, num_steps, gamma_coef, lambda_coef, device, num_envs):
        self.env = env
        self.model = model
        self.num_steps = num_steps
        self.lam = lambda_coef
        self.gamma = gamma_coef
        self.device = device

        self.obs = np.zeros(
            (num_envs,) + env.observation_space.shape,
            dtype=env.observation_space.dtype.name,
        )

        self.obs[:] = env.reset()
        self.dones = np.array([False for _ in range(num_envs)])

    def run(self, no_grad=False, with_adv_ret=True):

        # In case you want to collect data without tracking gradients (e.g validation / testing)
        if no_grad:
            with torch.no_grad():
                storage, epinfos = self.collect_experience(with_adv_ret)
        else:
            storage, epinfos = self.collect_experience(with_adv_ret)

        for key in storage:
            if len(storage[key].shape) < 3:
                storage[key] = np.expand_dims(storage[key], -1)
            s = storage[key].shape
            storage[key] = storage[key].swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

        return storage, epinfos

    def collect_experience(self, with_adv_ret):
        # Its a defaultdict and not a dict in order to initialize the default value with a list and append without
        # raising KeyError
        storage = defaultdict(list)  # should contain (state, action, reward, done, next state)
        epinfos = []

        for _ in range(self.num_steps):
            obs = input_preprocessing(self.obs, device=self.device)
            storage["states"] += [to_np(obs.clone())]
            # Forward pass
            prediction, infos = self.model.step(obs)
            actions = to_np(prediction)
            storage["actions"] += [actions]
            storage["values"] += [to_np(infos["value"])]
            storage["log_prob"] += [to_np(infos["log_prob"])]
            storage["mass"] += [infos["mass"]]

            self.obs[:], rewards, self.dones, _ = self.env.step(actions)
            storage["rewards"] += [rewards]
            # Convert booleans to integers
            storage["dones"] += [int(d is True) for d in self.dones]
            storage["next_states"] += [to_np(obs.clone())]
            for info in infos:
                if "episode" in info:
                    epinfos.append(info["episode"])

        # batch of steps to batch of rollouts
        for key in storage:
            storage[key] = np.asarray(storage[key])

        # Calculate PPO's advantages & returns
        if with_adv_ret:
            storage["advantages"], storage["returns"] = self.calc_adv_ret(storage["rewards"],
                                                                          storage["dones"],
                                                                          storage["values"])

        return storage, epinfos

    def calc_adv_ret(self, rewards, dones, values):
        obs = input_preprocessing(self.obs, device=self.device)
        last_values = to_np(self.model.step(obs)[1]["value"])

        # discount/bootstrap
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)

        last_gae_lam = 0
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_non_terminal = 1.0 - self.dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_values = values[t + 1]

            td_error = (
                    rewards[t]
                    + self.gamma * next_values * next_non_terminal
                    - values[t]
            )

            advantages[t] = last_gae_lam = (
                    td_error + self.gamma * self.lam * next_non_terminal * last_gae_lam
            )

        returns = advantages + values

        return advantages, returns
