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

    def run(self):
        # Its a defaultdict and not a dict in order to initialize the default value with a list and append without
        # raising KeyError
        storage = defaultdict(list)  # should contain (state, action, reward, done, next state)
        epinfos = []
        self.model.eval()

        with torch.no_grad():

            for _ in range(self.num_steps):
                obs = input_preprocessing(self.obs, device=self.device)
                storage["states"] += [to_np(obs.clone())]
                # Forward pass
                prediction = self.model.step(obs)
                actions = to_np(prediction)
                # Choose the action with the highest score
                # actions = np.argmax(actions, axis=1)
                # Sample a random action from every environment
                actions = np.random.choice(actions.shape[1], actions.shape[0])
                storage["actions"] += [actions]

                self.obs[:], rewards, self.dones, _ = self.env.step(actions)
                storage["rewards"] += [rewards]
                # Convert booleans to integers
                storage["dones"] += [int(d is True) for d in self.dones]
                storage["next_states"] += [to_np(obs.clone())]

            # batch of steps to batch of rollouts
            for key in storage:
                storage[key] = np.asarray(storage[key])

        for key in storage:
            if len(storage[key].shape) < 3:
                storage[key] = np.expand_dims(storage[key], -1)
            s = storage[key].shape
            storage[key] = storage[key].swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

        return storage, epinfos
