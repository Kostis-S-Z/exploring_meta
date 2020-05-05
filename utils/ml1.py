from metaworld.benchmarks import ML1


class MetaWorldML1(ML1):

    def __init__(self, task_name, env_type, n_goals=50, sample_all=False):
        super(MetaWorldML1, self).__init__(task_name, env_type, n_goals, sample_all)
        self.collected_steps = 0

    def step(self, action):
        obs, reward, done, info = super().step(action)
        self.collected_steps += 1

        # Manually set done at the end of the horizon
        if self.collected_steps >= self.active_env.max_path_length:
            done = True

        return obs, reward, done, info

    def reset(self, **kwargs):
        self.collected_steps = 0
        obs = super().reset(**kwargs)
        return obs
