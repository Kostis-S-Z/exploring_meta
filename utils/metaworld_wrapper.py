from learn2learn.gym.envs.meta_env import MetaEnv
from metaworld.envs.mujoco.multitask_env import MultiClassMultiTaskEnv
from metaworld.benchmarks import ML1, ML10, ML45


class MetaWorldMod(MultiClassMultiTaskEnv, MetaEnv):
    """
    Modification to return Done signal when reaching the end of the horizon
    """
    def __init__(self, task_env_cls_dict, task_args_kwargs, sample_all=True, sample_goals=False, obs_type='plain'):
        super(MetaWorldMod, self).__init__(task_env_cls_dict=task_env_cls_dict,
                                           task_args_kwargs=task_args_kwargs,
                                           sample_goals=sample_goals,
                                           obs_type=obs_type,
                                           sample_all=sample_all)
        self.collected_steps = 0
        self.max_path_length = self.active_env.max_path_length

    # -------- MetaEnv Methods --------
    def sample_tasks(self, meta_batch_size):
        return MultiClassMultiTaskEnv.sample_tasks(self, meta_batch_size)

    def set_task(self, task):
        return MultiClassMultiTaskEnv.set_task(self, task)

    def get_task(self):
        return MultiClassMultiTaskEnv.get_task(self)

    # -------- Gym Methods --------
    def step(self, action):
        obs, reward, done, info = super().step(action)
        self.collected_steps += 1

        # Manually set done at the end of the horizon
        if self.collected_steps >= self.max_path_length:
            done = True

        # Ignore all other info fields to be memory-friendly since we don't need them
        info = {'success': info['success']}
        return obs, reward, done, info

    def reset(self, **kwargs):
        self.collected_steps = 0
        obs = super().reset(**kwargs)
        return obs

    # -------- Custom Methods --------

    def set_max_path_length(self, max_path_length):
        self.max_path_length = max_path_length

    def get_max_path_length(self):
        return self.max_path_length


class MetaWorldML1(ML1, MetaWorldMod):

    def __init__(self, task_name, env_type='train', n_goals=50, sample_all=False):
        super(MetaWorldML1, self).__init__(task_name, env_type, n_goals, sample_all)


class MetaWorldML10(ML10, MetaWorldMod):

    def __init__(self, env_type='train', sample_all=False, task_name=None):
        super(MetaWorldML10, self).__init__(env_type, sample_all, task_name)


class MetaWorldML45(ML45, MetaWorldMod):

    def __init__(self, env_type='train', sample_all=False, task_name=None):
        super(MetaWorldML45, self).__init__(env_type, sample_all, task_name)
