import cherry as ch
import learn2learn as l2l

import gym
import utils

mujoco_envs = ['Particles2D-v1', 'AntDirection-v1']
metaworld_envs = ['ML1_reach-v1', 'ML1_pick-place-v1', 'ML1_push-v1', 'ML10', 'ML45']


def make_mujoco(env_name, n_workers):

    def init_env():
        env = gym.make(env_name)
        env = ch.envs.ActionSpaceScaler(env)
        return env

    if n_workers == 1:
        return init_env()
    else:
        return l2l.gym.AsyncVectorEnv([init_env for _ in range(n_workers)])


def _make_metaworld(env_name, n_workers, test, max_path_length):
    if 'ML1_' in env_name:
        env_name, task = env_name.split('_')
    else:
        task = False  # Otherwise, False corresponds to the sample_all argument

    benchmark_env = getattr(utils, f"MetaWorld{env_name}")  # Use modded MetaWorld env

    def init_env():
        if test:
            env = benchmark_env.get_test_tasks(task)
        else:
            env = benchmark_env.get_train_tasks(task)
        if max_path_length is not None:
            env.set_max_path_length(max_path_length)
        env = ch.envs.ActionSpaceScaler(env)
        return env

    if n_workers == 1:
        return init_env()
    else:
        return l2l.gym.AsyncVectorEnv([init_env for _ in range(n_workers)])


def make_env(env_name, n_workers, seed, test=False, max_path_length=None):

    if env_name in mujoco_envs:
        env = make_mujoco(env_name, n_workers)
    elif env_name in metaworld_envs:
        env = _make_metaworld(env_name, n_workers, test, max_path_length)
    else:
        raise NotImplementedError

    env.seed(seed)
    env.set_task(env.sample_tasks(1)[0])
    env = ch.envs.Torch(env)
    return env


def calculate_samples_seen(n_steps, n_episodes, n_inner_steps, n_tasks, n_iters):
    n_samples = dict()
    n_samples['rollout'] = n_steps  # Samples in one episode
    n_samples['task_batch'] = n_samples['rollout'] * n_episodes  # Samples per task (steps * episodes)
    n_samples['task_support'] = n_samples['task_batch'] * n_inner_steps  # Samples adapted to per task
    # Samples in inner loop per task (support + query set)
    n_samples['task_total'] = n_samples['task_support'] + n_samples['task_batch']
    n_samples['iter'] = n_samples['task_total'] * n_tasks  # Samples in one iteration
    n_samples['total'] = n_samples['iter'] * n_iters
    return n_samples
