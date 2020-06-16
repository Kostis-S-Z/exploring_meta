from utils import make_env
import cherry as ch

env_name = 'ML10'

env = make_env(env_name, n_workers=1, seed=1, test=False, max_path_length=100)

obs = env.reset()
done = False
while True:
    env.set_task(env.sample_tasks(1)[0])
    env.reset()
    task = ch.envs.Runner(env)

    def get_action(state):
        return env.action_space.sample()

    task.run(get_action, episodes=10, render=True)
