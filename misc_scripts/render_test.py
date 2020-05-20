from utils import make_metaworld
import cherry as ch

env_name = 'ML1_push-v1'

env = make_metaworld(env_name, n_workers=1, test=False)

obs = env.reset()
done = False
while True:
    env.set_task(env.sample_tasks(1)[0])
    env.reset()
    task = ch.envs.Runner(env)

    def get_action(state):
        return env.action_space.sample()

    task.run(get_action, episodes=10, render=True)
