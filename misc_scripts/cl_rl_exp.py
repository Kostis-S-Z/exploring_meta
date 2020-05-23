"""
Evaluate model performance in a Continual Learning setting

Setting 1: Tr1 = Te1 (Exactly same task)
Setting 2: Tr1 =/= Te1 (Same task, different reward function?)

"""

import os
import json
import numpy as np
from copy import deepcopy
import cherry as ch
from sklearn import preprocessing

from utils import calc_cl_metrics
from core_functions.rl import vpg_a2c_loss, ppo_update, trpo_update, get_ep_successes

from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator


def run_cl_rl_exp(path, env, policy, baseline, cl_params):
    cl_path = path + '/cl_exp'
    if not os.path.isdir(cl_path):
        os.mkdir(cl_path)

    # Matrix R NxN of rewards / success rates in tasks j after trained on a tasks i
    # (x_axis = test tasks, y_axis = train tasks)
    rew_matrix = np.zeros((cl_params['n_tasks'], cl_params['n_tasks']))
    suc_matrix = np.zeros((cl_params['n_tasks'], cl_params['n_tasks']))

    # Sample tasks
    tasks = env.sample_tasks(cl_params['n_tasks'])

    rew_adapt_progress = {}
    suc_adapt_progress = {}

    for i, train_task in enumerate(tasks):

        learner = deepcopy(policy)
        env.set_task(train_task)
        env.reset()

        task_i = ch.envs.Runner(env)

        rew_adapt_progress[f'task_{i + 1}'] = {}
        suc_adapt_progress[f'task_{i + 1}'] = {}

        if cl_params['anil']:
            learner.module.turn_off_body_grads()

        for step in range(cl_params['adapt_steps']):
            # Collect adaptation / support episodes
            adapt_ep = task_i.run(learner, episodes=cl_params['adapt_batch_size'])

            if cl_params['algo'] == 'vpg':
                # Calculate loss & fit the value function
                inner_loss = vpg_a2c_loss(adapt_ep, learner, baseline, cl_params['gamma'], cl_params['tau'])
                # Adapt model based on the loss
                learner.adapt(inner_loss, allow_unused=cl_params['anil'])

            elif cl_params['algo'] == 'ppo':
                # Calculate loss & fit the value function & update the policy
                ppo_update(adapt_ep, learner, baseline, cl_params, anil=cl_params['anil'])
            else:

                learner = trpo_update(adapt_ep, learner, baseline,
                                      cl_params['inner_lr'], cl_params['gamma'], cl_params['tau'],
                                      anil=cl_params['anil'])

            adapt_rew = adapt_ep.reward().sum().item() / cl_params['adapt_batch_size']
            adapt_suc = get_ep_successes(adapt_ep, cl_params['max_path_length']) / cl_params['adapt_batch_size']

            rew_adapt_progress[f'task_{i + 1}'][f'step_{step}'] = adapt_rew
            suc_adapt_progress[f'task_{i + 1}'][f'step_{step}'] = adapt_suc

        # Evaluate on all tasks
        for j, valid_task in enumerate(tasks):
            env.set_task(valid_task)
            env.reset()
            task_j = ch.envs.Runner(env)

            valid_episodes = task_j.run(learner, episodes=cl_params['adapt_batch_size'])
            task_j_reward = valid_episodes.reward().sum().item() / cl_params['adapt_batch_size']
            task_j_success = get_ep_successes(valid_episodes, cl_params['max_path_length']) / cl_params['adapt_batch_size']

            rew_matrix[i, j] = task_j_reward
            suc_matrix[i, j] = task_j_success

    # Plot adaptation progress
    plot_progress(rew_adapt_progress, y_axis='Reward')
    plot_progress(suc_adapt_progress, y_axis='Success Rate')

    print(rew_matrix)

    if cl_params['normalize_rewards']:
        norm_rew = preprocessing.normalize(rew_matrix)
        scaler = preprocessing.StandardScaler()
        stand_rew = scaler.fit_transform(rew_matrix)
        print(stand_rew)
        print(norm_rew)
        rew_matrix = norm_rew

    cl_res_rew = calc_cl_metrics(rew_matrix)
    cl_res_suc = calc_cl_metrics(suc_matrix)

    print(cl_res_rew)
    print(cl_res_suc)

    save_acc_matrix(cl_path, rew_matrix, name='cl_rew_matrix')
    save_acc_matrix(cl_path, suc_matrix, name='cl_suc_matrix')

    with open(cl_path + '/cl_params.json', 'w') as fp:
        json.dump(cl_params, fp, sort_keys=True, indent=4)

    with open(cl_path + '/cl_res_rew.json', 'w') as fp:
        json.dump(cl_res_rew, fp, sort_keys=True, indent=4)

    with open(cl_path + '/cl_res_suc.json', 'w') as fp:
        json.dump(cl_res_suc, fp, sort_keys=True, indent=4)

    return rew_matrix, cl_res_rew, cl_res_suc


def save_acc_matrix(path, acc_matrix, name='acc_matrix'):
    print('Saving accuracy matrix..')
    print(acc_matrix)
    np.savetxt(path + f'/{name}.out', acc_matrix, fmt='%1.2f')


def plot_progress(progress_dict, y_axis='Reward'):
    plt.figure().gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # Set integers only in x ticks
    plt.title('Adaptation progress')
    plt.xlabel('Adaptation steps')
    plt.ylabel(y_axis)
    for task, steps in progress_dict.items():
        y_axis = list(steps.values())
        x_axis = range(1, len(y_axis) + 1)
        plt.plot(x_axis, y_axis, label=task, marker='o')
    plt.legend()
    plt.show()
