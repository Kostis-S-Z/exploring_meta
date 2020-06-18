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
import torch
from sklearn import preprocessing

from utils import calc_cl_metrics, make_env
from core_functions.rl import vpg_a2c_loss, ppo_update, trpo_update, get_ep_successes, get_success_per_ep

from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator


def run_cl_rl_exp(path, env_name, policy, baseline, cl_params, workers, plots=False):
    cl_path = path + '/cl_exp'
    if not os.path.isdir(cl_path):
        os.mkdir(cl_path)

    env = make_env(env_name, workers, cl_params['seed'], test=True, max_path_length=cl_params['max_path_length'])

    # Matrix R NxN of rewards / success rates in tasks j after trained on a tasks i
    # (x_axis = test tasks, y_axis = train tasks)
    rew_matrix = np.zeros((cl_params['n_tasks'], cl_params['n_tasks']))
    suc_matrix = np.zeros((cl_params['n_tasks'], cl_params['n_tasks']))

    # Sample tasks
    # Randomly
    tasks = env.sample_tasks(cl_params['n_tasks'])
    # Manually
    tasks[0]['goal'] = 0
    tasks[1]['goal'] = 1
    tasks[2]['goal'] = 2
    tasks[3]['goal'] = 3
    tasks[4]['goal'] = 4

    rew_adapt_progress = {}
    suc_adapt_progress = {}

    for i, train_task in enumerate(tasks):
        print(f'Adapting on Task {i} with ID {train_task["task"]} and goal {train_task["goal"]}', end='...')
        learner = deepcopy(policy)
        env.set_task(train_task)
        env.reset()
        task_i = ch.envs.Runner(env, extra_info=cl_params['extra_info'])

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
                                      anil=cl_params['anil'], first_order=True)

            adapt_rew = adapt_ep.reward().sum().item() / cl_params['adapt_batch_size']
            adapt_suc_per_ep, _ = get_success_per_ep(adapt_ep, cl_params['max_path_length'])
            adapt_suc = sum(adapt_suc_per_ep.values()) / cl_params['adapt_batch_size']

            rew_adapt_progress[f'task_{i + 1}'][f'step_{step}'] = adapt_rew
            suc_adapt_progress[f'task_{i + 1}'][f'step_{step}'] = adapt_suc

        print(f'Done!')

        # Evaluate on all tasks
        for j, valid_task in enumerate(tasks):
            print(f'\tEvaluating on Task {j} with ID {valid_task["task"]} and goal {valid_task["goal"]}...', end='\t')
            evaluator = learner.clone()
            env.set_task(valid_task)
            env.reset()
            task_j = ch.envs.Runner(env, extra_info=cl_params['extra_info'])

            with torch.no_grad():
                eval_ep = task_j.run(evaluator, episodes=cl_params['eval_batch_size'])
            task_j_reward = eval_ep.reward().sum().item() / cl_params['eval_batch_size']
            task_j_success = get_ep_successes(eval_ep, cl_params['max_path_length']) / cl_params['eval_batch_size']

            _, success_step = get_success_per_ep(eval_ep, cl_params['max_path_length'])

            rew_matrix[i, j] = task_j_reward
            suc_matrix[i, j] = task_j_success
            print(f'Success: {task_j_success * 100}%')

    # Plot matrix results
    if plots:
        plot_task_res(rew_matrix, y_title='Reward')
        plot_task_res(suc_matrix, y_title='Success Rate')

        # Plot adaptation progress
        plot_progress(rew_adapt_progress, y_title='Reward')
        plot_progress(suc_adapt_progress, y_title='Success Rate')

    print(f'Rewards Matrix:\n{rew_matrix}\n')
    print(f'Success rates Matrix:\n{suc_matrix}\n')

    if cl_params['normalize_rewards']:
        norm_rew = preprocessing.normalize(rew_matrix)
        scaler = preprocessing.StandardScaler()
        stand_rew = scaler.fit_transform(rew_matrix)
        print(stand_rew)
        print(norm_rew)
        rew_matrix = norm_rew

    cl_res_rew = calc_cl_metrics(rew_matrix)
    cl_res_suc = calc_cl_metrics(suc_matrix)

    print(f'Metrics based on rewards: {cl_res_rew}')
    print(f'Metrics based on success rates: {cl_res_suc}')

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
    print('Saving matrix to file..')
    np.savetxt(path + f'/{name}.out', acc_matrix, fmt='%1.2f')


def plot_progress(progress_dict, y_title='Reward'):
    plt.figure().gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # Set integers only in x ticks
    plt.title('Adaptation progress')
    plt.xlabel('Adaptation steps')
    plt.ylabel(y_title)
    for task, steps in progress_dict.items():
        y_title = list(steps.values())
        x_axis = range(1, len(y_title) + 1)
        plt.plot(x_axis, y_title, label=task, marker='o', alpha=0.8)
    plt.legend()
    plt.show()


def plot_task_res(matrix, y_title='Reward'):
    plt.figure().gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # Set integers only in x ticks
    plt.title('Performance across tasks')
    plt.xlabel('Task ID')
    plt.ylabel(y_title)
    for i in range(matrix.shape[0]):
        y_title = matrix[i]
        x_axis = range(1, len(y_title) + 1)
        plt.plot(x_axis, y_title, label=f'Tr_task_{i + 1}', marker='o', alpha=0.5)
    plt.legend()
    plt.show()
