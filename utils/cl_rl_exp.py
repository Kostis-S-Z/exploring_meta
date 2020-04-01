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
from core_functions.rl import fast_adapt_a2c
from utils import calc_cl_metrics
from sklearn import preprocessing
setting = 1

default_params = {
    "adapt_steps": 1,
    "adapt_batch_size": 1,
    "inner_lr": 0.1,
    "gamma": 0.99,
    "tau": 1.0,
    "n_tasks": 5
}


def run_cl_rl_exp(path, env, policy, baseline, cl_params=default_params):
    cl_path = path + '/cl_exp'
    # os.mkdir(cl_path)

    # Matrix R NxN of accuracies in tasks j after trained on a tasks i (x_axis = test tasks, y_axis = train tasks)
    rew_matrix = np.zeros((cl_params['n_tasks'], cl_params['n_tasks']))

    # Sample tasks
    tasks = env.sample_tasks(cl_params['n_tasks'])
    # tasks = [
    #     {'goal': np.array((0, 0))},
    #     {'goal': np.array((0.5, 0.5))},
    #     {'goal': np.array((-0.5, 0.5))},
    #     {'goal': np.array((-0.5, -0.5))},
    #     {'goal': np.array((0.5, -0.5))}]

    for i, train_task in enumerate(tasks):

        clone = deepcopy(policy)
        env.set_task(train_task)
        env.reset()

        task_i = ch.envs.Runner(env, meta_env=True)

        # Adapt to specific task
        for step in range(cl_params['adapt_steps']):
            # print(f"Step {step}")
            train_episodes = task_i.run(clone, episodes=cl_params['adapt_batch_size'])
            clone = fast_adapt_a2c(clone, train_episodes, baseline,
                                   cl_params['inner_lr'], cl_params['gamma'], cl_params['tau'],
                                   first_order=False)

        print(f"Adapt reward {train_episodes.reward().sum().item() / cl_params['adapt_batch_size']}")

        # Evaluate on all tasks
        for j, valid_task in enumerate(tasks):
            env.set_task(valid_task)
            env.reset()
            task_j = ch.envs.Runner(env, meta_env=True)

            valid_episodes = task_j.run(clone, episodes=cl_params['adapt_batch_size'])
            task_j_reward = valid_episodes.reward().sum().item() / cl_params['adapt_batch_size']
            rew_matrix[i, j] = task_j_reward

    print(rew_matrix)

    # norm_rew = preprocessing.normalize(rew_matrix)
    # scaler = preprocessing.StandardScaler()
    # stand_rew = scaler.fit_transform(rew_matrix)
    # print(stand_rew)
    # print(norm_rew)

    cl_res = calc_cl_metrics(rew_matrix)
    print(cl_res)
    # save_acc_matrix(cl_path, rew_matrix)
    # with open(cl_path + '/cl_params.json', 'w') as fp:
    #     json.dump(cl_params, fp, sort_keys=True, indent=4)
    #
    # with open(cl_path + '/cl_res.json', 'w') as fp:
    #     json.dump(cl_res, fp, sort_keys=True, indent=4)
    #
    # return rew_matrix, cl_res


def save_acc_matrix(path, acc_matrix):
    print('Saving accuracy matrix..')
    print(acc_matrix)
    np.savetxt(path + '/acc_matrix.out', acc_matrix, fmt='%1.2f')
