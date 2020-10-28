"""
Evaluate model performance in a Continual Learning setting

Setting 1: Tr1 = Te1 (Exactly same samples & class)
Setting 2: Tr1 =/= Te1 (Same class, different samples)

"""

import os
import json
import numpy as np
from core_functions.vision import accuracy
from utils import calc_cl_metrics, prepare_batch

setting = 1

default_params = {
    "adapt_steps": 1,
    "inner_lr": 0.1,
    "n_tasks": 10
}


def run_cl_exp(path, maml, loss, tasks, device, ways, shots, cl_params=default_params, features=None):
    cl_path = path + '/cl_exp'
    if os.path.exists(cl_path):
        ans = input('Overriding previous results! Are you sure? (y/n)')
        if ans == 'n':
            exit(0)
    else:
        os.mkdir(cl_path)

    # Randomly select some batches for training and evaluation
    tasks_pool = []
    for task_i in range(cl_params['n_tasks']):
        batch = tasks.sample()
        adapt_d, adapt_l, eval_d, eval_l = prepare_batch(batch, shots, ways, device, features=features)

        task = {'adapt': (adapt_d, adapt_l)}

        if setting == 1:
            task['eval'] = (adapt_d, adapt_l)
        else:
            task['eval'] = (eval_d, eval_l)

        tasks_pool.append(task)

    # Matrix R NxN of accuracies in tasks j after trained on a tasks i (x_axis = test tasks, y_axis = train tasks)
    acc_matrix = np.zeros((cl_params['n_tasks'], cl_params['n_tasks']))

    # Training loop
    for i, task_i in enumerate(tasks_pool):
        adapt_i_data, adapt_i_labels = task_i['adapt']
        adapt_i_data, adapt_i_labels = adapt_i_data.to(device), adapt_i_labels.to(device)

        learner = maml.clone()
        # Adapt to task i
        for step in range(cl_params['adapt_steps']):
            train_error = loss(learner(adapt_i_data), adapt_i_labels)
            learner.adapt(train_error)

        # Evaluation loop
        for j, task_j in enumerate(tasks_pool):
            eval_j_data, eval_j_labels = task_j['eval']
            eval_j_data, eval_j_labels = eval_j_data.to(device), eval_j_labels.to(device)

            predictions = learner(eval_j_data)
            valid_accuracy_j = accuracy(predictions, eval_j_labels)

            acc_matrix[i, j] = valid_accuracy_j  # Accuracy on task j after trained on task i

    cl_res = calc_cl_metrics(acc_matrix)

    save_acc_matrix(cl_path, acc_matrix)
    with open(cl_path + '/cl_params.json', 'w') as fp:
        json.dump(cl_params, fp, sort_keys=True, indent=4)

    with open(cl_path + '/cl_res.json', 'w') as fp:
        json.dump(cl_res, fp, sort_keys=True, indent=4)

    return acc_matrix, cl_res


def save_acc_matrix(path, acc_matrix):
    print('Saving accuracy matrix..')
    print(acc_matrix)
    np.savetxt(path + '/acc_matrix.out', acc_matrix, fmt='%1.2f')
