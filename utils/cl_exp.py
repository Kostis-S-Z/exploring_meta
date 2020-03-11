"""
Evaluate model performance in a Continual Learning setting

Setting 1: Tr1 = Te1 (Exactly same samples & class)
Setting 2: Tr1 =/= Te1 (Same class, different samples)

"""

import numpy as np
from utils import accuracy, calc_cl_metrics, prepare_batch

setting = 2


def run_cl_exp(maml, loss, tasks, device, ways, shots, adapt_steps, n_tasks=5, features=None):

    # Randomly select some batches for training and evaluation
    tasks_pool = []
    for task_i in range(n_tasks):
        batch = tasks.sample()
        adapt_d, adapt_l, eval_d, eval_l = prepare_batch(batch, shots, ways, device, features=features)

        task = {'adapt': (adapt_d, adapt_l)}

        if setting == 1:
            task['eval'] = (adapt_d, adapt_l)
        else:
            task['eval'] = (eval_d, eval_l)

        tasks_pool.append(task)

    # Matrix R NxN of accuracies in tasks j after trained on a tasks i (x_axis = test tasks, y_axis = train tasks)
    acc_matrix = np.zeros((n_tasks, n_tasks))

    # Training loop
    for i, task_i in enumerate(tasks_pool):
        adapt_i_data, adapt_i_labels = task_i['adapt']
        adapt_i_data, adapt_i_labels = adapt_i_data.to(device), adapt_i_labels.to(device)

        learner = maml.clone()
        # Adapt to task i
        for step in range(adapt_steps):
            train_error = loss(learner(adapt_i_data), adapt_i_labels)
            learner.adapt(train_error)

        # Evaluation loop
        for j, task_j in enumerate(tasks_pool):
            eval_j_data, eval_j_labels = task_j['eval']
            eval_j_data, eval_j_labels = eval_j_data.to(device), eval_j_labels.to(device)

            predictions = learner(eval_j_data)
            valid_error = loss(predictions, eval_j_labels)
            valid_accuracy_j = accuracy(predictions, eval_j_labels)

            acc_matrix[i, j] = valid_accuracy_j  # Accuracy on task j after trained on task i

    return acc_matrix, calc_cl_metrics(acc_matrix)
