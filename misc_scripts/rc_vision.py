"""
Measure how much the representation changes during evaluation

Features from:
    # adapted_rep_0 = adapt_model.get_rep_i(adapt_d, 0)  # Conv1
    # adapted_rep_1 = adapt_model.get_rep_i(adapt_d, 1)  # Conv2
    # adapted_rep_2 = adapt_model.get_rep_i(adapt_d, 2)  # Conv3
    # adapted_rep_3 = adapt_model.get_rep_i(adapt_d, 3)  # Conv4
    # adapted_rep_4 = adapt_model.get_rep_i(adapt_d, 4)  # Fully connected (same as get_rep)\

Setting:
    1: adapt_d -> On the data the adaptation model was adapted
    2: eval_d -> On the data the models where evaluated
    3: different batch -> On a completely different batch of data
"""

import os
import json
import numpy as np
import statistics
from core_functions.vision import accuracy
from utils import prepare_batch, plot_dict, plot_dict_explicit
from utils import get_cca_similarity, get_linear_CKA, get_kernel_CKA
from tqdm import tqdm

default_params = {
    "adapt_steps": 1,
    "inner_lr": 0.1,
    "n_tasks": 10,
    "layers": [0, 1, 2, 3, 4]
}


def run_rep_exp(path, model, loss, tasks, device, ways, shots, rep_params=default_params, features=None):
    rep_path = path + '/rep_exp'
    if os.path.exists(rep_path):
        ans = input('Overriding previous results! Are you sure? (y/n) ')
        if ans == 'n':
            exit(0)
    else:
        os.mkdir(rep_path)

    # Ignore labels
    sanity_batch, _ = tasks.sample()
    sanity_batch = sanity_batch.to(device)

    # An instance of the model before adaptation
    init_model = model.clone()
    adapt_model = model.clone()

    init_rep_sanity = get_rep_from_batch(init_model, sanity_batch)

    # column 0: adaptation results, column 1: init results
    acc_results = np.zeros((rep_params['n_tasks'], 2))
    # Create a dictionary of layer : results for each metric (e.g cca_results["0"] = [0.3, 0.2, 0.1])
    cca_results = {str(layer): [] for layer in rep_params['layers']}
    cka_l_results = {str(layer): [] for layer in rep_params['layers']}
    cka_k_results = {str(layer): [] for layer in rep_params['layers']}

    for task in tqdm(range(rep_params['n_tasks']), desc="Tasks"):

        batch = tasks.sample()

        adapt_d, adapt_l, eval_d, eval_l = prepare_batch(batch, shots, ways, device)

        # Adapt the model
        for step in range(rep_params['adapt_steps']):
            train_error = loss(adapt_model(adapt_d), adapt_l)
            train_error /= len(adapt_d)
            adapt_model.adapt(train_error)

            # Evaluate the adapted model
            a_predictions = adapt_model(eval_d)
            a_valid_acc = accuracy(a_predictions, eval_l)

            # Evaluate the init model
            i_predictions = init_model(eval_d)
            i_valid_acc = accuracy(i_predictions, eval_l)

            acc_results[task, 0] = a_valid_acc
            acc_results[task, 1] = i_valid_acc

        # Get their representations for every layer
        for layer in cca_results.keys():
            adapted_rep_i = get_rep_from_batch(adapt_model, adapt_d, int(layer))
            init_rep_i = get_rep_from_batch(init_model, adapt_d, int(layer))

            cca_results[layer].append(get_cca_similarity(adapted_rep_i.T, init_rep_i.T, epsilon=1e-10)[1])
            # NOTE: Currently CKA takes too long to compute so leave it out
            # cka_l_results[layer].append(get_linear_CKA(adapted_rep_i, init_rep_i))
            # cka_k_results[layer].append(get_kernel_CKA(adapted_rep_i, init_rep_i))

    # Average and calculate standard deviation
    cca_mean = []
    cca_std = []
    for layer, values in cca_results.items():
        mean = statistics.mean(values)
        std = statistics.stdev(values)
        cca_mean.append(mean)
        cca_std.append(std)

    cca_plot = dict(title="CCA Evolution",
                    x_legend="Inner loop steps",
                    y_legend="CCA similarity",
                    y_axis=cca_results,
                    path=path + "/inner_CCA_evolution.png")
    cka_l_plot = dict(title="Linear CKA Evolution",
                      x_legend="Inner loop steps",
                      y_legend="CKA similarity",
                      y_axis=cka_l_results,
                      path=path + "/inner_Linear_CKA_evolution.png")
    cka_k_plot = dict(title="Kernel CKA Evolution",
                      x_legend="Inner loop steps",
                      y_legend="CKA similarity",
                      y_axis=cka_k_results,
                      path=path + "/inner_Kernel_CKA_evolution.png")
    # plot_dict(cca_plot, save=True)
    # plot_dict(cka_l_plot, save=True)
    # plot_dict(cka_k_plot, save=True)

    with open(rep_path + '/rep_params.json', 'w') as fp:
        json.dump(rep_params, fp, sort_keys=True, indent=4)

    with open(rep_path + '/cca_results.json', 'w') as fp:
        json.dump(cca_results, fp, sort_keys=True, indent=4)

    x_axis = []
    for i in cca_results.keys():
        if int(i) > 0:
            x_axis.append(f'Conv{i}')
        else:
            x_axis.append('Head')
    cca_plot_2 = dict(title="Layer-wise changes before / after adaptation",
                      x_legend="Layer",
                      y_legend="CCA similarity",
                      x_axis=x_axis,
                      y_axis=cca_mean,
                      std=cca_std)

    cka_plot_2 = dict(title="CKA Evolution layer-wise",
                      x_legend="Layer",
                      y_legend="CKA similarity",
                      x_axis=list(cka_l_results.keys()),
                      y_axis=list(cka_l_results.values()))

    plot_dict_explicit(cca_plot_2)
    return cca_results


def get_rep_from_batch(model, batch, layer=4):
    if layer == -1:
        representation = model(batch).cpu().detach().numpy()
    else:
        representation = model.get_rep_i(batch, layer)
        representation = representation.cpu().detach().numpy()

        batch_size = representation.shape[0]
        conv_neurons = representation.shape[1]
        conv_filters_1 = representation.shape[2]
        conv_filters_2 = representation.shape[3]

        representation = representation.reshape((conv_neurons * conv_filters_1 * conv_filters_2, batch_size))
    return representation
