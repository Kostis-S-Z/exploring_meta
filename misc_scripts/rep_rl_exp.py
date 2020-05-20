"""
Measure how much the representation changes during evaluation


Setting:
    1: adapt_d -> On the data the adaptation model was adapted
    2: eval_d -> On the data the models where evaluated
    3: different batch -> On a completely different batch of data
"""

import os
import json
import numpy as np
from core_functions.vision import accuracy
from utils import prepare_batch, plot_dict
from utils import get_cca_similarity, get_linear_CKA, get_kernel_CKA
from tqdm import tqdm


def run_rep_exp(path, model, loss, tasks, device, ways, shots, rep_params):
    rep_path = path + '/rep_exp'
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

            # TODO: We want to compare representations / weights
            # what is the difference with the activations? -> weights vs activations?

            # Get their representations for every layer
            for i, layer in enumerate(cca_results.keys()):
                adapted_rep_i = get_rep_from_batch(adapt_model, adapt_d, i + 2)
                init_rep_i = get_rep_from_batch(init_model, adapt_d, i + 2)

                cca_results[layer].append(get_cca_similarity(adapted_rep_i.T, init_rep_i.T, epsilon=1e-10)[1])
                # NOTE: Currently CKA takes too long to compute so leave it out
                # cka_l_results[layer].append(get_linear_CKA(adapted_rep_i, init_rep_i))
                # cka_k_results[layer].append(get_kernel_CKA(adapted_rep_i, init_rep_i))

            acc_results[task, 0] = a_valid_acc
            acc_results[task, 1] = i_valid_acc

    # print("We expect that column 0 has higher values than column 1")
    # print(acc_results)
    # print("We expect that the values decrease over time?")
    # print("CCA:", cca_results)
    # print("We expect that the values decrease over time?")
    # print("linear CKA:", cka1_results)
    # print("We expect that the values decrease over time?")
    # print("Kernerl CKA:", cka2_results)

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
    plot_dict(cca_plot, save=True)
    # plot_dict(cka_l_plot, save=True)
    # plot_dict(cka_k_plot, save=True)

    with open(rep_path + '/rep_params.json', 'w') as fp:
        json.dump(rep_params, fp, sort_keys=True, indent=4)

    with open(rep_path + '/cca_results.json', 'w') as fp:
        json.dump(cca_results, fp, sort_keys=True, indent=4)

    return cca_results


def get_rep_from_batch(model, batch, layer=4):
    representation = model.get_rep_i(batch, layer)
    representation = representation.cpu().detach().numpy()

    # TODO: reshape representation
    return representation