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

import numpy as np
from core_funtions.vision import accuracy
from utils import prepare_batch, plot_dict
from utils import get_cca_similarity, get_linear_CKA, get_kernel_CKA


def run_rep_exp(model, loss, tasks, device, ways, shots, model_path, n_tasks=5):
    # Ignore labels
    sanity_batch, _ = tasks.sample()
    sanity_batch = sanity_batch.to(device)

    # An instance of the model before adaptation
    init_model = model.clone()
    adapt_model = model.clone()

    init_rep_sanity = get_rep_from_batch(init_model, sanity_batch)

    # column 0: adaptation results, column 1: init results
    acc_results = np.zeros((n_tasks, 2))
    cca_results = {"l2": [], "l3": [], "l4": []}  # {"l0": [], "l1": [], "l2": [], "l3": [], "l4": []}
    cka_l_results = {"l0": [], "l1": [], "l2": [], "l3": [], "l4": []}
    cka_k_results = {"l0": [], "l1": [], "l2": [], "l3": [], "l4": []}

    for task in range(n_tasks):
        batch = tasks.sample()

        adapt_d, adapt_l, eval_d, eval_l = prepare_batch(batch, shots, ways, device)

        # Adapt the model
        for step in range(5):  # self.params['adaptation_steps']):
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
                cka_l_results[layer].append(get_linear_CKA(adapted_rep_i, init_rep_i))
                cka_k_results[layer].append(get_kernel_CKA(adapted_rep_i, init_rep_i))

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
                    path=model_path + "/inner_CCA_evolution.png")
    cka_l_plot = dict(title="Linear CKA Evolution",
                      x_legend="Inner loop steps",
                      y_legend="CKA similarity",
                      y_axis=cka_l_results,
                      path=model_path + "/inner_Linear_CKA_evolution.png")
    cka_k_plot = dict(title="Kernel CKA Evolution",
                      x_legend="Inner loop steps",
                      y_legend="CKA similarity",
                      y_axis=cka_k_results,
                      path=model_path + "/inner_Kernel_CKA_evolution.png")
    plot_dict(cca_plot, save=True)
    plot_dict(cka_l_plot, save=True)
    plot_dict(cka_k_plot, save=True)

    return cca_results


def get_rep_from_batch(model, batch, layer=4):
    representation = model.get_rep_i(batch, layer)
    representation = representation.cpu().detach().numpy()

    batch_size = representation.shape[0]
    conv_neurons = representation.shape[1]
    conv_filters_1 = representation.shape[2]
    conv_filters_2 = representation.shape[3]

    representation = representation.reshape((conv_neurons * conv_filters_1 * conv_filters_2, batch_size))
    return representation
