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
import torch
from copy import deepcopy

import cherry as ch

from core_functions.rl import ppo_update
from utils import plot_dict
from utils import get_cca_similarity, get_linear_CKA, get_kernel_CKA


def sanity_check(env, model_1, model_2):

    # Sample a sanity batch
    env.active_env.random_init = False

    sanity_task = env.sample_tasks(1)

    with torch.no_grad():
        env.set_task(sanity_task[0])
        env.reset()
        env_task = ch.envs.Runner(env)
        init_sanity_ep = env_task.run(model_1, episodes=1)

        env.set_task(sanity_task[0])
        env.reset()
        env_task = ch.envs.Runner(env)
        adapt_sanity_ep = env_task.run(model_2, episodes=1)

        init_san_rew = init_sanity_ep.reward().sum().item()
        adapt_san_rew = adapt_sanity_ep.reward().sum().item()

        print(f'These should be equal: {init_san_rew}={adapt_san_rew}')
        # assert (init_san_rew == adapt_san_rew), "Environment initial states are random"
        init_sanity_state = init_sanity_ep[0].state

        init_rep_sanity = model_1.get_representation(init_sanity_state)
        init_rep_sanity_2 = model_1.get_representation(init_sanity_state, layer=3)

        adapt_rep_sanity = model_2.get_representation(init_sanity_state)
        adapt_rep_sanity_2 = model_2.get_representation(init_sanity_state, layer=3)

        init_rep_array = init_rep_sanity.detach().numpy()
        init_rep_2_array = init_rep_sanity_2.detach().numpy()
        adapt_rep_array = adapt_rep_sanity.detach().numpy()
        adapt_rep_2_array = adapt_rep_sanity_2.detach().numpy()

        assert np.array_equal(init_rep_array, adapt_rep_array), "Representations not identical"
        assert np.array_equal(init_rep_2_array, adapt_rep_2_array), "Representations not identical"


def run_rep_rl_exp(path, env, policy, baseline, rep_params):
    rep_path = path + '/rep_exp'
    if not os.path.isdir(rep_path):
        os.mkdir(rep_path)

    # An instance of the model before adaptation
    init_model = deepcopy(policy)
    adapt_model = deepcopy(policy)

    sanity_check(env, init_model, adapt_model)

    # column 0: adaptation results, column 1: init results
    acc_results = np.zeros((rep_params['n_tasks'], 2))
    # Create a dictionary of layer : results for each metric (e.g cca_results["0"] = [0.3, 0.2, 0.1])
    cca_results = {str(layer): [] for layer in rep_params['layers']}
    cka_l_results = {str(layer): [] for layer in rep_params['layers']}
    cka_k_results = {str(layer): [] for layer in rep_params['layers']}

    tasks = env.sample_tasks(rep_params['n_tasks'])

    for task in range(rep_params['n_tasks']):

        # Sample task
        env.set_task(tasks)
        env.reset()
        task_i = ch.envs.Runner(env)

        # Adapt the model
        for step in range(rep_params['adapt_steps']):
            # Adapt the model to support episodes
            adapt_ep = task_i.run(adapt_model, episodes=rep_params['adapt_batch_size'])
            ppo_update(adapt_ep, adapt_model, baseline, rep_params, anil=rep_params['anil'])

            # Evaluate the adapted model on evaluation episodes

            # Evaluate the init model on evaluation episodes

            # Get their representations for every layer
            for i, layer in enumerate(cca_results.keys()):
                adapted_rep_i = get_rep_from_batch(adapt_model, adapt_ep, i + 2)
                init_rep_i = get_rep_from_batch(init_model, adapt_ep, i + 2)

                cca_results[layer].append(get_cca_similarity(adapted_rep_i.T, init_rep_i.T, epsilon=1e-10)[1])
                # NOTE: Currently CKA takes too long to compute so leave it out
                # cka_l_results[layer].append(get_linear_CKA(adapted_rep_i, init_rep_i))
                # cka_k_results[layer].append(get_kernel_CKA(adapted_rep_i, init_rep_i))

            acc_results[task, 0] = 0  # a_valid_acc
            acc_results[task, 1] = 0  # i_valid_acc

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
    representation = representation.detach().numpy()

    # TODO: reshape representation
    return representation
