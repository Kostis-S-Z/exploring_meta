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

from core_functions.rl import ppo_update, vpg_a2c_loss, trpo_update
from utils import plot_dict
from utils import get_cca_similarity, get_linear_CKA, get_kernel_CKA

metrics = []


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
    global metrics
    metrics = rep_params['metrics']

    rep_path = path + '/rep_exp'
    if not os.path.isdir(rep_path):
        os.mkdir(rep_path)

    # An instance of the model before adaptation
    init_model = deepcopy(policy)
    adapt_model = deepcopy(policy)

    sanity_check(env, init_model, adapt_model)
    del adapt_model

    # column 0: adaptation results, column 1: init results
    acc_results = np.zeros((rep_params['n_tasks'], 2))
    # Create a dictionary of layer : results for each metric (e.g cca_results["0"] = [0.3, 0.2, 0.1])
    cca_results = {str(layer): [] for layer in rep_params['layers']}
    cka_l_results = {str(layer): [] for layer in rep_params['layers']}
    cka_k_results = {str(layer): [] for layer in rep_params['layers']}

    tasks = env.sample_tasks(rep_params['n_tasks'])

    for task in tasks:

        # Sample task
        env.set_task(task)
        env.reset()
        task_i = ch.envs.Runner(env)

        before_adapt_model = deepcopy(policy)
        after_adapt_model = deepcopy(policy)

        for step in range(rep_params['adapt_steps']):
            # Adapt the model to support episodes
            adapt_ep = task_i.run(before_adapt_model, episodes=rep_params['adapt_batch_size'])

            if rep_params['algo'] == 'vpg':
                # Calculate loss & fit the value function
                inner_loss = vpg_a2c_loss(adapt_ep, after_adapt_model, baseline, rep_params['gamma'], rep_params['tau'])
                # Adapt model based on the loss
                after_adapt_model.adapt(inner_loss, allow_unused=rep_params['anil'])
            elif rep_params['algo'] == 'ppo':
                # Calculate loss & fit the value function & update the policy
                ppo_update(adapt_ep, after_adapt_model, baseline, rep_params, anil=rep_params['anil'])
            else:
                after_adapt_model = trpo_update(adapt_ep, after_adapt_model, baseline,
                                                rep_params['inner_lr'], rep_params['gamma'], rep_params['tau'],
                                                anil=rep_params['anil'])

            # Compare representations with initial model
            init_mean_change, init_var_change = episode_mean_var(adapt_ep, init_model, after_adapt_model)
            # Compare representations before & after adaptation
            adapt_mean_change, adapt_var_change = episode_mean_var(adapt_ep, before_adapt_model, after_adapt_model)

            print(f'Change between initial and adapted model:'
                  f'\n\t mean: {init_mean_change} | var: {init_var_change}'
                  f'Change between before & after 1 adaptation step model:'
                  f'\n\t mean: {adapt_mean_change} | var: {adapt_var_change}')

            before_adapt_model = deepcopy(after_adapt_model)

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


def episode_mean_var(episode, model_1, model_2):
    """
    Find the mean & variance of the representation difference
    between two models in a series of states of an episode.
    """
    results = []
    mean = {}
    var = {}
    for state in episode.state():
        rep_1 = model_1.get_representation(state)
        rep_2 = model_2.get_representation(state)

        result = calculate_rep_change(rep_1, rep_2)

        results.append(result)

    for metric, values in results.item():
        mean[metric] = np.mean(values)
        var[metric] = np.var(values)
        print(f'{metric} mean: {mean[metric]}')
        print(f'{metric} mean: {var[metric]}')

    return mean, var


def calculate_rep_change(rep_1, rep_2):
    results = {}
    if 'CCA' in metrics:
        results['CCA'] = get_cca_similarity(rep_1, rep_2, epsilon=1e-10)[1]
    if 'CKA_L' in metrics:
        results['CKA_L'] = get_linear_CKA(rep_1, rep_2)
    if 'CKA_K' in metrics:
        results['CKA_K'] = get_kernel_CKA(rep_1, rep_2)

    return results


def get_rep_from_batch(model, batch, layer=4):
    representation = model.get_rep_i(batch, layer)
    representation = representation.detach().numpy()

    # TODO: reshape representation
    return representation
