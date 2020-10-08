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
from collections import defaultdict

from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

from core_functions.policies import DiagNormalPolicy
from learn2learn.algorithms import MAML

from core_functions.rl import vpg_a2c_loss, trpo_update, single_ppo_update
from core_functions.runner import Runner
from utils import plot_dict
from utils import get_cca_similarity, get_linear_CKA, get_kernel_CKA
from utils import make_env

metrics = []


def sanity_check(env_name, model_1, model_2, rep_params):
    # Sample a sanity batch
    env = make_env(env_name, 1, rep_params['seed'], max_path_length=rep_params['max_path_length'])

    env.active_env.random_init = False

    sanity_task = env.sample_tasks(1)

    with torch.no_grad():
        env.set_task(sanity_task[0])
        env.seed(rep_params['seed'])
        env.reset()
        env_task = Runner(env)
        init_sanity_ep = env_task.run(model_1, episodes=1)

        env.set_task(sanity_task[0])
        env.seed(rep_params['seed'])
        env.reset()
        env_task = Runner(env)
        adapt_sanity_ep = env_task.run(model_2, episodes=1)
        env_task.reset()
        adapt_2_sanity_ep = env_task.run(model_2, episodes=1)

        init_san_rew = init_sanity_ep.reward().sum().item()
        adapt_san_rew = adapt_sanity_ep.reward().sum().item()
        adapt_2_san_rew = adapt_2_sanity_ep.reward().sum().item()

        # print(f'Why are these not equal? They should be equal: {init_san_rew}={adapt_san_rew}={adapt_2_san_rew}')
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

        print(f'Are the representations of the two models for the same state identical? '
              f'{np.array_equal(init_rep_array, adapt_rep_array)}')

        assert np.array_equal(init_rep_array, adapt_rep_array), "Representations not identical"
        assert np.array_equal(init_rep_2_array, adapt_rep_2_array), "Representations not identical"


def run_rep_rl_exp(path, env_name, policy, baseline, rep_params):
    global metrics
    metrics = rep_params['metrics']

    rep_path = path + '/rep_exp'
    if not os.path.isdir(rep_path):
        os.mkdir(rep_path)

    # An instance of the model before adaptation
    init_model = deepcopy(policy)
    adapt_model = deepcopy(policy)

    sanity_check(env_name, init_model, adapt_model, rep_params)
    del adapt_model

    # column 0: adaptation results, column 1: init results
    # acc_results = np.zeros((rep_params['n_tasks'], 2))
    # Create a dictionary of layer : results for each metric (e.g cca_results["0"] = [0.3, 0.2, 0.1])
    # cca_results = {str(layer): [] for layer in rep_params['layers']}
    # cka_l_results = {str(layer): [] for layer in rep_params['layers']}
    # cka_k_results = {str(layer): [] for layer in rep_params['layers']}

    env = make_env(env_name, 1, rep_params['seed'], max_path_length=rep_params['max_path_length'])

    tasks = env.sample_tasks(rep_params['n_tasks'])

    init_mean = defaultdict(list)
    init_var = defaultdict(list)
    adapt_mean = defaultdict(list)
    adapt_var = defaultdict(list)

    for task in tasks:

        # Sample task
        env.set_task(task)
        env.reset()
        task_i = Runner(env)

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
                single_ppo_update(adapt_ep, after_adapt_model, baseline, rep_params, anil=rep_params['anil'])
            else:
                after_adapt_model = trpo_update(adapt_ep, after_adapt_model, baseline,
                                                rep_params['inner_lr'], rep_params['gamma'], rep_params['tau'],
                                                anil=rep_params['anil'])

            # Compare representations with initial model
            init_mean_change, init_var_change = episode_mean_var(adapt_ep, init_model, after_adapt_model)
            # Compare representations before & after adaptation
            adapt_mean_change, adapt_var_change = episode_mean_var(adapt_ep, before_adapt_model, after_adapt_model)

            print(f'\nSimilarity between initial and adapted model after {step + 1} steps:'
                  f'\n\t mean: {init_mean_change} | var: {init_var_change}'
                  f'\nSimilarity between before & after 1 adaptation step model:'
                  f'\n\t mean: {adapt_mean_change} | var: {adapt_var_change}')
            for metric in metrics:
                init_mean[metric] += [init_mean_change[metric]]
                init_var[metric] += [init_var_change[metric]]
                adapt_mean[metric] += [adapt_mean_change[metric]]
                adapt_var[metric] += [adapt_var_change[metric]]

            before_adapt_model = after_adapt_model.clone()

    for metric in metrics:
        plot_sim(init_mean[metric], init_var[metric], metric=metric, title='Similarity between init and adapted (in %)')

    for metric in metrics:
        difference = [1 - x for x in adapt_mean[metric]]
        plot_sim(difference, adapt_var[metric], metric=metric, title='Representation difference after each step (in %)')
    """
    print("We expect that column 0 has higher values than column 1")
    print(acc_results)
    print("We expect that the values decrease over time?")
    print("CCA:", cca_results)
    print("We expect that the values decrease over time?")
    print("linear CKA:", cka1_results)
    print("We expect that the values decrease over time?")
    print("Kernerl CKA:", cka2_results)

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
    """

    with open(rep_path + '/rep_params.json', 'w') as fp:
        json.dump(rep_params, fp, sort_keys=True, indent=4)

    return 0


def episode_mean_var(episode, model_1, model_2, layer=3):
    """
    Find the mean & variance of the representation difference
    between two models in a series of states of an episode.
    """
    results = defaultdict(list)
    for state in episode.state():
        rep_1 = get_state_representation(model_1, state, layer)
        rep_2 = get_state_representation(model_2, state, layer)

        result = calculate_rep_change(rep_1, rep_2)

        # Append results to dictionary with list for each metric
        for metric, value in result.items():
            results[metric] += [value]

    mean = {}
    var = {}
    for metric, values in results.items():
        mean[metric] = np.mean(values)
        var[metric] = np.var(values)
        # print(f'{metric} mean: {mean[metric]}')
        # print(f'{metric} var: {var[metric]}')

    return mean, var


def calculate_rep_change(rep_1, rep_2):
    results = {}
    if 'CCA' in metrics:
        results['CCA'] = get_cca_similarity(rep_1.T, rep_2.T, epsilon=1e-10)[1]
    if 'CKA_L' in metrics:
        results['CKA_L'] = get_linear_CKA(rep_1, rep_2)
    if 'CKA_K' in metrics:
        results['CKA_K'] = get_kernel_CKA(rep_1, rep_2)

    return results


def get_state_representation(model, state, layer=3):
    representation = model.get_representation(state, layer)
    representation = representation.detach().numpy().reshape(-1, 1)

    return representation


def measure_change_through_time(path, env_name, policy, rep_params):
    env = make_env(env_name, 1, rep_params['seed'], max_path_length=rep_params['max_path_length'])
    global metrics
    metrics = ['CCA']

    sanity_task = env.sample_tasks(1)

    with torch.no_grad():
        env.set_task(sanity_task[0])
        env.seed(rep_params['seed'])
        env.reset()
        env_task = Runner(env)
        sanity_ep = env_task.run(policy, episodes=1)

    init_change_m = defaultdict(list)
    init_change_v = defaultdict(list)
    adapt_change_m = defaultdict(list)
    adapt_change_v = defaultdict(list)
    checkpoints = path + f'/model_checkpoints/'
    i = 0

    file_list = os.listdir(checkpoints)
    file_list = [file for file in file_list if 'baseline' not in file]
    models_list = {}
    for file in file_list:
        n_file = file.split('_')[-1]
        n_file = n_file.split('.')[0]
        n_file = int(n_file)
        models_list[n_file] = f'model_{n_file}.pt'

    prev_policy = policy
    for key in sorted(models_list.keys()):
        model_chckpnt = models_list[key]
        if i > 40:
            break
        i += 1

        print(f'Loading {model_chckpnt} ...')
        chckpnt_policy = DiagNormalPolicy(9, 4)
        chckpnt_policy.load_state_dict(torch.load(os.path.join(checkpoints, model_chckpnt)))
        chckpnt_policy = MAML(chckpnt_policy, lr=rep_params['inner_lr'])

        mean, variance = episode_mean_var(sanity_ep, policy, chckpnt_policy, layer=6)
        a_mean, a_variance = episode_mean_var(sanity_ep, prev_policy, chckpnt_policy, layer=6)
        init_change_m['CCA'] += [mean['CCA']]
        init_change_v['CCA'] += [variance['CCA']]
        adapt_change_m['CCA'] += [a_mean['CCA']]
        adapt_change_v['CCA'] += [a_variance['CCA']]

        prev_policy = chckpnt_policy

    for metric in metrics:
        plot_sim(init_change_m[metric], init_change_v[metric], metric=metric,
                 title='Similarity between init and adapted (in %)')

    for metric in metrics:
        difference = [1 - x for x in adapt_change_m[metric]]
        plot_sim(difference, adapt_change_v[metric], metric=metric,
                 title='Representation difference after each step (in %)')


def plot_sim(r_mean, r_var, metric='CCA', title=''):
    plt.figure().gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # Set integers only in x ticks

    plt.title(title)
    plt.xlabel('Adaptation step')
    plt.ylabel(f'{metric} Similarity')

    x_axis = range(1, len(r_mean) + 1)
    y_axis = r_mean
    y_err = r_var
    plt.errorbar(x_axis, y_axis, yerr=y_err, marker='o')
    # plt.legend()
    plt.show()
