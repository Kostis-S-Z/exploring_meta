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
import statistics
import torch
from copy import deepcopy
from collections import defaultdict, OrderedDict

from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

from core_functions.policies import DiagNormalPolicy
from learn2learn.algorithms import MAML

from core_functions.rl import vpg_a2c_loss, trpo_update, single_ppo_update, get_ep_successes, ML10_eval_task_names
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
    cca_results = {str(layer): [] for layer in rep_params['layers']}
    cka_l_results = {str(layer): [] for layer in rep_params['layers']}
    cka_k_results = {str(layer): [] for layer in rep_params['layers']}

    env = make_env(env_name, 1, rep_params['seed'], test=True, max_path_length=rep_params['max_path_length'])
    if rep_params['eval_each_task']:
        tasks = sample_from_each_task(env)
    else:
        tasks = env.sample_tasks(rep_params['n_tasks'])

    # Measure changes (mean and variance) of a specific layer across steps from the initial model and ith model
    init_mean = defaultdict(list)
    init_var = defaultdict(list)
    # Measure changes (mean and variance) of a specific layer across steps from the (i-1)th model and ith model
    adapt_mean = defaultdict(list)
    adapt_var = defaultdict(list)

    # Average layer changes across all tasks
    av_layer_changes_mean = defaultdict(list)
    av_layer_changes_std = defaultdict(list)

    for task in tasks:
        print(f'Adapting on Task: {ML10_eval_task_names[task["task"]]}')
        # Sample task
        env.set_task(task)
        env.reset()
        task_i = Runner(env)

        before_adapt_model = deepcopy(policy)  # for step 0: before adapt == init model
        after_adapt_model = deepcopy(policy)

        for step in range(rep_params['adapt_steps']):
            # Adapt the model to support episodes
            adapt_ep = task_i.run(before_adapt_model, episodes=rep_params['adapt_batch_size'])

            if step == 0:
                performance_before = get_ep_successes(adapt_ep, rep_params['max_path_length']) / rep_params[
                    'adapt_batch_size']

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

            performance_after = get_ep_successes(adapt_ep, rep_params['max_path_length']) / rep_params[
                'adapt_batch_size']

            """ ACROSS STEPS """
            i_m_change, i_v_change, a_m_change, a_v_change = change_across_steps(adapt_ep, init_model,
                                                                                 before_adapt_model, after_adapt_model,
                                                                                 step)
            for metric in metrics:
                init_mean[metric] += [i_m_change[metric]]
                init_var[metric] += [i_v_change[metric]]
                adapt_mean[metric] += [a_m_change[metric]]
                adapt_var[metric] += [a_v_change[metric]]

            before_adapt_model = after_adapt_model.clone()

        """ ACROSS LAYERS """
        layer_changes = change_across_layers(rep_params['layers'], adapt_ep, before_adapt_model, after_adapt_model)
        for layer, changes in layer_changes.items():
            av_layer_changes_mean[layer] += [changes[0]['CCA']]
            av_layer_changes_std[layer] += [changes[1]['CCA']]
        print(f'Performance before: {performance_before}\nPerformance after: {performance_after}')

        """ ACROSS LAYERS PER TASK """
        # for metric in metrics:
        #     plot_sim_across_layers(layer_changes, metric)

    """ ACROSS LAYERS AVERAGE """
    for layer, changes in av_layer_changes_mean.items():
        av_layer_changes_mean[layer] = statistics.mean(changes)
        av_layer_changes_std[layer] = statistics.stdev(changes)

    print(av_layer_changes_mean)
    print(av_layer_changes_std)

    plot_sim_across_layers_average(av_layer_changes_mean, av_layer_changes_std,
                                   title='Before / After adaptation on the ML10 test tasks')
    """ ACROSS STEPS """
    # for metric in metrics:
    #     plot_sim_across_steps(init_mean[metric], init_var[metric], metric=metric,
    #                           title='Similarity between init and adapted (in %)')
    #     difference = [1 - x for x in adapt_mean[metric]]
    #     plot_sim_across_steps(difference, adapt_var[metric], metric=metric,
    #                           title='Representation difference after each step (in %)')

    """
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


def change_across_layers(layers, adapt_ep, before_adapt_model, after_adapt_model):
    layer_changes = {}
    for layer in layers:
        # Compare representations with initial model
        m, v = episode_mean_var(adapt_ep, before_adapt_model, after_adapt_model, layer)
        layer_changes[layer] = [m, v]

    return layer_changes


def change_across_steps(adapt_ep, init_model, before_adapt_model, after_adapt_model, step):
    # Compare representations with initial model
    init_mean_change, init_var_change = episode_mean_var(adapt_ep, init_model, after_adapt_model)
    # Compare representations before & after adaptation
    adapt_mean_change, adapt_var_change = episode_mean_var(adapt_ep, before_adapt_model, after_adapt_model)
    print(f'\nSimilarity between initial and adapted model after {step + 1} steps:'
          f'\n\t mean: {init_mean_change} | var: {init_var_change}'
          f'\nSimilarity between before & after 1 adaptation step model:'
          f'\n\t mean: {adapt_mean_change} | var: {adapt_var_change}')
    return init_mean_change, init_var_change, adapt_mean_change, adapt_var_change


def episode_mean_var(episode, model_1, model_2, layer=3):
    """
    Find the mean & variance of the representation difference
    between two models in a series of states of an episode.
    """
    episode_results = defaultdict(list)
    for state in episode.state():
        rep_1 = get_state_representation(model_1, state, layer)
        rep_2 = get_state_representation(model_2, state, layer)

        result = calculate_rep_change(rep_1, rep_2)

        # Append results to dictionary with list for each metric
        for metric, value in result.items():
            episode_results[metric] += [value]

    mean = {}
    var = {}
    for metric, values in episode_results.items():
        mean[metric] = statistics.mean(values)
        var[metric] = statistics.stdev(values)
        # print(f'{metric} mean: {mean[metric]}')
        # print(f'{metric} var: {var[metric]}')

    return mean, var


def calculate_rep_change(rep_1, rep_2):
    change_results = {}
    if 'CCA' in metrics:
        change_results['CCA'] = get_cca_similarity(rep_1.T, rep_2.T, epsilon=1e-10)[1]
    if 'CKA_L' in metrics:
        change_results['CKA_L'] = get_linear_CKA(rep_1, rep_2)
    if 'CKA_K' in metrics:
        change_results['CKA_K'] = get_kernel_CKA(rep_1, rep_2)

    return change_results


def get_state_representation(model, state, layer_i=3):
    if layer_i == -1:
        representation = model(state)
    else:
        representation = model.get_representation(state, layer_i)
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
        plot_sim_across_steps(init_change_m[metric], init_change_v[metric], metric=metric,
                              title='Similarity between init and adapted (in %)')

    for metric in metrics:
        difference = [1 - x for x in adapt_change_m[metric]]
        plot_sim_across_steps(difference, adapt_change_v[metric], metric=metric,
                              title='Representation difference after each step (in %)')


def plot_sim_across_layers(changes_per_layer, metric='CCA', title=''):
    plt.figure().gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # Set integers only in x ticks

    plt.title(title)
    plt.xlabel('Layers')
    plt.ylabel(f'{metric} Similarity')

    # Order layers for clear visibility
    changes_per_layer = OrderedDict(sorted(changes_per_layer.items(), reverse=True))
    x_axis = range(len(changes_per_layer.keys()))
    plt.xticks(x_axis, ('L1', 'L2', 'Head'))
    y_axis_mean = [e[0][metric] for e in changes_per_layer.values()]
    y_err_var = [e[1][metric] for e in changes_per_layer.values()]
    plt.errorbar(x_axis, y_axis_mean, yerr=y_err_var, marker='o')
    # plt.legend()
    plt.show()


def plot_sim_across_layers_average(changes_per_layer_mean, changes_per_layer_std, title=''):
    # Order layers for clear visibility
    changes_per_layer_m = OrderedDict(sorted(changes_per_layer_mean.items(), reverse=True))
    changes_per_layer_s = OrderedDict(sorted(changes_per_layer_std.items(), reverse=True))
    x_axis = range(len(changes_per_layer_m.keys()))
    y_axis_mean = list(changes_per_layer_m.values())
    y_err_var = list(changes_per_layer_s.values())

    plt.figure().gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # Set integers only in x ticks

    plt.title(title)
    plt.xlabel('Layers')
    plt.ylabel(f'CCA Similarity')
    plt.xticks(x_axis, ('L1', 'L2', 'Head'))
    plt.plot(x_axis, y_axis_mean, linestyle='-', marker='o', alpha=0.7)
    plt.errorbar(x_axis, y_axis_mean, yerr=y_err_var, fmt='o')
    # plt.legend()
    plt.show()


def plot_sim_across_steps(r_mean, r_var, metric='CCA', title=''):
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


def sample_from_each_task(env):
    task_list = env.sample_tasks(50)
    check = defaultdict(list)
    for i, k in enumerate(task_list):
        check[k['task']] += [i]

    final_task_list = []
    for key, val in check.items():
        for sample in val[:1]:
            final_task_list.append(task_list[sample])

    return final_task_list
