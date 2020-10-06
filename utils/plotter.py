#!/usr/bin/env python3

import matplotlib.pyplot as plt
import json
import numpy as np

filename = "results/maml_min_05_03_17h13_42_2368/metrics.json"


def plot_dict(a_dict, save=False):
    title = a_dict['title']
    x_legend = a_dict['x_legend']
    y_legend = a_dict['y_legend']
    y_axis = a_dict['y_axis']

    plt.title(title)
    plt.xlabel(x_legend)
    plt.ylabel(y_legend)

    for label, values in y_axis.items():
        plt.plot(values, label=label)

    if save:
        path = a_dict['path']
        plt.savefig(path)
    plt.show()


def plot_dict_explicit(a_dict, save=False):
    title = a_dict['title']
    x_legend = a_dict['x_legend']
    y_legend = a_dict['y_legend']
    y_axis = a_dict['y_axis']
    x_axis = a_dict['x_axis']

    plt.title(title)
    plt.xlabel(x_legend)
    plt.ylabel(y_legend)

    plt.plot(x_axis, y_axis, linestyle='-', marker='o', alpha=0.7)

    if save:
        path = a_dict['path']
        plt.savefig(path)
    plt.show()


def bar_plot_ml10(results):
    x_axis = []
    trial1 = []
    trial1_suc = []
    trial2 = []
    trial2_suc = []
    trial3 = []
    trial3_suc = []

    # results['lever-pull'] = [x * -1 for x in results['lever-pull']]

    for key, val in results.items():
        x_axis.append(key)
        trial1.append(val[0])
        trial1_suc.append('green' if val[1] > 0.9 else 'red')

        trial2.append(val[2])
        trial2_suc.append('green' if val[3] > 0.9 else 'red')

        trial3.append(val[4])
        trial3_suc.append('green' if val[5] > 0.9 else 'red')

    x = np.arange(len(x_axis))
    width = 0.2

    r1 = np.arange(len(trial1))
    r2 = [x + width for x in r1]
    r3 = [x + width for x in r2]

    fig, ax = plt.subplots()
    bar1 = ax.barh(r1, trial1, width, color=trial1_suc, edgecolor='white', label='Trial 1')
    bar2 = ax.barh(r2, trial2, width, color=trial2_suc, edgecolor='white', label='Trial 2')
    bar3 = ax.barh(r3, trial3, width, color=trial3_suc, edgecolor='white', label='Trial 3')

    ax.set_xlabel('Rewards')
    ax.set_title('Meta-Testing Performance on ML10')
    ax.set_yticks(x)
    ax.set_yticklabels(x_axis)
    ax.set_xlim([-10, 70000])
    # ax.legend()
    plt.xscale('symlog')
    fig.tight_layout()
    plt.show()


def bar_plot_ml10_one_task(results):
    trials = []
    trials_suc = []
    fig, ax = plt.subplots()

    for key, val in results.items():
        trials.append(val[0])
        trials.append(val[2])
        trials.append(val[4])

        trials_suc.append('green' if val[1] > 0.9 else 'red')
        trials_suc.append('green' if val[3] > 0.9 else 'red')
        trials_suc.append('green' if val[5] > 0.9 else 'red')

    y_labels = ['Trial 1', 'Trial 2', 'Trial 3']
    y_pos = np.arange(len(y_labels))

    ax.barh(y_pos, trials, color=trials_suc, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Performance')
    ax.set_title('How fast do you want to go today?')

    plt.show()


def plot_list(a_list, path="plot.png", save=False):
    plt.plot(a_list)
    if save:
        plt.savefig(path)
    plt.show()


def plot_from_json():
    with open(filename) as f:
        metrics = json.load(f)

    y_axis = metrics['0']
