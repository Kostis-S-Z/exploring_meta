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
    plt.legend()
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
    if 'std' in a_dict:
        plt.errorbar(x_axis, y_axis, yerr=a_dict['std'], fmt='o')

    if save:
        path = a_dict['path']
        plt.savefig(path)
    plt.show()


def bar_plot_ml10(results, path):
    x_axis = []
    trial1 = []
    trial1_suc = []
    trial1_col = []
    trial2 = []
    trial2_suc = []
    trial2_col = []
    trial3 = []
    trial3_suc = []
    trial3_col = []
    from collections import OrderedDict
    for key, val in OrderedDict(sorted(results.items(), reverse=True)).items():
        x_axis.append(key)
        trial1.append(val[0])
        trial1_suc.append(val[1])
        trial1_col.append('green' if val[1] > 0.5 else 'red')

        trial2.append(val[2])
        trial2_suc.append(val[3])
        trial2_col.append('green' if val[3] > 0.5 else 'red')

        trial3.append(val[4])
        trial3_suc.append(val[5])
        trial3_col.append('green' if val[5] > 0.5 else 'red')

    x = np.arange(len(x_axis))
    width = 0.25

    r1 = np.arange(len(trial1))
    r2 = [x + width for x in r1]
    r3 = [x + width for x in r2]

    fig, ax = plt.subplots()
    bar1 = ax.barh(r1, trial1, width, color=trial1_col, edgecolor='white', label='Trial 1')
    bar2 = ax.barh(r2, trial2, width, color=trial2_col, edgecolor='white', label='Trial 2')
    bar3 = ax.barh(r3, trial3, width, color=trial3_col, edgecolor='white', label='Trial 3')

    for i, v in enumerate(trial1):
        vp = -8 if v < 0 else 1000
        ax.text(v + vp, i - 0.1, str(int(trial1_suc[i] * 100)) + '%', fontsize='x-small')
    for i, v in enumerate(trial2):
        vp = -8 if v < 0 else 1000
        ax.text(v + vp, i + 0.2, str(int(trial2_suc[i] * 100)) + '%', fontsize='x-small')
    for i, v in enumerate(trial3):
        vp = -8 if v < 0 else 1000
        ax.text(v + vp, i + 0.5, str(int(trial3_suc[i] * 100)) + '%', fontsize='x-small')

    ax.set_xlabel('Rewards')
    ax.set_title('Meta-Testing Performance on ML10')
    ax.set_yticks(x)
    ax.set_yticklabels(x_axis)
    # ax.set_xlim([-50, 500000])
    # ax.set_xlim([0, 1000000])

    plt.xscale('symlog')
    plt.tight_layout()
    # you can also log the plots into wandb!
    # wandb.log({'chart':plt})
    plt.savefig(path)
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
    # ax.set_xlabel('Performance')

    plt.show()


def plot_list(a_list, path="plot.png", save=False):
    plt.plot(a_list)
    if save:
        plt.savefig(path)
    plt.show()

