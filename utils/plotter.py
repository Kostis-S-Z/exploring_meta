#!/usr/bin/env python3

import matplotlib.pyplot as plt
import json

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


def plot_list(a_list, path="plot.png", save=False):
    plt.plot(a_list)
    if save:
        plt.savefig(path)
    plt.show()


def plot_from_json():
    with open(filename) as f:
        metrics = json.load(f)

    y_axis = metrics['0']
