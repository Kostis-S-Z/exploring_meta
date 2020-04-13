#!/usr/bin/env python3

import matplotlib.pyplot as plt
import json
import numpy as np
import scipy.stats
from collections import OrderedDict


def plot_from_json():
    values = []
    for i, filename in enumerate([filename_1, filename_2, filename_3]):
        with open(filename) as f:
            a_dict = json.load(f)

        x_axis = []
        y_axis = []
        for key in sorted(a_dict.keys(), key=lambda key: int(key.split("model", 2)[-1][1:-3])):
            key_as_int = int(key.split("model", 2)[-1][1:-3])
            if key_as_int < 20000:
                x_axis.append(key_as_int)
                y_axis.append(a_dict[key])

        values.append(y_axis)

    plt.xlabel("Checkpoints")
    plt.ylabel("Test Accuracy")

    for i, values_i in enumerate(values):
        plt.plot(x_axis, values_i, '-o', label=f'seed_{i+1}')
    # plt.savefig("test_checkpoints_seeds.png")
    plt.show()


def plot_with_confidence():
    all_vals = OrderedDict()
    for i, filename in enumerate([filename_1, filename_2, filename_3]):
        with open(filename) as f:
            a_dict = json.load(f)

        for key in sorted(a_dict.keys(), key=lambda key: int(key.split("model", 2)[-1][1:-3])):
            key_as_int = int(key.split("model", 2)[-1][1:-3])
            if key_as_int < 20000:
                if key_as_int in all_vals:
                    all_vals[key_as_int].append(a_dict[key])
                else:
                    all_vals[key_as_int] = [a_dict[key]]

    mean, std = get_mean_and_std(all_vals)
    x_axis = list(all_vals.keys())

    plt.xlabel("Checkpoints")
    plt.ylabel("Test Accuracy")

    plt.plot(x_axis, mean)
    plt.fill_between(x_axis, mean - std, mean + std, alpha=0.3)
    # plt.savefig("test_checkpoints_seeds_conf_0_5.png")
    plt.show()


def get_mean_and_std(results, confidence=0.5):
    mean = []
    std = []
    for key, values in results.items():
        n = len(values)
        se = scipy.stats.sem(values)
        std_c = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)

        mean.append(np.mean(values))
        std.append(std_c)

    mean = np.array(mean)
    std = np.array(std)
    return mean, std


if __name__ == '__main__':
    fig_path = "/home/kosz/Projects/KTH/Thesis/exploring_meta/results/seed_check/compare_iter.png"
    filename_1 = "/home/kosz/Projects/KTH/Thesis/exploring_meta/results/" \
                 "seed_check/maml_min_23_03_16h08_1_1215/ckpnt_results.json"
    filename_2 = "/home/kosz/Projects/KTH/Thesis/exploring_meta/results/" \
                 "seed_check/maml_min_24_03_12h33_2_2646/ckpnt_results.json"
    filename_3 = "/home/kosz/Projects/KTH/Thesis/exploring_meta/results/" \
                 "seed_check/maml_min_24_03_12h33_3_6449/ckpnt_results.json"
    plot_from_json()
    plot_with_confidence()
