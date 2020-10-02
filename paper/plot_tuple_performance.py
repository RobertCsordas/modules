#!/usr/bin/env python3

import lib
from lib import StatTracker
from lib.common import group, calc_stat
import collections
import matplotlib.pyplot as plt
import os

plots = collections.OrderedDict()
plots["Full"] = "final_accuracy/0/iid/accuracy/tuple/"
plots["Pair 1"] = "final_accuracy/1/iid/accuracy/tuple/"
plots["$\\neg$Pair 1"] = "inverse_mask_test/split_1/iid/accuracy/tuple/"
plots["Pair 2"] = "final_accuracy/2/iid/accuracy/tuple/"
plots["$\\neg$Pair 2"] = "inverse_mask_test/split_2/iid/accuracy/tuple/"

names = list(plots.keys())

BASE_DIR = "out/tuple_performance"

def plot_one(stats):
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect=0.03)
    for t in range(2):
        this_stats = [stats[f"{plots[n]}{t}"].get() for n in names]
        means = [s.mean * 100 for s in this_stats]
        std = [s.std * 100 for s in this_stats]
        plt.bar([3 * r + t for r in range(len(names))], means, yerr=std, align='center')

    plt.xticks([3 * r + 0.5 for r in range(len(names))], names)
    return fig


def plot_all(name, groups):
    stats = calc_stat(groups, lambda k: (k.startswith("final_accuracy/") and '/iid/accuracy/' in k and '/tuple/' in k) or (k.startswith("inverse_mask_test/") and '/iid/accuracy/' in k))
    for k, s in stats.items():
        print("---------------------", k)
        for n in s.keys():
           print(n)

    for k, s in stats.items():
        fig = plot_one(s)
        fname = os.path.join(BASE_DIR, name, f"{k}.pdf")
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        fig.savefig(fname, bbox_inches='tight')


def plot_both(ff, rnn):
    assert len(ff) == 10
    assert len(rnn) == 10
    ff_stats = calc_stat({"a":ff}, lambda k: (k.startswith("final_accuracy/") and '/iid/accuracy/' in k and '/tuple/' in k) or (k.startswith("inverse_mask_test/") and '/iid/accuracy/' in k))["a"]
    rnn_stats = calc_stat({"a":rnn}, lambda k: (k.startswith("final_accuracy/") and '/iid/accuracy/' in k and '/tuple/' in k) or (k.startswith("inverse_mask_test/") and '/iid/accuracy/' in k))["a"]

    fig = plt.figure(figsize=[4,0.95])
    # ax = fig.add_subplot(111, aspect=0.07)

    for t in range(2):
        this_ff_stats = [ff_stats[f"{plots[n]}{t}"].get() for n in names]
        means_ff = [s.mean * 100 for s in this_ff_stats]
        std_ff = [s.std * 100 for s in this_ff_stats]
        plt.bar([5.5 * r + t * 2.5 for r in range(len(names))], means_ff, yerr=std_ff, align='center')

    for t in range(2):
        this_rnn_stats = [rnn_stats[f"{plots[n]}{t}"].get() for n in names]
        means_rnn = [s.mean * 100 for s in this_rnn_stats]
        std_rnn = [s.std * 100 for s in this_rnn_stats]
        plt.bar([5.5 * r + 1+ t * 2.5 for r in range(len(names))], means_rnn, yerr=std_rnn, align='center')

    plt.xticks([5.5 * r + 1.75 for r in range(len(names))], names)
    plt.ylabel("Accuracy [\\%]")

    # plt.legend(["F1", "F2", "R1", "R2"], bbox_to_anchor=(1.1, 1.05))

    fname = f"{BASE_DIR}/tuple_performance.pdf"
    fig.axes[0].yaxis.set_label_coords(-0.12,0.4)
    fig.savefig(fname, bbox_inches='tight', pad_inches = 0.01)

rnn_runs = lib.get_runs(["tuple_rnn"])
feedforward_runs = lib.get_runs(["tuple_feedforward_big"])

feedforward_runs = group(feedforward_runs, ["layer_sizes"])
rnn_runs = group(rnn_runs, ["tuple.mode"])
plot_all("rnn", rnn_runs)
plot_all("feedforward", feedforward_runs)

plot_both(feedforward_runs["layer_sizes_2000,2000,2000,2000"], rnn_runs["tuple.mode_together"])
