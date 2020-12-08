#!/usr/bin/env python3

from typing import List, Dict
import lib
from lib import StatTracker
from lib.common import group, calc_stat
import collections
import matplotlib.pyplot as plt
import os

plots = collections.OrderedDict()
plots["Full"] = "analyzer/baseline/validation/"
plots["$+$"] = "analyzer/add/validation/"
plots["$\\neg +$"] = "inverse_mask_test/add/"
plots["$*$"] = "analyzer/mul/validation/"
plots["$\\neg *$"] = "inverse_mask_test/mul/"

names = list(plots.keys())

ops = ["add","mul"]


def plot_both(ff, rnn):
    ff_stats = calc_stat({"a":ff}, lambda k: (k.startswith("analyzer/") and k.endswith("/accuracy") and '/validation/' in k) or (k.startswith("inverse_mask_test/") and k.endswith("/accuracy")))["a"]
    rnn_stats = calc_stat({"a":rnn}, lambda k: (k.startswith("analyzer/") and k.endswith("/accuracy") and '/validation/' in k) or (k.startswith("inverse_mask_test/") and k.endswith("/accuracy")))["a"]

    fig = plt.figure(figsize=[6,1.6])

    for t in range(2):
        this_ff_stats = [ff_stats[f"{plots[n]}{ops[t]}/accuracy"].get() for n in names]
        means_ff = [s.mean * 100 for s in this_ff_stats]
        std_ff = [s.std * 100 for s in this_ff_stats]
        plt.bar([5.5 * r + t * 2.5 for r in range(len(names))], means_ff, yerr=std_ff, align='center')

    for t in range(2):
        this_rnn_stats = [rnn_stats[f"{plots[n]}{ops[t]}/accuracy"].get() for n in names]
        means_rnn = [s.mean * 100 for s in this_rnn_stats]
        std_rnn = [s.std * 100 for s in this_rnn_stats]
        plt.bar([5.5 * r + 1+ t * 2.5 for r in range(len(names))], means_rnn, yerr=std_rnn, align='center')

    plt.xticks([5.5 * r + 1.75 for r in range(len(names))], names)
    plt.ylabel("Accuracy [\\%]")
    plt.legend(["FNN $+$", "FNN $*$", "RNN $+$", "RNN $*$"])

    fname = "out/admmul_performance.pdf"
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    fig.savefig(fname, bbox_inches='tight')

    print("\\begin{tabular}{ll|c|cc|cc}")
    print("\\toprule")
    print(" & ".join(["", ""] + names) + " \\\\")
    print("\\midrule")

    row = ["\\multirow{2}{*}{FNN}"]
    for t in range(2):
        this_stats = [ff_stats[f"{plots[n]}{ops[t]}/accuracy"].get() for n in names]
        row.append(f"Pair {t}")

        for m, s in zip([s.mean * 100 for s in this_stats], [s.std * 100 for s in this_stats]):
            row.append(f"${m:.0f} \pm {s:.1f}$")

        print(" & ".join(row) + " \\\\")
        row = [""]

    print("\\midrule")
    row = ["\\multirow{2}{*}{LSTM}"]
    for t in range(2):
        this_stats = [rnn_stats[f"{plots[n]}{ops[t]}/accuracy"].get() for n in names]
        row.append(f"Pair {t}")

        for m, s in zip([s.mean * 100 for s in this_stats], [s.std * 100 for s in this_stats]):
            row.append(f"${m:.0f} \pm {s:.1f}$")

        print(" & ".join(row) + " \\\\")
        row = [""]

    print("\\bottomrule")
    print("\end{tabular}")





rnn_runs = lib.get_runs(["addmul_rnn"])
feedforward_runs = lib.get_runs(["addmul_feedforward_big"])

feedforward_runs = group(feedforward_runs, ["layer_sizes"])
rnn_runs = group(rnn_runs, ["tuple.mode"])
plot_both(feedforward_runs["layer_sizes_2000,2000,2000,2000"], rnn_runs["tuple.mode_together"])
