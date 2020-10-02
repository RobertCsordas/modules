#!/usr/bin/env python3

import lib
from lib import StatTracker
from lib.common import group, calc_stat
import wandb
import collections
import os

import matplotlib.pyplot as plt

os.makedirs("out", exist_ok=True)

plots = collections.OrderedDict()
plots["Turn Left"] = "analysis_results/turn_left/turn_left/accuracy/total"
plots["Jump"] = "analysis_results/jump/jump/accuracy/total"
plots["Length"] = "analysis_results/length/length/accuracy/total"

refs = {
    "Turn Left": "turn_left",
    "Jump": "jump",
    "Length": "length",
}

names = list(plots.keys())
print(names)


def plot(runs, baseline, fname: str):
    groups = {"runs": runs}

    stats = calc_stat(groups, lambda k: k.startswith("analysis_results/") and "/accuracy/" in k and "/train/" not in k)["runs"]
    baseline_stats = calc_stat(group(baseline, ["scan.train_split"]), lambda k: k.startswith("validation/") and "/accuracy/" in k)
    for k, s in stats.items():
        print(k)

    print("Baseline groups",baseline_stats.keys())

    means = {k: stats[v].get().mean for k, v in plots.items()}
    std = {k: stats[v].get().std for k, v in plots.items()}

    #validation/jump/accuracy/total
    for k, v in refs.items():
        print("----------------================---------------------")
        print(baseline_stats[f"scan.train_split_{v}"])

    ref_stats = {k: baseline_stats[f"scan.train_split_{v}"][f"validation/{v}/accuracy/total"].get() for k, v in refs.items()}

    ref_means = {k: v.mean for k, v in ref_stats.items()}
    ref_std = {k: v.std for k, v in ref_stats.items()}

    fig = plt.figure(figsize=[3,1.5])

    plt.bar([2.25*x for x in range(len(names))], [ref_means[n]*100 for n in names], yerr=[ref_std[n]*100 for n in names], align='center')
    plt.bar([2.25*x+1 for x in range(len(names))], [means[n]*100 for n in names], yerr=[std[n]*100 for n in names], align='center')
    plt.xticks([2.25*x+0.5 for x in range(len(names))], names)
    plt.ylabel("Test accuracy [\\%]")
    # plt.legend(["Before", "After"])

    fig.savefig(fname, bbox_inches='tight')

plot(lib.get_runs(["scan"]), lib.get_runs(["scan_baseline"], filters={"config.task":"scan"}), "out/scan_removed_vs_baseline_lstm.pdf")
plot(lib.get_runs(["trafo_scan"]), lib.get_runs(["scan_baseline"], filters={"config.task":"trafo_scan"}), "out/scan_removed_vs_baseline_transformer.pdf")
