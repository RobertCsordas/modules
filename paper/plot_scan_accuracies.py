#!/usr/bin/env python3

import lib
from lib import StatTracker
from lib.common import group, calc_stat
import wandb
import collections
import os

import matplotlib.pyplot as plt

runs = lib.get_runs(["scan"])
groups = {"all": runs}

stats = calc_stat(groups, lambda k: k.startswith("analysis_results/") and "/accuracy/" in k and "/train/" not in k)["all"]
for k, s in stats.items():
    print(k)

plots = collections.OrderedDict()
plots["Turn Left"] = "analysis_results/turn_left/turn_left/accuracy/total"
plots["Jump"] = "analysis_results/jump/jump/accuracy/total"
plots["Length"] = "analysis_results/length/length/accuracy/total"

refs = {
    "Turn Left": "analysis_results/simple/turn_left/accuracy/total",
    "Jump": "analysis_results/simple/jump/accuracy/total",
    "Length": "analysis_results/simple/length/accuracy/total",
}

names = list(plots.keys())
print(names)

means = {k: stats[v].get().mean for k, v in plots.items()}
std = {k: stats[v].get().std for k, v in plots.items()}

ref_means = {k: stats[v].get().mean for k, v in refs.items()}
ref_std = {k: stats[v].get().std for k, v in refs.items()}


fig = plt.figure(figsize=[3,1.5])

plt.bar([2.25*x for x in range(len(names))], [ref_means[n]*100 for n in names], yerr=[ref_std[n]*100 for n in names], align='center')
plt.bar([2.25*x+1 for x in range(len(names))], [means[n]*100 for n in names], yerr=[std[n]*100 for n in names], align='center')
plt.xticks([2.25*x+0.5 for x in range(len(names))], names)
plt.ylabel("Test accuracy [\\%]")
# plt.legend(["Before", "After"])

os.makedirs("out", exist_ok=True)
fig.savefig("out/scan_accuracies.pdf", bbox_inches='tight')