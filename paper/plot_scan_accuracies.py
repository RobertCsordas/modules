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
    "Turn Left": "analysis_results/simple/turn_left/accuracy/total",
    "Jump": "analysis_results/simple/jump/accuracy/total",
    "Length": "analysis_results/simple/length/accuracy/total",
}

names = list(plots.keys())


def get_mean_err(runs):
    groups = {"all": runs}

    stats = calc_stat(groups, lambda k: k.startswith("analysis_results/") and "/accuracy/" in k and "/train/" not in k)["all"]
    for k, s in stats.items():
        print(k)

    means = {k: stats[v].get().mean for k, v in plots.items()}
    std = {k: stats[v].get().std for k, v in plots.items()}

    ref_means = {k: stats[v].get().mean for k, v in refs.items()}
    ref_std = {k: stats[v].get().std for k, v in refs.items()}

    return [ref_means[n]*100 for n in names], [ref_std[n]*100 for n in names], \
           [means[n]*100 for n in names], [std[n]*100 for n in names]

def printres(name, vals, errs):
    print(name+": "+(" ".join([f"{names[i]}: {v:.1f} ({e:.2f})" for i, (v, e) in enumerate(zip(vals, errs))])))

fig = plt.figure(figsize=[4,1])

ref_m, ref_err, res_m, res_err = get_mean_err(lib.get_runs(["scan"]))
t_ref_m, t_ref_err, t_res_m, t_res_err = get_mean_err(lib.get_runs(["trafo_scan"]))

printres('LSTM iid', ref_m, ref_err)
printres('LSTM splits', res_m, res_err)
printres('Trafo iid', t_ref_m, t_ref_err)
printres('Trafo splits', t_res_m, t_res_err)
print("LSTM - trafo:", " ".join(f"{names[i]}: {l-t:.2f}" for i, (l, t) in enumerate(zip(res_m, t_res_m))))

plt.bar([5*x for x in range(len(names))], ref_m, yerr=ref_err, align='center')
plt.bar([5*x+1 for x in range(len(names))], t_ref_m, yerr=t_ref_err, align='center')

plt.bar([5*x+2+0.25 for x in range(len(names))], res_m, yerr=res_err, align='center')
plt.bar([5*x+3+0.25 for x in range(len(names))], t_res_m, yerr=t_res_err, align='center')

plt.xticks([5*x+1.625 for x in range(len(names))], names)
plt.ylabel("Accuracy [\\%]")
# plt.legend(["Before", "After"])

fig.savefig("out/scan_accuracies.pdf", bbox_inches='tight', pad_inches = 0.01)

