#!/usr/bin/env python3

import lib
from lib import StatTracker
from lib.common import group
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.pyplot as plt

runs = lib.get_runs(["addmul_ff_alpha_analysis"])
runs = group(runs, ["mask_loss_weight"])

sharing_stats = {}
accuracy_stats = {}

for grp, runs in runs.items():
    print("----------------------------------- ", grp)
    for run in runs:
        print("RUN ID", run.id)
        tsum = 0
        ssum = 0
        # print(stats)
        for k, v in run.summary.items():
            kparts = k.split("/")
            if kparts[-1] != "n_1" or "/all/" in k or not k.startswith("mask_stat/"):
                continue

            print(k, v)

            shared = run.summary["/".join(kparts[:-1] + ["shared_1"])]

            print("SHARED", shared, v * shared, v)
            tsum += v
            ssum += v * shared

        stat_name = [run.config["layer_sizes"], run.config["mask_loss_weight"]]

        if stat_name[0] not in sharing_stats:
            sharing_stats[stat_name[0]] = {}
            accuracy_stats[stat_name[0]] = {}

        if stat_name[1] not in sharing_stats[stat_name[0]]:
            sharing_stats[stat_name[0]][stat_name[1]] = StatTracker()
            accuracy_stats[stat_name[0]][stat_name[1]] = StatTracker()

        accuracy_stats[stat_name[0]][stat_name[1]].add(run.summary["analyzer/baseline/validation/iid/accuracy"]*100)
        sharing_stats[stat_name[0]][stat_name[1]].add(ssum / tsum * 100)


def plot(accuracy_stats, sharing_stats):
    sharing = list(sorted(sharing_stats.keys()))

    def get_col():
        return [sharing_stats[s].get().mean for s in sharing]

    def get_y():
        return [accuracy_stats[s].get().mean for s in sharing]

    def get_x():
        return sharing

    figure = plt.figure(figsize=[5,1.5])
    ax = plt.gca()

    x, y, c = get_x(), get_y(), get_col()
    selected = x.index(1e-4)
    xs, ys, cs = [x[selected]], [y[selected]], [c[selected]]
    del x[selected]
    del c[selected]
    del y[selected]

    sc=plt.scatter(x=x, y=y, s=[50 for _ in c], vmin=25, vmax=55, c=c, cmap='viridis')
    plt.scatter(x=xs, y=ys, s=[130 for _ in cs], vmin=25, vmax=55, c=cs, cmap='viridis', marker='*', edgecolors=(0,0,0), linewidths=0.5)
    plt.axvline(x=1e-4, color="red", zorder=-100)
    plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel("$\\beta = b\\alpha$")
    plt.ylabel("Accuracy [\%]")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(sc, cax)

    return figure

BASE_DIR = 'out/mask_weight_sensitivity'
os.makedirs(BASE_DIR, exist_ok=True)

for set in sharing_stats.keys():
    fig = plot(accuracy_stats[set], sharing_stats[set])
    fig.savefig(os.path.join(BASE_DIR, f"size_{set}.pdf"), bbox_inches='tight')
