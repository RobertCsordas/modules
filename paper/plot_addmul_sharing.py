#!/usr/bin/env python3

import lib
from lib import StatTracker
from lib.common import group
import os

import matplotlib.pyplot as plt

runs = lib.get_runs(["addmul_feedforward_big", "addmul_rnn"])

runs = group(runs, ["layer_sizes", "task"])

all_stats = {}

for grp, rn in runs.items():
    if grp not in all_stats:
        all_stats[grp] = {}

    stats = all_stats[grp]

    for r in rn:
        for k,v  in r.summary.items():
            if not k.startswith("mask_stat/") or "/shared_" not in k:
                continue

            if k not in stats:
                stats[k] = StatTracker()

            stats[k].add(v)

    if not all_stats[grp]:
        del all_stats[grp]

def friendly_name(name: str) -> str:
    if name.startswith("mask_"):
        name = name[5:]

    if name.endswith("_weight"):
        name = name[:-7]

    name=name.replace("_weight_", "_")
    name=name.replace("_cells_", "_")

    lparts = name.split("_")
    if lparts[0] == "layers" and lparts[1].isdecimal():
        name = f"layer {int(lparts[1])+1}"

    if name in ["output_projection", "layer 5"]:
        name = "output"

    return name.replace("_","\\_")



for grp, stats in all_stats.items():
    print("-------------------- GROUP --------", grp)
    print(stats.keys())

    fig = plt.figure(figsize=[4.5,1.2])

    keys = list(sorted({k.split("/")[1] for k in stats.keys()}))
    keys = [k for k in keys if k!="all"]
    print(keys)
    if keys[0].startswith("mask_lstm_cells"):
        for i in range(1, len(keys), 2):
            keys[i], keys[i-1] = keys[i-1], keys[i]


    names = [friendly_name(k) for k in keys]
    legend = ["$+$", "$*$"]
    for it, task in enumerate(legend):
        d = [stats[f"mask_stat/{k}/shared_{it+1}"].get() for k in keys]
        means = [s.mean*100 for s in d]
        stds = [s.std*100 for s in d]

        print(means, stds)
        plt.bar([2.25 * x + it for x in range(len(names))], means,
                yerr=stds, align='center')

    plt.xticks([2.25 * x + 0.5 for x in range(len(names))], names)
    plt.ylabel("Shared weights [\%]")
    plt.ylim(0,100)
    plt.legend(legend, ncol=2, loc="upper center")
    fig.axes[0].yaxis.set_label_coords(-0.115, 0.415)

    f = f"out/addmul_sharing/{grp}.pdf"
    os.makedirs(os.path.dirname(f), exist_ok=True)
    fig.savefig(f, bbox_inches='tight', pad_inches = 0.01)
