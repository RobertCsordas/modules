#!/usr/bin/env python3

import lib
from lib import StatTracker
from lib.common import group
import os

import matplotlib.pyplot as plt

runs = lib.get_runs(["tuple_rnn", "tuple_feedforward_big"])

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

    fig = plt.figure(figsize=[4.5,1.4])

    keys = list(sorted({k.split("/")[1] for k in stats.keys()}))
    keys = [k for k in keys if k!="all"]
    print(keys)
    if keys[0].startswith("mask_lstm_cells"):
        for i in range(1, len(keys), 2):
            keys[i], keys[i-1] = keys[i-1], keys[i]


    # print([friendly_name(k) for k in keys])
    names = [friendly_name(k) for k in keys]
    legend = ["Pair 1", "Pair 2"]
    for it, task in enumerate(legend):
        d = [stats[f"mask_stat/{k}/shared_{it+1}"].get() for k in keys]
        means = [s.mean*100 for s in d]
        stds = [s.std*100 for s in d]

        print(means, stds)
        plt.bar([2.25 * x + it for x in range(len(names))], means,
                yerr=stds, align='center')

    plt.xticks([2.25 * x + 0.5 for x in range(len(names))], names)
    plt.ylabel("Shared weights [\%]")
    plt.ylim(0,25)
    plt.legend(legend)

    f = f"out/tuple_sharing/{grp}.pdf"
    os.makedirs(os.path.dirname(f), exist_ok=True)
    fig.savefig(f, bbox_inches='tight')
