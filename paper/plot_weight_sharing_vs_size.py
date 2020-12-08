#!/usr/bin/env python3

import lib
from lib import StatTracker
from lib.common import group
import os

import matplotlib.pyplot as plt

runs = lib.get_runs(["addmul_feedforward", "addmul_feedforward_big", "addmul_feedforward_huge"])


os.makedirs("out", exist_ok=True)


runs = group(runs, ["layer_sizes"])
print(runs)
all_stats = {}

for grp, runs in runs.items():
    print("----------------------------------- ", grp)
    for run in runs:
        tsum = 0
        ssum = 0
        # print(stats)
        for k, v in run.summary.items():
            kparts = k.split("/")
            if kparts[-1] != "n_1" or "/all/" in k or not k.startswith("mask_stat/"):
                continue

            print(k, v)

            shared = run.summary["/".join(kparts[:-1]+["shared_1"])]

            print("SHARED", shared, v*shared, v)
            tsum += v
            ssum += v*shared


        if grp not in all_stats:
            all_stats[grp] = StatTracker()

        all_stats[grp].add(ssum/tsum)

order = ["layer_sizes_400,400,200", "layer_sizes_800,800,800,800", "layer_sizes_2000,2000,2000,2000", "layer_sizes_4000,4000,4000,4000"]
stats = [all_stats[o].get() for o in order]

fig = plt.figure(figsize=[6,2])
plt.bar([x for x in range(len(order))], [s.mean*100 for s in stats], yerr=[s.std*100 for s in stats], align='center')
plt.xticks([x for x in range(len(order))], ["small", "medium", "big", "huge"])
plt.ylabel("Total shared [\\%]")
fig.savefig(f"out/sharing_vs_size.pdf", bbox_inches='tight')
