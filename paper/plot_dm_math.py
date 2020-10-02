import lib
from lib import StatTracker
from collections import OrderedDict
import os

os.makedirs("out", exist_ok=True)

import matplotlib.pyplot as plt

run_table = OrderedDict()

run_table["Add or sub"] = "dm_math_add_or_sub"
run_table["Linear 1D"] = "dm_math_lin1d"
run_table["Differentiate"] = "dm_math_diff"
run_table["Sort"] = "dm_math_sort"
run_table["Poly. Collect"] = "dm_math_polycollect"

runs = lib.get_runs(list(run_table.values()))

trackers = {}
for r in runs:
    accuracies = {}
    if r.config["restore_pretrained"]:
        accuracies["train"] = r.summary["load_validation/hard/accuracy/total"]
    else:
        step = r.config["stop_after"]
        hist = r.scan_history(keys=["validation/hard/accuracy/total"], min_step=step-1, max_step=step+1)
        accuracies["train"] = None
        for h in hist:
            assert accuracies["train"] is None
            accuracies["train"] = h["validation/hard/accuracy/total"]

    accuracies["control"] = r.summary["analysis_results/verify/hard/accuracy/total"]
    accuracies["hard"] = r.summary["analysis_results/hard/hard/accuracy/total"]

    if r.sweep.id not in trackers:
        trackers[r.sweep.id] = {k: StatTracker() for k in accuracies.keys()}

    for k, v in accuracies.items():
        trackers[r.sweep.id][k].add(v*100)


trackers = {k: trackers[lib.source.sweep_table[n]] for k, n in run_table.items()}

cols = ["train", "control", "hard"]
p = {n: [trackers[k][n].get() for k in run_table.keys()] for n in cols}

fig = plt.figure(figsize=[4,0.9])
for i, n in enumerate(cols):
    plt.bar([(len(cols)+1)*x+i for x in range(len(run_table))], [d.mean for d in p[n]], yerr=[d.std for d in p[n]],
            align='center')


plt.xticks([(len(cols)+1)*x+(len(cols)-1)/2 for x in range(len(run_table))], list(run_table.keys()))
plt.ylabel("Accuracy [\\%]")

for tick in fig.axes[0].xaxis.get_major_ticks()[1::2]:
    tick.set_pad(15)

fig.axes[0].yaxis.set_label_coords(-0.115, 0.415)
fig.savefig("out/dm_math.pdf", bbox_inches='tight', pad_inches = 0.01)
# plt.show()