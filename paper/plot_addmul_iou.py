#!/usr/bin/env python3

import lib
from lib.common import group
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import torch
import numpy as np
from typing import Dict, Union

runs = lib.get_runs(["addmul_feedforward_big", "addmul_rnn"])

runs = group(runs, ["layer_sizes", "task"])

all_stats = {}

download_dir = "out/addmul_iou/weights"

@dataclass
class Similarity:
    iou: Union[float, lib.StatTracker, lib.Stat]
    subsetness: Union[float, lib.StatTracker, lib.Stat]

def calc_stats(run: str) -> Dict[str, Similarity]:
    base_dir = os.path.join(download_dir, run, "export/stage_final_masks")
    dir1=f"{base_dir}/stage_1/"
    dir2=f"{base_dir}/stage_2/"

    res = {}

    for f in os.listdir(dir1):
        assert f.endswith(".pth")
        m1 = (torch.load(os.path.join(dir1, f)) > 0)
        m2 = (torch.load(os.path.join(dir2, f)) > 0)

        n_min = min(m1.astype(np.int64).sum(), m2.astype(np.int64).sum())

        intersect = (m1 & m2).astype(np.int64).sum()
        union = (m1 | m2).astype(np.int64).sum()

        res[f[:-4]] = Similarity(intersect/union, intersect/n_min)

    return res

for grp, rn in runs.items():
    if grp not in all_stats:
        all_stats[grp] = {}

    stats = all_stats[grp]

    for run in rn:
        for f in run.files(per_page=10000):
            if not f.name.startswith("export") or "/stage_final_masks" not in f.name:
                continue

            fname = os.path.join(download_dir, run.id, f.name)
            if not os.path.isfile(fname):
                print(fname)
                target_dir = os.path.dirname(fname)

                os.makedirs(target_dir, exist_ok=True)

                print(f"Run {run.id}: downloading {fname}...")
                f.download(root=os.path.join(download_dir, run.id), replace=True)

        for name, val in calc_stats(run.id).items():
            if name not in stats:
                stats[name] = Similarity(lib.StatTracker(), lib.StatTracker())

            stats[name].iou.add(val.iou)
            stats[name].subsetness.add(val.subsetness)

    for v in stats.values():
        v.iou = v.iou.get()
        v.subsetness = v.subsetness.get()
#
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

    keys = list(sorted(stats.keys()))
    print(keys)
    if keys[0].startswith("mask_lstm_cells"):
        for i in range(1, len(keys), 2):
            keys[i], keys[i-1] = keys[i-1], keys[i]

    names = [friendly_name(k) for k in keys]
    legend = ["IoU", "IoMin"]

    plt.bar([2.25 * x for x in range(len(names))], [stats[n].iou.mean for n in keys], yerr=[stats[n].iou.std for n in keys], align='center')
    plt.bar([2.25 * x+1 for x in range(len(names))], [stats[n].subsetness.mean for n in keys], yerr=[stats[n].subsetness.std for n in keys], align='center')

    plt.xticks([2.25 * x + 0.5 for x in range(len(names))], names)
    plt.ylabel("Proportion")
    plt.ylim(0,1)
    if grp.endswith("task_addmul"):
        plt.legend(legend, ncol=2, bbox_to_anchor=(0.39,0.6), columnspacing=1)
    else:
        plt.legend(legend, ncol=2, loc="upper center", )
    # fig.axes[0].yaxis.set_label_coords(-0.115, 0.415)

    f = f"out/addmul_iou/{grp}.pdf"
    os.makedirs(os.path.dirname(f), exist_ok=True)
    fig.savefig(f, bbox_inches='tight', pad_inches = 0.01)
