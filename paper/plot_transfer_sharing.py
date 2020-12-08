#!/usr/bin/env python3

import lib
from lib import StatTracker
import matplotlib.pyplot as plt
from collections import OrderedDict
from typing import List
import os


basedir = "out/transfer_sharing"
os.makedirs(basedir, exist_ok=True)

def run(name: str, shape: List[int], y_lim=None, coord=None, lloc="upper center", lncol=2, **kwargs):
    runs = lib.get_runs([name], **kwargs)

    trackers = {}

    for r in runs:
        for k, s in r.summary.items():
            if "/mask_stat/shared/" not in k or (not (k.endswith("_weight") and k.startswith("permuted_mnist/perm_"))):
                continue
            if k not in trackers:
                trackers[k] = StatTracker()

            # trackers[k].add(s*100)
            trackers[k].add(s)


    perm_by_layer = {}

    for perm in range(1,10,2):
        prefix = f"permuted_mnist/perm_{perm}/mask_stat/shared/"
        for k, v in trackers.items():
            if not k.startswith(prefix):
                continue

            layer_name = k.split("/")[-1]
            if layer_name not in perm_by_layer:
                perm_by_layer[layer_name]=[]

            perm_by_layer[layer_name].append(v)


    fig = plt.figure(figsize=shape)
    # ax = fig.add_subplot(111,aspect=0.06)
    n_col = 5
    d = n_col + 1

    names = OrderedDict()
    # names["Layer 1"] = "layers_0_weight"
    names["Layer 2"] = "layers_1_weight"
    names["Layer 3"] = "layers_2_weight"
    names["Layer 4"] = "layers_3_weight"

    for j in range(n_col):
        stats = [perm_by_layer[n][j].get() for n in names.values()]
        plt.bar([i*d + j for i in range(len(names))], [s.mean for s in stats], yerr=[s.std for s in stats], align='center')

    plt.xticks([d * x + n_col/2 - 0.5  for x in range(len(names))], names.keys())
    plt.legend([f"T{c*2+2}" for c in range(n_col)], ncol=lncol,  loc=lloc,  columnspacing=1)
    plt.ylabel("Proportion")
    if y_lim is not None:
        plt.ylim(*y_lim)

    # if coord is not None:
    #     fig.axes[0].yaxis.set_label_coords(*coord)
    fig.savefig(f"{basedir}/{name}.pdf", bbox_inches='tight', pad_inches = 0.01)


run("transfer_sharing", [4,1.2],  y_lim=(0,1), lloc="upper left", lncol=3, coord=(-0.1, 0.4))
run("transfer_sharing_prefer_old", [4,2], y_lim=(0,1), lncol=3, lloc="upper left")
run("transfer_sharing_prefer_old_even_more", [4,2], y_lim=(0,1), lncol=3, lloc="lower right")
