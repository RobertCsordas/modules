#!/usr/bin/env python3
import wandb
import os
from typing import List, Dict
import lib
from lib import StatTracker
import torch
import shutil
import matplotlib.pyplot as plt
import numpy as np
import itertools
from mpl_toolkits.axes_grid1 import make_axes_locatable
from lib.common import group
from collections import OrderedDict

TEST = False

runs = lib.get_runs(["cifar10"])

BASE_DIR = "out/plot_cifar10_removed"

if not TEST:
    shutil.rmtree(BASE_DIR, ignore_errors=True)

if not os.path.isdir(BASE_DIR):
    os.makedirs(BASE_DIR, exist_ok=True)

    for ri, r in enumerate(runs):
        print(ri)
        for f in r.files():
            if not f.name.startswith("export/stage_final_masks/"):
                continue

            prefix = f"{BASE_DIR}/masks/{ri}/"
            os.makedirs(os.path.dirname(prefix+f.name), exist_ok=True)
            f.download(root=prefix, replace=True)

def load_masks(dir) -> Dict[str, np.ndarray]:
    return {os.path.splitext(f)[0]: torch.load(f"{dir}/{f}")>0 for f in os.listdir(dir)}

def plot_removed_per_class():
    def calc_sharing(dir):
        dir = f"{dir}/export/stage_final_masks"
        ref_masks = load_masks(dir+"/stage_0")
        n_ref = sum([r.astype(np.float32).sum() for r in ref_masks.values()])

        res = {}
        for stage in range(1,11):
            m = load_masks(dir + f"/stage_{stage}")
            m = {k: v & ref_masks[k] for k, v in m.items()}
            n = sum([a.astype(np.float32).sum() for a in m.values()])

            res[stage] = (n_ref-n)/n_ref * 100

        return res

    trackers = [StatTracker() for i in range(10)]
    for d in os.listdir(f"{BASE_DIR}/masks/"):
        sharing = calc_sharing(f"{BASE_DIR}/masks/{d}/")
        for k, v in sharing.items():
            trackers[k-1].add(v)

    stats = [t.get() for t in trackers]
    labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    fig = plt.figure(figsize=[5.5,1.5])
    plt.bar([x for x in range(len(labels))], [s.mean for s in stats], yerr=[s.std for s in stats], align='center')
    plt.xticks([x for x in range(len(labels))], labels, rotation=45)
    plt.ylabel("Weights removed [\%]")
    fig.savefig(f"{BASE_DIR}/cifar10_removed_per_class.pdf", bbox_inches='tight')
    # plt.show()

def plot_removed_per_layer():
    trackers = {}

    def calc_sharing(dir):
        dir = f"{dir}/export/stage_final_masks"
        ref_masks = load_masks(dir+"/stage_0")
        n_ref = {k: r.astype(np.float32).sum() for k, r in ref_masks.items()}

        sum_total = {k: 0 for k in n_ref.keys()}
        for stage in range(1,11):
            m = load_masks(dir + f"/stage_{stage}")
            m = {k: v & ref_masks[k] for k, v in m.items()}
            sum_total = {k: sum_total[k] + v.astype(np.float32).sum() for k, v in m.items()}

        return {k: 100*(r-sum_total[k]/10)/r for k, r in n_ref.items()}

    for d in os.listdir(f"{BASE_DIR}/masks/"):
        sharing = calc_sharing(f"{BASE_DIR}/masks/{d}/")
        for k, v in sharing.items():
            if k not in trackers:
                trackers[k] = StatTracker()
            trackers[k].add(v)

    trackers = {k: v for k, v in trackers.items() if k.endswith("_weight")}

    fields  = OrderedDict()
    fields["conv 1"] = "features_0_weight"
    fields["conv 2"] = "features_3_weight"
    fields["conv 3"] = "features_6_weight"
    fields["conv 4"] = "features_10_weight"
    fields["output"] = "out_layer_weight"
    stats = [trackers[k].get() for k in fields.values()]
    fig = plt.figure(figsize=[5.5,1.5])
    plt.bar([x for x in range(len(fields))], [s.mean for s in stats], yerr=[s.std for s in stats], align='center')
    plt.xticks([x for x in range(len(fields))], fields.keys(), rotation=0)
    plt.ylabel("Weights removed [\%]")
    fig.savefig(f"{BASE_DIR}/cifar10_removed_per_layer.pdf", bbox_inches='tight')
    plt.show()

plot_removed_per_class()
plot_removed_per_layer()
# print(trackers)