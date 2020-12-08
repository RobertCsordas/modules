#!/usr/bin/env python3
import os
import lib
from lib import StatTracker
import torch
import shutil
import matplotlib.pyplot as plt
import numpy as np
import itertools
from mpl_toolkits.axes_grid1 import make_axes_locatable
from lib.common import group
from _collections import OrderedDict

TEST = False

BASE_DIR = "out/cifar10_drop_per_class/"
cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

if not TEST:
    shutil.rmtree(BASE_DIR, ignore_errors=True)

def download(dir, runs):
    for i_run, run in enumerate(runs):
        t_dir = os.path.join(BASE_DIR, dir, run.id)
        if os.path.isdir(t_dir):
            continue

        for f in run.files(per_page=10000):
            if not f.name.startswith("export") or "/confusion" not in f.name:
                continue

            full_name = os.path.join(t_dir, f.name)
            print("Downloading", full_name)
            if not os.path.isfile(full_name):
                f.download(root=t_dir)

def get_relative_drop(dir, runs):
    trackers = {k: StatTracker() for k in cifar10_classes}

    for run in runs:
        t_dir = os.path.join(BASE_DIR, dir, run.id)
        ref = torch.load(os.path.join(t_dir, "export/class_removal/confusion_reference.pth")).astype(np.float32)
        ref = ref / np.sum(ref, axis=1, keepdims=True)
        ref = np.diag(ref)

        for i, cls in enumerate(cifar10_classes):
            d = torch.load(os.path.join(t_dir,"export/class_removal/confusion_difference/",cls+".pth"))
            d = np.diag(d)

            rel_drop = d / ref * 100
            trackers[cls].add(-rel_drop[i])

    return [trackers[k].get() for k in cifar10_classes]


def get_worst(dir, runs):
    trackers = {k: StatTracker() for k in cifar10_classes}

    for run in runs:
        t_dir = os.path.join(BASE_DIR, dir, run.id)
        ref = torch.load(os.path.join(t_dir, "export/class_removal/confusion_reference.pth")).astype(np.float32)
        ref = ref / np.sum(ref, axis=1, keepdims=True)
        ref = np.diag(ref)

        for i, cls in enumerate(cifar10_classes):
            d = torch.load(os.path.join(t_dir,"export/class_removal/confusion_difference/",cls+".pth"))
            d = np.diag(d)

            rel_drop = d / ref * 100
            target_drop = rel_drop[i]
            rel_drop[i] = 0

            trackers[cls].add(min(np.min(rel_drop),0)/target_drop * 100)

    return [trackers[k].get() for k in cifar10_classes]

runs = OrderedDict()
runs["Simple"] = "cifar10"
runs["No dropout"] = "cifar10_no_dropout"
runs["Resnet 110"] = "cifar10_resnet"

rel_drops = {}
worsts = {}

for k, v in runs.items():
    r = lib.get_runs([v])
    print("Downloading", k)
    download(v, r)
    rel_drops[k] = get_relative_drop(v, r)
    worsts[k] = get_worst(v, r)

def draw_stats(stats, shape):
    fig = plt.figure(figsize=shape)
    for i, k in enumerate(runs.keys()):
        stat = stats[k]
        plt.bar([x * (len(runs) + 1) + i for x in range(len(stat))], [s.mean for s in stat],
                yerr=[s.std for s in stat], align='center')

    plt.xticks([(len(runs) + 1) * x + (len(runs) - 1) / 2 for x in range(len(cifar10_classes))], cifar10_classes)
    for tick in fig.axes[0].xaxis.get_major_ticks()[1::2]:
        tick.set_pad(15)

    # plt.legend(runs.keys())
    # plt.show()
    return fig

fig = draw_stats(worsts, [5.5, 1.5])
plt.ylabel("Largest other/target [\%]")
plt.ylim((0,10))
fig.axes[0].yaxis.set_label_coords(-0.115, 0.4)
fig.savefig(os.path.join(BASE_DIR, "cnn_worst_drops.pdf"), bbox_inches='tight', pad_inches = 0.01)

fig = draw_stats(rel_drops, [5.5, 1.5])
plt.ylabel("Relative drop [\%]")
fig.savefig(os.path.join(BASE_DIR, "cnn_relative_drops.pdf"), bbox_inches='tight', pad_inches = 0.01)
