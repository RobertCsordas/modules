#!/usr/bin/env python3
import wandb
import os
from typing import List, Dict
import lib
from lib.common import group
import torch
import shutil
import matplotlib.pyplot as plt
import numpy as np
import itertools
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams['text.usetex'] = True #Let TeX do the typsetting
plt.rcParams['text.latex.preamble'] = [r'\usepackage{sansmath}', r'\sansmath'] #Force sans-serif math mode (for axes labels)
plt.rcParams['font.family'] = 'sans-serif' # ... for regular text
plt.rcParams['font.sans-serif'] = 'Helvetica, Avant Garde, Computer Modern Sans serif' # Choose a nice font here

TEST=False

api = wandb.Api()

runs = lib.get_runs(["addmul_feedforward_big", "addmul_rnn"])

BASE_DIR = "out/addmul_confusion_plot/download"
shutil.rmtree(BASE_DIR, ignore_errors=True)
os.makedirs(BASE_DIR, exist_ok=True)


runs = group(runs, ['layer_sizes', "task"])
print(runs.keys())


def draw_confusion(means: np.ndarray, std: np.ndarray):
    print("MEAN", means)
    figure = plt.figure(figsize=[2.5,0.5])#means.shape)

    ax = plt.gca()
    im = plt.imshow(means, interpolation='nearest', cmap=plt.cm.viridis, aspect='auto', vmin=0, vmax=100)
    x_marks = ["$+$", "$*$", "none"]
    assert len(x_marks) == means.shape[1]

    y_marks = ["$+$", "$*$"]
    assert len(y_marks) == means.shape[0]

    plt.xticks(np.arange(means.shape[1]), x_marks)
    plt.yticks(np.arange(means.shape[0]), y_marks)

    # Use white text if squares are dark; otherwise black.
    threshold = (means.min() + means.max()) / 2

    rmap = (means+0.5).astype(np.int32)
    std = (std+0.5).astype(np.int32)
    for i, j in itertools.product(range(means.shape[0]), range(means.shape[1])):
        color = "white" if means[i, j] < threshold else "black"
        plt.text(j, i, f"${rmap[i, j]}\\pm{std[i,j]}$", ha="center", va="center", color=color)

    plt.ylabel("True")
    plt.xlabel("Predicted")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax)

    # plt.tight_layout()
    return figure

def create_trackers(runs):
    trackers = {}

    for i_run, run in enumerate(runs):
        for f in run.files():
            if not f.name.startswith("export") or "confusion" not in f.name:
                continue

            print("DOWNLOADING", f.name)

            if f.name not in trackers:
                trackers[f.name] = lib.StatTracker()

            full_name = os.path.join(BASE_DIR, f.name)
            f.download(root=BASE_DIR, replace=True)

            data = torch.load(full_name)
            data = data.transpose(1, 0).astype(np.float32)
            data = data / np.sum(data, axis=1, keepdims=True) * 100
            trackers[f.name].add(data)
            if TEST:
                break

        if TEST and i_run >= 2:
            break

    return trackers

trackers = {k: create_trackers(v) for k, v in runs.items()}

for grp, runs in trackers.items():
    for k, v in runs.items():
        s = v.get()
        figure = draw_confusion(s.mean, s.std)
        prefix = f"out/addmul_confusion_plot/{grp}"
        dir = os.path.join(prefix, os.path.dirname(k))
        os.makedirs(dir, exist_ok=True)
        figure.savefig(f"{prefix}/{k}.pdf", bbox_inches='tight')
        plt.close()
# runs = filter(lambda x: x.config["task"].startswith("addmul"), runs)
