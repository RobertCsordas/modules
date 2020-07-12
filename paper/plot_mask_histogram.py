#!/usr/bin/env python3
import os
import torch
import numpy as np
import lib
from lib import StatTracker
from tqdm import tqdm
from pathlib import Path
from typing import Dict
from lib.common import group
from collections import OrderedDict

import matplotlib.pyplot as plt

import wandb
api = wandb.Api()

runs = lib.get_runs(["addmul_rnn", "addmul_feedforward_big", "tuple_rnn", "tuple_feedforward_big"])

BASE_DIR = "out/mask_histogram"
WEIGHTS_DIR = os.path.join(BASE_DIR, f"weights")
os.makedirs(BASE_DIR, exist_ok=True)

for i, r in enumerate(runs):
    print(f"Downloading run {i}, {r.name}, {r.id}")
    run_dir = os.path.join(WEIGHTS_DIR, r.name, r.id)
    if os.path.isdir(run_dir):
        continue

    for f in tqdm(r.files()):
        if "export/stage_final_masks/stage_0" not in f.name:
            continue

        dl_name = os.path.join(run_dir, f.name)
        os.makedirs(os.path.dirname(dl_name), exist_ok=True)
        f.download(root=run_dir, replace=True)


N_POINTS=500

runs = group(runs, ["task", 'layer_sizes', "tuple.mode"])

trackers: Dict[str, StatTracker] = {}
trackers_all: Dict[str, StatTracker] = {}

def add_tracker(trackers, name, data):
    if name not in trackers:
        trackers[name] = StatTracker()

    hist, _ = np.histogram(data, N_POINTS, [0, 1])
    trackers[name].add(hist)


human_readable_names = OrderedDict()
human_readable_names['task_addmul_ff/layer_sizes_2000,2000,2000,2000/tuple.mode_together'] = "$+/*$ FNN"
human_readable_names['task_addmul/layer_sizes_800,800,256/tuple.mode_together'] = "$+/*$ LSTM"
human_readable_names['task_tuple_ff/layer_sizes_2000,2000,2000,2000/tuple.mode_together'] = "$+/+$ FNN"
human_readable_names['task_tuple/layer_sizes_800,800,256/tuple.mode_together'] = "$+/+$ LSTM"

for n, runs in runs.items():
    for r in runs:
        rundir = os.path.join(WEIGHTS_DIR, r.name, r.id, "export", "stage_final_masks")

        print(run_dir)
        weights = [torch.load(w).reshape(-1) for w in Path().glob(f"{rundir}/stage_0/**/*.pth")]
        weights = np.concatenate(weights)
        m = 1 / (1 + np.exp(-weights))

        add_tracker(trackers, n, m[m > (0.1 / N_POINTS)])
        add_tracker(trackers_all, n, m)

def draw(trackers, fname, log=False, **kwargs):
    fig = plt.figure(figsize=[5,2])
    N_SKIP = 0
    t = (lambda x: np.log(x)) if log else (lambda x: x)
    for name in human_readable_names.keys():
        stat = trackers[name].get()
        plt.plot([i/N_POINTS for i in range(N_SKIP, N_POINTS)], t(stat.mean[N_SKIP:]))
        plt.fill_between([i / N_POINTS for i in range(N_SKIP, N_POINTS)], t(stat.mean - stat.std)[N_SKIP:], t(stat.mean + stat.std)[N_SKIP:], alpha=0.3)

    plt.legend(human_readable_names.values(), loc="upper center")
    plt.ylim(bottom=0, **kwargs)
    plt.xlabel('mask value')
    plt.ylabel('log density' if log else 'density')

    fig.savefig(os.path.join(BASE_DIR,fname), bbox_inches='tight')

draw(trackers, "mask_histogram.pdf", top=1300)
draw(trackers_all, "mask_histogram_all.pdf", log=True)