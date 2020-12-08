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

runs = lib.get_runs(["cifar10"])

BASE_DIR = "out/cifar10_confusion/"
shutil.rmtree(BASE_DIR, ignore_errors=True)

def draw(runs, name):
    VER_DIR = f"{BASE_DIR}/{name}/download/"
    os.makedirs(VER_DIR, exist_ok=True)


    def draw_confusion(means: np.ndarray, std: np.ndarray):
        print("MEAN", means)
        figure = plt.figure(figsize=[7,3])#means.shape)

        ax = plt.gca()
        #, vmin = -65, vmax = 65
        im = plt.imshow(means, interpolation='nearest', cmap=plt.cm.viridis, aspect='auto')
        x_marks = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        assert len(x_marks) == means.shape[1]

        y_marks = x_marks
        assert len(y_marks) == means.shape[0]

        plt.xticks(np.arange(means.shape[1]), x_marks, rotation=45)
        # plt.xticks(np.arange(means.shape[1]), x_marks)
        plt.yticks(np.arange(means.shape[0]), y_marks)
        # for tick in figure.axes[0].xaxis.get_major_ticks()[1::2]:
        #     tick.set_pad(15)

        # Use white text if squares are dark; otherwise black.
        threshold = (means.max() + means.min()) / 2.
        print("THRESHOLD", threshold)

        # rmap = np.around(means, decimals=0)
        rmap = np.round(means).astype(np.int)
        std = np.round(std).astype(np.int)
        for i, j in itertools.product(range(means.shape[0]), range(means.shape[1])):
            color = "white" if means[i, j] < threshold else "black"
            plt.text(j, i, f"${rmap[i, j]}\\pm{std[i,j]}$", ha="center", va="center", color=color)

        plt.ylabel("True label", labelpad=-10)
        plt.xlabel("Predicted label", labelpad=-10)
        # plt.xlabel("Predicted label", labelpad=10)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.1)
        plt.colorbar(im, cax)

        # plt.tight_layout()
        return figure

    def create_trackers(runs):
        trackers = {}

        for i_run, run in enumerate(runs):
            for f in run.files(per_page=10000):
                if not f.name.startswith("export") or "/confusion" not in f.name:
                    continue

                if f.name not in trackers:
                    trackers[f.name] = StatTracker()

                full_name = os.path.join(VER_DIR, f.name)
                print(f"Downloading {full_name}")
                f.download(root=VER_DIR, replace=True)

                data = torch.load(full_name)
                data = data.astype(np.float32)
                if "confusion_difference" not in f.name:
                    data = data / np.sum(data, axis=1, keepdims=True)
                data = data * 100
                trackers[f.name].add(data)
                # break
            #
            # if i_run >= 2:
            #     break

        return trackers


    trackers = create_trackers(runs)

    for k, v in trackers.items():
        s = v.get()
        figure = draw_confusion(s.mean, s.std)
        prefix = f"out/cifar10_confusion/{name}/"
        dir = os.path.join(prefix, os.path.dirname(k))
        os.makedirs(dir, exist_ok=True)
        figure.savefig(f"{prefix}/{k}.pdf", bbox_inches='tight', pad_inches = 0.01)
        plt.close()

draw(lib.get_runs(["cifar10_no_dropout"]), "no_dropout")
draw(lib.get_runs(["cifar10"]), "with_dropout")
draw(lib.get_runs(["cifar10_resnet"]), "resnet")
