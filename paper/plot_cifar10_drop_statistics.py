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

BASE_DIR = "out/cifar_drop_stat_confusion/"

VER_DIR = f"{BASE_DIR}/cifar10/download/"


#shutil.rmtree(VER_DIR, ignore_errors=True)
os.makedirs(VER_DIR, exist_ok=True)

cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
cifar100_classes = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle',
                    'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar',
                    'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile',
                    'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
                    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster',
                    'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid',
                    'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
                    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew',
                    'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower',
                    'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
                    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']


for i_run, run in enumerate(runs):
    for f in run.files():
        if not f.name.startswith("export") or "/confusion" not in f.name or f.name.endswith("confusion_reference.pth"):
            continue

        print("DOWNLOADING", f.name)

        full_name = os.path.join(VER_DIR, f.name)
        f.download(root=VER_DIR, replace=True)


        data = torch.load(full_name)
        data = data.astype(np.float32)
        if "confusion_difference" not in f.name:
            data = data / np.sum(data, axis=1, keepdims=True)
        data = data * 100

        class_index = cifar10_classes.index(f.name.split("/")[-1].split(".")[0])

        tp = data.diagonal().tolist()
        this_drop = tp[class_index]

        other_drop = min(min(tp[:class_index]+tp[class_index+1:]),0)

        print(this_drop, other_drop, this_drop/other_drop)


