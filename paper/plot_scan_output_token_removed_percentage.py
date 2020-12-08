#!/usr/bin/env python3
import os
import torch
import lib
from tqdm import tqdm

import matplotlib.pyplot as plt

import wandb
api = wandb.Api()


BASE_DIR = "out/scan_jump_output"
os.makedirs(BASE_DIR, exist_ok=True)

runs = lib.get_runs(["scan"])
for i, r in tqdm(enumerate(runs)):
    dest = os.path.join(BASE_DIR, "weights", str(i))
    os.makedirs(dest, exist_ok=True)
    r.file("export/stage_final_masks/stage_1/decoder_output_projection_weight.pth").download(root=dest, replace=True)
    r.file("export/stage_final_masks/stage_0/decoder_output_projection_weight.pth").download(root=dest, replace=True)

avg = lib.StatTracker()
for i in range(len(runs)):
    dest = os.path.join(BASE_DIR, "weights", str(i))
    m_base = torch.load(f"{BASE_DIR}/weights/{i}/export/stage_final_masks/stage_0/"
                        f"decoder_output_projection_weight.pth") > 0
    m_jump = torch.load(f"{BASE_DIR}/weights/{i}/export/stage_final_masks/stage_1/"
                        f"decoder_output_projection_weight.pth") > 0

    removed = m_base & (~m_jump)
    avg.add(removed.sum(-1) / m_base.sum(-1) * 100)

data = avg.get()

fig = plt.figure(figsize=[5,1.05])

percent_removed = (removed.sum(-1) / m_base.sum(-1) * 100).tolist()
tokens = ["I_TURN_LEFT", "I_TURN_RIGHT", "I_JUMP", "I_WALK", "I_RUN", "I_LOOK", "EOS"]

plt.bar([x for x in range(len(tokens))], data.mean, yerr=data.std, align='center')
plt.xticks([x for x in range(len(tokens))], [t.replace("_","\_") for i, t in enumerate(tokens)])
plt.ylabel("Removed [\%]")

for tick in fig.axes[0].xaxis.get_major_ticks()[1::2]:
    tick.set_pad(15)

fig.savefig(f"{BASE_DIR}/scan_out_removed.pdf", bbox_inches='tight', pad_inches = 0.01)


