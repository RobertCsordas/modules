#!/usr/bin/env python3
import os
import torch
import numpy as np
import lib

import matplotlib.pyplot as plt

import wandb
api = wandb.Api()

run = api.run(lib.get_config()["scan_jump_output_mask_run"])

BASE_DIR = "out/scan_jump_output"
os.makedirs(BASE_DIR, exist_ok=True)
run.file("export/stage_final_masks/stage_1/decoder_output_projection_weight.pth").download(root=BASE_DIR, replace=True)
run.file("export/stage_final_masks/stage_0/decoder_output_projection_weight.pth").download(root=BASE_DIR, replace=True)

m_base = torch.load(f"{BASE_DIR}/export/stage_final_masks/stage_0/decoder_output_projection_weight.pth") > 0
m_jump = torch.load(f"{BASE_DIR}/export/stage_final_masks/stage_1/decoder_output_projection_weight.pth") > 0

background_color = np.asfarray([1,1,1])
orig_color = np.asfarray([1.0,0.498,0.055])
removed_color = np.asfarray([0x28,0x33,0x4a])/255

def colorize(map, color):
    map = np.expand_dims(map, 2).astype(np.float32)
    color = np.expand_dims(color, (0,1))
    return map * color

background = ~m_base
kept = m_base & m_jump
removed = m_base & (~m_jump)

image = colorize(background, background_color) + colorize(kept, orig_color) + colorize(removed, removed_color)

fig = plt.figure(figsize=[5,1.3])


fig.add_subplot(2,1,1)
plt.imshow(image[:,:100], interpolation='nearest')

fig.add_subplot(2,1,2)
plt.imshow(image[:,100:], interpolation='nearest', extent=[100,199,6.5,-0.5])

plt.xlabel("Input")

fig.text(0.06, 0.5, 'Output', va='center', rotation='vertical')
fig.savefig(f"{BASE_DIR}/scan_jump_mask.pdf", bbox_inches='tight')

