import torch
from typing import List, Optional, Dict
import functools
from masked_model import Masks
import framework

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def nearest_scale(images: torch.Tensor, scale: int) -> torch.Tensor:
    images = images.unsqueeze(-2).unsqueeze(-1)
    images = images.expand([-1] * (images.ndim - 4) + [-1, scale, -1, scale]).contiguous()
    return images.view(*images.shape[:-4], images.shape[-4] * scale, images.shape[-2] * scale)


def gen_palette(n_channels: int) -> torch.Tensor:
    if n_channels <= 2:
        palette = ((0,0,1), (1,0,0), (0,1,0))
    elif n_channels <= 9:
        palette = plt.get_cmap("Set1").colors
    else:
        palette = plt.get_cmap("tab20").colors

    assert len(palette) >= n_channels

    palette = ((0, 0, 0),) + palette[1:(n_channels + 1)] + (palette[0],)
    return torch.tensor(palette, dtype=torch.float32)


def to_valid_image_format(t: torch.Tensor) -> torch.Tensor:
    if t.ndim == 2:
        t = t.unsqueeze(1)
    elif t.ndim == 5:
        # Conv filter
        t = t.flatten(1,-2)

    assert t.ndim == 3 and t.shape[2] == 3
    return t.permute([2,1,0])


def draw_mask(mask_list: List[torch.Tensor], n_channels: Optional[int] = None,
              presence_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    palette = gen_palette(n_channels or len(mask_list)).to(mask_list[0].device)

    n_elems = functools.reduce(torch.add, [m.long() for m in mask_list])

    res = sum([palette[((n_elems == 1) & mask).long() * (i + 1)] for i, mask in enumerate(mask_list)])
    res += palette[-1] * (n_elems > 1).float().unsqueeze(-1)

    if presence_mask is not None:
        res += ((n_elems == 0) & presence_mask).float().unsqueeze(-1) * 0.2

    return to_valid_image_format(res)


def draw_mask_histogram(masks: Masks, threshold: float=0) -> framework.visualize.plot.Histogram:
    all_w = torch.cat([m.detach().view(-1) for m in masks.values()], 0)
    all_w = all_w[all_w > threshold]
    return framework.visualize.plot.Histogram(all_w)
