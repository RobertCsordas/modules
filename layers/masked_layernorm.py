import torch
from typing import Tuple, Union
from .batch_ops import batch_const_mul, batch_bias_add
from .masked_module import MaskedModule


class LayerNorm(MaskedModule):
    def __init__(self, normalized_shape: Union[int, Tuple[int]], eps=1e-5):
        super().__init__()

        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)

        self.gamma = torch.nn.Parameter(torch.ones(*normalized_shape))
        self.beta = torch.nn.Parameter(torch.zeros(*normalized_shape))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return batch_bias_add(batch_const_mul((x-mean) / (std + self.eps), self.gamma), self.beta)
