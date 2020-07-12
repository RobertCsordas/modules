import torch
import torch.nn.functional as F
from typing import List
from layers import Linear


class FeedforwardModel(torch.nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int, layer_sizes: List[int]):
        super().__init__()
        self.layers = torch.nn.ModuleList([Linear(in_s, out_s)
                                           for in_s, out_s in zip([n_inputs] + layer_sizes, layer_sizes + [n_outputs])])

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        for i, l in enumerate(self.layers):
            if i!=0:
                input = F.relu(input)
            input = l(input)

        return input

    def reset_parameters(self):
        last_layer = len(self.layers)-1
        for i, l in enumerate(self.layers):
            torch.nn.init.xavier_normal_(l.weight, gain=1 if i == last_layer else torch.nn.init.calculate_gain("relu"))
            l.bias.data.zero_()
