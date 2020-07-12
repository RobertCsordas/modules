import torch
from .masked_module import MaskedModule
import math
from .batch_ops import batch_conv2d


class Conv2d(MaskedModule):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int=0):
        super().__init__()

        self.weight = torch.nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

        self.stride = stride
        self.padding = padding

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        return batch_conv2d(input, self.weight, self.bias, stride=self.stride, padding=self.padding)
