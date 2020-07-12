import torch
from typing import Dict, Any
from dataclasses import dataclass


class Result:
    outputs: torch.Tensor
    loss: torch.Tensor

    def plot(self) -> Dict[str, Any]:
        return {}


@dataclass
class RecurrentResult(Result):
    outputs: torch.Tensor
    loss: torch.Tensor

    @property
    def batch_size(self) -> int:
        return self.outputs.shape[1]


@dataclass
class FeedforwardResult(Result):
    outputs: torch.Tensor
    loss: torch.Tensor

    @property
    def batch_size(self) -> int:
        return self.outputs.shape[0]
