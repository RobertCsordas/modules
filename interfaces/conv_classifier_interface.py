import torch
from torch.nn import functional as F
from typing import Dict
from .result import FeedforwardResult
from .model_interface import ModelInterface


class ConvClassifierInterface(ModelInterface):
    def __init__(self, model: torch.nn.Module):
        self.model = model

    def create_input(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        return data["image"]

    def loss(self, net_out: torch.Tensor, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.cross_entropy(net_out, data["label"].long().flatten())

    def decode_outputs(self, outputs: FeedforwardResult) -> torch.Tensor:
        return outputs.outputs.argmax(-1)

    def __call__(self, data: Dict[str, torch.Tensor]) -> FeedforwardResult:
        input = self.create_input(data)

        res = self.model(input)
        loss = self.loss(res, data)

        return FeedforwardResult(res, loss)
