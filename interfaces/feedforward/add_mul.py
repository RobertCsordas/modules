import torch
from torch.nn import functional as F
from typing import Dict
from ..result import FeedforwardResult
from ..model_interface import ModelInterface


class FFAddMulInterface(ModelInterface):
    def __init__(self, model: torch.nn.Module):
        self.model = model

    def create_input(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        onehot_inputs = F.one_hot(data["input"].long(), 10)
        onehot_op = F.one_hot(data["op"].long(), 2)

        return torch.cat([onehot_inputs.flatten(1), onehot_op.flatten(1)], -1).float()

    def to_reference_order(self, net_out: torch.Tensor) -> torch.Tensor:
        return net_out.view(net_out.shape[0], -1, 10)

    def loss(self, net_out: torch.Tensor, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.cross_entropy(net_out.flatten(end_dim=-2), data["output"].long().flatten())

    def decode_outputs(self, outputs: FeedforwardResult) -> torch.Tensor:
        return outputs.outputs.argmax(-1)

    def __call__(self, data: Dict[str, torch.Tensor]) -> FeedforwardResult:
        input = self.create_input(data)

        res = self.model(input)

        res = self.to_reference_order(res)
        loss = self.loss(res, data)

        return FeedforwardResult(res, loss)
