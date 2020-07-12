import torch
import torch.nn.functional as F
from typing import Dict, Optional, List
from models import FeedforwardModel
from ..result import Result
from dataclasses import dataclass
from ..model_interface import ModelInterface


@dataclass
class TupleRunResultFF(Result):
    output: torch.Tensor
    loss: torch.Tensor
    n_tuples: int

    @property
    def batch_size(self) -> int:
        return self.output.shape[0]


class FFTupleArithmeticDatasetInterface(ModelInterface):
    def __init__(self, model: FeedforwardModel, n_tuples: int, n_digits: int):
        self.model = model
        self.train_only_tuple = -1
        self.n_tuples = n_tuples
        self.n_digits = n_digits

    def restrict(self, train_only_tuple: int):
        self.train_only_tuple = train_only_tuple

    def create_input(self, data: Dict[str, torch.Tensor], tuple: Optional[int]) -> torch.Tensor:
        onehot_inputs = F.one_hot(data["input"].long(), 10)
        if tuple is not None:
            onehot_inputs[:, 0:tuple].fill_(0)
            onehot_inputs[:, tuple+1:].fill_(0)

        return onehot_inputs.float().flatten(1)

    def to_reference_order(self, outputs: List[torch.Tensor]) -> torch.Tensor:
        outputs = [o.view(o.shape[0], -1, self.n_digits, 10)[:, i:i+1] for i, o in enumerate(outputs)]
        return torch.cat(outputs, 1)

    def loss(self, outputs: torch.Tensor, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        ref = data["output"]

        if self.train_only_tuple >= 0:
            outputs = outputs[:, self.train_only_tuple:self.train_only_tuple+1]
            ref = ref[:, self.train_only_tuple:self.train_only_tuple+1]

        return F.cross_entropy(outputs.flatten(end_dim=-2), ref.long().flatten())

    def decode_outputs(self, outputs: TupleRunResultFF) -> torch.Tensor:
        return outputs.output.argmax(-1)

    def __call__(self, data: Dict[str, torch.Tensor]) -> TupleRunResultFF:
        n_tuples = data["input"].shape[1]
        res = []
        for t in range(n_tuples):
            input = self.create_input(data, t)
            res.append(self.model(input))

        res = self.to_reference_order(res)
        loss = self.loss(res, data)

        return TupleRunResultFF(res, loss, self.n_tuples)
