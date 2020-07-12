import torch
import torch.nn
import torch.nn.functional as F
from typing import Dict, Any
from dataclasses import dataclass
from ..common import separate_output_digits
from ..result import Result
from ..model_interface import ModelInterface


@dataclass
class TupleRunResult(Result):
    reference_order: torch.Tensor
    loss: torch.Tensor
    n_tuples: int

    @property
    def batch_size(self) -> int:
        return self.reference_order.shape[0]


class TupleArithmeticDatasetInterface(ModelInterface):
    def __init__(self, model: torch.nn.Module, n_tuples: int, steps_per_tuple: int = 3, mode: str = "together"):
        self.model = model
        self.steps_per_tuple = steps_per_tuple
        self.n_tuples = n_tuples
        self.train_only_tuple = -1
        self.mode = mode

    def apply_mode(self, data: torch.Tensor) -> torch.Tensor:
        if self.mode in ["together", "same_output"]:
            res = data
        elif self.mode in ["only_one", "only_one_io"]:
            res = torch.zeros_like(data)
            for t in range(self.n_tuples):
                this_tuple = slice(t * self.steps_per_tuple, (t + 1) * self.steps_per_tuple)
                res[this_tuple, :, t] = data[this_tuple, :, t]
        elif self.mode in ["same_input", "same_io"]:
            res = torch.zeros_like(data[:,:,0])
            for t in range(self.n_tuples):
                this_tuple = slice(t * self.steps_per_tuple, (t + 1) * self.steps_per_tuple)
                res[this_tuple] = data[this_tuple, :, t]
        else:
            assert False, f"Invalid mode: {self.mode}"

        return res

    def create_input(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        onehot_inputs = F.one_hot(data["input"].long(), 10)
        if self.train_only_tuple >= 0:
            onehot_inputs[:, 0:self.train_only_tuple].fill_(0)
            onehot_inputs[:, self.train_only_tuple+1:].fill_(0)

        onehot_inputs = onehot_inputs.float()
        res = onehot_inputs.unsqueeze(0).expand(self.n_tuples * self.steps_per_tuple, *([-1]*onehot_inputs.ndim))
        res = self.apply_mode(res)
        return res.flatten(2)

    def restrict(self, train_only_tuple: int):
        self.train_only_tuple = train_only_tuple

    def to_reference_order(self, outputs: torch.Tensor) -> torch.Tensor:
        if self.mode in ["together", "only_one", "same_input"]:
            res = outputs[-1].view(outputs.shape[1], self.n_tuples, -1)
        elif self.mode in ["same_output", "same_io"]:
            res = outputs[self.steps_per_tuple - 1 :: self.steps_per_tuple].transpose(1,0)
        elif self.mode in ["only_one_io"]:
            res = outputs.view(*outputs.shape[:2], self.n_tuples, -1)
            return torch.stack([res[self.steps_per_tuple * (i+1) - 1, :, i] for i in range(self.n_tuples)], dim=1)
        else:
            assert False, f"Invalid mode {self.mode}"

        return res

    def loss(self, outputs: torch.Tensor, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = separate_output_digits(outputs)
        ref = data["output"]

        if self.train_only_tuple >= 0:
            outputs = outputs[:, self.train_only_tuple:self.train_only_tuple+1]
            ref = ref[:, self.train_only_tuple:self.train_only_tuple+1]

        return F.cross_entropy(outputs.flatten(end_dim=-2), ref.long().flatten())

    def decode_outputs(self, outputs: TupleRunResult) -> torch.Tensor:
        return separate_output_digits(outputs.reference_order).argmax(-1)

    def __call__(self, data: Dict[str, torch.Tensor]) -> TupleRunResult:
        res = self.model(self.create_input(data))
        reforder = self.to_reference_order(res)

        loss = self.loss(reforder, data)
        return TupleRunResult(reforder, loss, self.n_tuples)
