import numpy as np
from typing import Optional, List, Dict, Any
import torch
import torch.utils.data
import torch.nn.functional as F
from .helpers import split_digits
from framework.visualize import plot


class DoubleOpTest:
    def __init__(self, owner):
        self.n_ok = 0
        self.n_total = 0
        self.confusion = 0
        self.owner = owner

    def step(self, net_out: torch.Tensor, data: Dict[str, torch.Tensor]):
        eq = (data["all_res"] == net_out.unsqueeze(-2)).all(-1)
        self.n_total += net_out.shape[0]
        self.n_ok += (data["output"] == net_out).all(-1).long().sum().item()

        eq_with_none = torch.cat((eq, ~eq.any(-1, keepdim=True)), -1)
        op = F.one_hot(data["op"].long(), 2)
        self.confusion = self.confusion +\
                         (op.unsqueeze(1) * eq_with_none.unsqueeze(2).type_as(data["op"])).float().sum(0)

    @property
    def accuracy(self) -> float:
        return self.n_ok / self.n_total

    def plot(self) -> Dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "confusion_matrix": plot.ConfusionMatrix(self.confusion, x_marks = self.owner.OP_IDS + ["none"],
                                                     y_marks = self.owner.OP_IDS)
        }


class DoubleOpDataset(torch.utils.data.Dataset):
    DATA = {}
    OP_IDS: List[str]
    SETS = ["train", "test", "valid"]

    def __init__(self, set: str, n_samples: int, n_digits: int, restrict: Optional[List[str]] = None):
        super().__init__()
        self.n_digits = n_digits
        self.set = set
        self.full_name = f"{set}_{n_digits}"
        self.classes = self.OP_IDS

        if self.full_name not in self.DATA:
            seed = np.random.RandomState(0x12345678+self.SETS.index(set))
            self.DATA[self.full_name] = {
                "args": seed.randint(0, 10**n_digits, (n_samples, 2)),
                "op": seed.randint(0, 2, (n_samples,))
            }

        self.data = self.DATA[self.full_name]
        if restrict:
            mask = True
            for r in restrict:
                mask = (self.data["op"] == self.OP_IDS.index(r)) & mask

            self.data = {
                "args": self.data["args"][mask],
                "op": self.data["op"][mask]
            }

    def out_channels(self) -> int:
        return self.n_digits * 10

    def in_channels(self) -> int:
        return self.n_digits*2*10 + 2

    def __len__(self) -> int:
        return self.data["op"].shape[0]

    def __getitem__(self, item: int) -> Dict[str, np.ndarray]:
        args = self.data["args"][item]
        op = self.data["op"][item]
        res = self.get_res(item)

        return {
            "input": split_digits(self.n_digits, args),
            "output": split_digits(self.n_digits, res[op]),
            "op": np.array(op, dtype=np.uint8),
            "all_res": split_digits(self.n_digits, np.stack(res, 0))
        }

    def start_test(self) -> DoubleOpTest:
        return DoubleOpTest(self)


class AddMul(DoubleOpDataset):
    OP_IDS = ["add", "mul"]

    def get_res(self, item: int):
        args = self.data["args"][item]
        res = [args[0] + args[1], args[0] * args[1]]
        max = 10 ** self.n_digits
        return [r % max for r in res]
