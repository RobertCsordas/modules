import torch.utils.data
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any
from .helpers import split_digits


@dataclass
class TupleArithmeticData:
    input: np.ndarray
    output: np.ndarray


class TupleArithmeticTestState:
    def __init__(self, batch_dim: int = 0):
        self.n_ok_per_tuple = 0
        self.n_ok_all = 0
        self.n_total = 0
        self.batch_dim = batch_dim
        self.tuples_dim = 1 - self.batch_dim

    def step(self, net_out: torch.Tensor, data: Dict[str, torch.Tensor]):
        # net_out: [n_tuples, n_batch, n_digits] or [n_batch, n_tuples, n_digits]
        tuples_ok = (net_out == data["output"]).all(-1)

        self.n_ok_all += tuples_ok.all(self.tuples_dim).long().sum().item()
        self.n_total += net_out.shape[self.batch_dim]
        self.n_ok_per_tuple = self.n_ok_per_tuple + tuples_ok.long().sum(self.batch_dim).cpu()

    @property
    def accuracy(self):
        return self.n_ok_all / self.n_total

    def plot(self) -> Dict[str, Any]:
        charts = {"accuracy/total": self.accuracy}
        for i in range(self.n_ok_per_tuple.shape[0]):
            charts[f"accuracy/tuple/{i}"] = float(self.n_ok_per_tuple[i].item()) / self.n_total
        return charts


class TupleArithmetic(torch.utils.data.Dataset):
    DATA: Dict[str, TupleArithmeticData] = {}
    SETS = ["train", "test", "valid"]

    def __init__(self, set: str, n_digits: int, n_tuples: int, n_samples: int, op: str = "add"):
        assert set in self.SETS
        self.n_digits = n_digits
        self.n_tuples = n_tuples
        self.n_samples = n_samples
        self.maxnum = 10 ** n_digits
        self.op = op
        self.id = f"{set}_{n_digits}_{n_tuples}_{n_samples}_{op}"
        self.set_index = self.SETS.index(set)

        self.data = self.DATA.get(self.id)
        if self.data is None:
            TupleArithmetic.DATA[self.id] = self.data = self.generate(set)

    def generate(self, set: str) -> TupleArithmeticData:
        assert self.op in ["add", "mul"]
        op_fn = (lambda a, b: a + b) if self.op == "add" else (lambda a, b: a * b)

        seed = np.random.RandomState(0x12345678 + self.SETS.index(set))

        inputs = seed.randint(0, self.maxnum, (self.n_samples, self.n_tuples, 2))
        outputs = op_fn(inputs[:, :, 0], inputs[:, :, 1]) % self.maxnum

        return TupleArithmeticData(inputs, outputs)

    def __getitem__(self, item: int):
        return {
            "input": split_digits(self.n_digits, self.data.input[item]),
            "output": split_digits(self.n_digits, self.data.output[item])
        }

    def __len__(self):
        return self.n_samples

    def in_channels(self) -> int:
        return self.n_digits * self.n_tuples * 2

    def out_channels(self) -> int:
        return self.n_digits * self.n_tuples

    def start_test(self) -> TupleArithmeticTestState:
        return TupleArithmeticTestState()
