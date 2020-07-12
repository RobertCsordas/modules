from dataclasses import dataclass
from typing import Union
import math
import numpy as np


@dataclass
class Stat:
    mean: Union[np.ndarray, float]
    std: Union[np.ndarray, float]
    n: int


class StatTracker:
    def __init__(self):
        self.sum = 0
        self.sqsum = 0
        self.n = 0

    def add(self, v: float):
        if isinstance(v, np.ndarray):
            v = v.astype(np.float32)
        self.sum = self.sum + v
        self.sqsum = self.sqsum + v**2
        self.n += 1

    def get(self) -> Stat:
        assert self.n > 0
        mean = self.sum / self.n
        var = (self.sqsum / self.n - mean ** 2) * self.n/(self.n-1)

        return Stat(mean, np.sqrt(np.maximum(var,0)), self.n)

    def __repr__(self) -> str:
        s = self.get()
        return f"Stat(mean: {s.mean}, std: {s.std})"
