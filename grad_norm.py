import torch
from typing import Dict


class GradNormTracker:
    def __init__(self, win_size: int = 10):
        self.clear()
        self.win_size = win_size

    def clear(self):
        self.norms = {}
        self.sums = {}

    def get(self):
        return {k: self.sums[k]/len(v) for k, v in self.norms.items() if v}

    def add(self, name: str, param: torch.nn.Parameter):
        if param.grad is None:
            return

        norm = param.grad.norm().item()
        l = self.norms.get(name)
        if l is None:
            self.norms[name] = [norm]
            self.sums[name] = norm
        else:
            if len(l) > self.win_size:
                self.sums[name] -= l.pop(0)

            l.append(norm)
            self.sums[name] += norm

    def add_dict(self, data: Dict[str, torch.Tensor]):
        for k, v in data.items():
            self.add(k, v)
