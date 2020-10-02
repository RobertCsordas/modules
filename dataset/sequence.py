from typing import Tuple, Dict, Any
import torch


class SequenceTestState:
    def __init__(self, batch_dim: int = 1):
        self.n_ok = 0
        self.n_total = 0
        self.batch_dim = batch_dim
        self.time_dim = 1 - self.batch_dim

    def step(self, net_out: Tuple[torch.Tensor, torch.Tensor], data: Dict[str, torch.Tensor]):
        out, len = net_out
        ref = data["out"]

        if out.shape[0] > ref.shape[0]:
            out = out[: ref.shape[0]]
        elif out.shape[0] < ref.shape[0]:
            ref = ref[: out.shape[0]]

        unused = torch.arange(0, out.shape[0], dtype=torch.long, device=ref.device).unsqueeze(self.batch_dim) >=\
                 data["out_len"].unsqueeze(self.time_dim)

        ok_mask = ((out == ref) | unused).all(self.time_dim) & (len == data["out_len"])

        self.n_total += ok_mask.nelement()
        self.n_ok += ok_mask.long().sum().item()

    @property
    def accuracy(self):
        return self.n_ok / self.n_total

    def plot(self) -> Dict[str, Any]:
        return {"accuracy/total": self.accuracy}