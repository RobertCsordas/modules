import torch
from typing import Dict
from ..feedforward.add_mul import FFAddMulInterface


class RecurrentAddMulInterface(FFAddMulInterface):
    def __init__(self, model: torch.nn.Module, n_steps: int):
        super().__init__(model)
        self.n_steps = n_steps

    def create_input(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        inputs = super().create_input(data)
        return inputs. unsqueeze(0).expand(self.n_steps, *inputs.shape)

    def to_reference_order(self, net_out: torch.Tensor) -> torch.Tensor:
        return super().to_reference_order(net_out[-1])
