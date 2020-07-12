import torch
from typing import Dict
from .conv_classifier_interface import ConvClassifierInterface


class FeedforwardImageClassifierInterface(ConvClassifierInterface):
    def create_input(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        return super().create_input(data).flatten(1)
