import torch
from layers import Conv2d, Linear

class ConvModel(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: bool = True):
        super().__init__()

        self.features = torch.nn.Sequential(
            Conv2d(in_channels, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            Conv2d(32, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            Conv2d(64, 128, 3, padding=1),
            torch.nn.Dropout(0.25 if dropout else 0.0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            Conv2d(128, 256, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5 if dropout else 0.0)
        )

        # Certain neurons play a crucial role

        self.out_layer = Linear(256, out_channels)

    def __call__(self, inp: torch.Tensor) -> torch.Tensor:
        return self.out_layer(self.features(inp).mean(dim=(2,3)))