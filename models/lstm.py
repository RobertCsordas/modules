import torch
import torch.nn
import layers


class LSTMModel(torch.nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int, state_size: int, n_layers: int):
        super().__init__()

        self.lstm = layers.LSTM(n_inputs, state_size, n_layers)
        self.output_projection = layers.Linear(state_size, n_outputs)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        net, _ = self.lstm(data)
        return self.output_projection(net)