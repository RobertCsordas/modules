import torch

def separate_output_digits(outputs: torch.Tensor) -> torch.Tensor:
    return outputs.view(*outputs.shape[:-1], -1, 10)