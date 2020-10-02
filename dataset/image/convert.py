from PIL import Image
import numpy as np
import torch


def np_to_pil(img: np.ndarray) -> Image:
    return Image.fromarray(np.uint8(img.transpose(1,2,0)))


def np_to_tensor(img: np.ndarray) -> torch.Tensor:
    return torch.Tensor(img)
