import os
import numpy as np
from framework.utils import download
from .classification_dataset import ImageClassificationDataset, ImageSet
from typing import Any


class PermutedMNIST(ImageClassificationDataset):
    FILES = [
        "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"
    ]

    URL = "http://yann.lecun.com/exdb/mnist/"
    CACHE = "mnist"

    def download(self, cache: str):
        for u in self.FILES:
            download(self.URL+u, cache)

    def __init__(self, permutation: int, *args, **kwargs):
        self.permutation = permutation
        self.n_classes = 10

        super().__init__(*args, **kwargs)

    @staticmethod
    def load(file, offset):
        with open(file, 'rb') as f:
            return np.frombuffer(f.read(), np.uint8, offset=offset)

    def get_cache_key(self, cache_dir: str, set: str) -> Any:
        return super().get_cache_key(cache_dir, set) + (self.permutation,)

    def load_data(self, cache_dir: str, set: str) -> ImageSet:
        assert set in ["train", "test", "valid"]

        if set == "test":
            f_offset = 2
        elif set in ["train", "valid"]:
            f_offset = 0

        images = self.load(os.path.join(cache_dir, self.FILES[f_offset][:-3]), 16).reshape(-1, 1, 28, 28).astype(np.uint8)
        labels = self.load(os.path.join(cache_dir, self.FILES[f_offset + 1][:-3]), 8).astype(np.uint8)

        if self.permutation>0:
            seed = np.random.RandomState(self.permutation)
            perm = seed.permutation(28 * 28)
            images = images.reshape(-1, 28 * 28)[:, perm].reshape(-1, 1, 28, 28)

        return ImageSet(images, labels)