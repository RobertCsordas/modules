import os
import numpy as np
from framework.utils import LockFile, U
from framework.visualize import plot
import functools
import torch
import torch.utils.data
import operator
from typing import List, Tuple, Optional, Dict, Any
from tqdm import tqdm
from dataclasses import dataclass
import torch.nn.functional as F


class ImageSetTest:
    def __init__(self, owner):
        self.owner = owner

        self.n_ok = 0
        self.n_total = 0
        self.confusion = 0

    def confusion_matrix(self, ref: torch.Tensor, prediction: torch.Tensor):
        ref = F.one_hot(ref.long(), self.owner.n_classes)
        prediction = F.one_hot(prediction.long(), self.owner.n_classes)
        return (ref.unsqueeze(-2) * prediction.unsqueeze(-1)).long().sum(0)

    def step(self, net_out: torch.Tensor, data: Dict[str, torch.Tensor]):
        net_out = net_out.to(data["label"].dtype)
        ok_bin = (net_out == data["label"]).long()

        self.confusion = self.confusion_matrix(data["label"], net_out) + self.confusion
        self.n_ok += ok_bin.sum().item()
        self.n_total += ok_bin.shape[0]

    @property
    def accuracy(self) -> float:
        return self.n_ok / self.n_total

    def plot(self) -> Dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "confusion": plot.ConfusionMatrix(self.confusion, class_names=getattr(self.owner, "class_names", None))
        }


@dataclass
class ImageSet:
    images: np.ndarray
    labels: np.ndarray

    def filter(self, mask: np.ndarray):
        return ImageSet(self.images[mask], self.labels[mask])

    def __len__(self):
        return self.images.shape[0]

    def filter_by_classes(self, classes: List[int]):
        return self.filter(functools.reduce(np.logical_or, [self.labels == c for c in classes]))

    def normalize(self, mean: np.ndarray, std: np.ndarray):
        return ImageSet((self.images.astype(np.float32) - mean) / std, self.labels)


class ImageClassificationDataset(torch.utils.data.Dataset):
    load_sets: Dict[str, ImageSet] = {}

    def download(self, cache: str):
        raise NotImplementedError

    def load_data(self, cache_dir: str, set: str) -> ImageSet:
        raise NotImplementedError

    def get_input_size(self) -> int:
        return functools.reduce(operator.mul, self.data.images.shape[1:])

    def in_channels(self) -> int:
        return self.data.images.shape[1]

    def get_cache_key(self, cache_dir: str, set: str) -> Any:
        return (cache_dir, self.__class__.__name__, set)

    def load_cached(self, cache_dir: str, set: str) -> ImageSet:
        key = self.get_cache_key(cache_dir, set)
        data = ImageClassificationDataset.load_sets.get(key)
        if data is None:
            data = self.load_data(cache_dir, set)
            ImageClassificationDataset.load_sets[key] = data
        return data

    def load_mean_std(self, cache: str):
        mean_std_fname = os.path.join(cache, "mean_std.pth")
        if not os.path.isfile(mean_std_fname):
            self.measure_mean_std(cache)
            torch.save({"mean": self.mean_tensor, "std": self.std_tensor}, mean_std_fname)
        else:
            data = torch.load(mean_std_fname)
            self.mean_tensor = data["mean"]
            self.std_tensor = data["std"]

    def check_if_downloaded(self, cache: str) -> bool:
        return os.path.isdir(cache)

    def split_train_valid(self, set: str, valid_split_size: float):
        if set in ["train", "valid"]:
            l = len(self.data)

            filter = np.random.RandomState(0xC1CAFA52).rand(l) > valid_split_size

            if set == "valid":
                filter = ~filter

            self.data = self.data.filter(filter)

    def __init__(self, set: str, cache: str = "./cache", valid_split_size: float = 0.2,
                 normalize: bool = True, restrict:Optional[List[int]] = None):

        cache = os.path.join(cache, self.__class__.__name__)
        assert hasattr(self, "n_classes"), "n_classes must be defined for all classification datasets"

        with LockFile("/tmp/download_lock"):
            if not self.check_if_downloaded(cache):
                os.makedirs(cache, exist_ok=True)
                self.download(cache)

        self.load_mean_std(cache)
        self.data = self.load_cached(cache, set)
        self.split_train_valid(set, valid_split_size)

        if restrict is not None:
            self.data = self.data.filter_by_classes(restrict)

        if normalize:
            self.data = self.data.normalize(self.mean_tensor, self.std_tensor)

    def __getitem__(self, item: int) -> Dict[str, any]:
        return {
            "image": self.data.images[item],
            "label": int(self.data.labels[item])
        }

    def __len__(self) -> int:
        return len(self.data)

    def start_test(self) -> ImageSetTest:
        return ImageSetTest(self)

    def measure_mean_std(self, cache):
        data = self.load_cached(cache, "train")

        print("MEASURING MEAN STD")
        images = data.images.astype(np.float32)
        sum = np.sum(images, axis=(0,2,3))
        sq_sum = np.sum(images**2, axis=(0,2,3))
        cnt = data.images.shape[0]*data.images.shape[2]*data.images.shape[3]

        mean = sum / cnt
        var = (sq_sum / cnt - mean ** 2) * cnt/(cnt-1)
        mean = mean * cnt/(cnt-1)
        std = np.sqrt(var)

        assert np.isfinite(std).all() and np.isfinite(mean).all()

        print(f"MEAN {mean}\nSTD {std}")

        self.mean_tensor = np.expand_dims(mean, [-1,-2]).astype(np.float32)
        self.std_tensor = np.expand_dims(std, [-1,-2]).astype(np.float32)
