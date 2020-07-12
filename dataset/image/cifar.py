import os
import numpy as np
from framework.utils import download
from .classification_dataset import ImageClassificationDataset, ImageSet
import pickle


class CIFAR(ImageClassificationDataset):
    @staticmethod
    def _unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')

        dict = {k.decode(): v for k, v in dict.items()}
        del dict["filenames"]
        return dict

    @classmethod
    def _unpickle_multiple(cls, dir, files):
        data = [cls._unpickle(os.path.join(dir, f)) for f in files]
        return {
            "data": np.concatenate([d["data"] for d in data], 0).astype(np.float32),
            "labels": np.concatenate([d[cls.LABELS_NAME] for d in data], 0).astype(np.uint8)
        }

    def download(self, cache: str):
        download(self.URL, cache)

    def load_data(self, cache_dir: str, set: str) -> ImageSet:
        cache = os.path.join(cache_dir, self.DIR_IN_ZIP)

        if set == "test":
            data = self._unpickle(os.path.join(cache, "test_batch"))
        else:
            data = self._unpickle_multiple(cache, self.TRAIN_FILES)

        return ImageSet(data["data"].reshape(-1, 3, 32, 32), data["labels"])


class CIFAR10(CIFAR):
    URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    DIR_IN_ZIP = "cifar-10-batches-py"
    LABELS_NAME = "labels"
    TRAIN_FILES = ["data_batch_%d" % i for i in range(1,6)]
    class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    n_classes = 10

    def out_channels(self) -> int:
        return 10

