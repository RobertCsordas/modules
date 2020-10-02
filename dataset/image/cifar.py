import os
import numpy as np
from framework.utils import download
from .classification_dataset import ImageClassificationDataset, ImageSet
import pickle


class CIFAR(ImageClassificationDataset):
    class_names = None
    n_classes = None

    @staticmethod
    def _unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')

        dict = {k.decode(): v for k, v in dict.items()}
        if "filenames" in dict:
            del dict["filenames"]
        return dict

    @classmethod
    def _unpickle_multiple(cls, dir, files):
        data = [cls._unpickle(os.path.join(dir, f)) for f in files]
        return {
            "data": np.concatenate([d["data"] for d in data], 0).astype(np.float32),
            cls.LABELS_NAME: np.concatenate([d[cls.LABELS_NAME] for d in data], 0).astype(np.uint8)
        }

    def download(self, cache: str):
        download(self.URL, cache)

    def load_labels(self, cache: str):
        if type(self).n_classes is not None:
            return

        type(self).class_names = [l.decode() for l in self._unpickle(os.path.join(cache, self.META_NAME))
                                  [self.LABEL_NAMES]]

        type(self).n_classes = len(type(self).class_names)

    def load_data(self, cache_dir: str, set: str) -> ImageSet:
        cache = os.path.join(cache_dir, self.DIR_IN_ZIP)

        self.load_labels(cache)

        if set == "test":
                data = self._unpickle(os.path.join(cache, self.TEST_NAME))
        else:
            data = self._unpickle_multiple(cache, self.TRAIN_FILES)

        return ImageSet(data["data"].reshape(-1, 3, 32, 32), data[self.LABELS_NAME])

    def out_channels(self) -> int:
        return self.n_classes


class CIFAR10(CIFAR):
    URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    DIR_IN_ZIP = "cifar-10-batches-py"
    LABELS_NAME = "labels"
    TRAIN_FILES = ["data_batch_%d" % i for i in range(1,6)]
    META_NAME = "batches.meta"
    LABEL_NAMES = "label_names"
    TEST_NAME = "test_batch"


class CIFAR100(CIFAR):
    URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    DIR_IN_ZIP = "cifar-100-python"
    CACHE = "cifar100"
    LABELS_NAME = "fine_labels"
    TRAIN_FILES = ["train"]
    META_NAME = "meta"
    LABEL_NAMES = "fine_label_names"
    TEST_NAME = "test"
