import torch
import torch.utils.data
from .. import utils
import numpy as np


class InfiniteSampler(torch.utils.data.Sampler):
    def __init__(self, data_source: torch.utils.data.Dataset, replacement=True, seed=None):
        super().__init__(data_source)
        self.data_source = data_source
        self.replacement = replacement
        self.seed = utils.seed.get_randstate(seed)

    def __iter__(self):
        n = len(self.data_source)
        if self.replacement:
            while True:
                yield self.seed.randint(0, n, dtype=np.int64)
        else:
            i_list = None
            pos = n
            while True:
                if pos >= n:
                    i_list = self.seed.permutation(n).tolist()
                    pos = 0

                sample = i_list[pos]
                pos += 1
                yield sample

    def __len__(self):
        return 0x7FFFFFFF


class FixedRandomSampler(torch.utils.data.Sampler):
    def __init__(self, data_source: torch.utils.data.Dataset):
        super().__init__(data_source)
        self.data_source = data_source
        self.order = utils.seed.get_randstate(0xB0C1FA53).permutation(len(self.data_source)).tolist()

    def __iter__(self):
        for i in self.order:
            yield i

    def __len__(self):
        return len(self.data_source)


class SubsetSampler(torch.utils.data.Sampler):
    def __init__(self, data_source: torch.utils.data.Dataset, n_max: int):
        super().__init__(data_source)
        self.data_source = data_source
        self._len = min(len(self.data_source), n_max)
        self.order = utils.seed.get_randstate(0xB0C1FA53).choice(len(self.data_source), self._len, replace=False)

    def __iter__(self):
        for i in self.order:
            yield i

    def __len__(self):
        return self._len