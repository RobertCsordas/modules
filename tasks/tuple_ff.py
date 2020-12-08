from .tuple import TupleTask
from interfaces import FFTupleArithmeticDatasetInterface
from models import FeedforwardModel
from typing import List, Set
import torch
from masked_model import Masks


class TupleTaskFeedforward(TupleTask):
    model_interface: FFTupleArithmeticDatasetInterface

    def create_model(self) -> torch.nn.Module:
        return FeedforwardModel(self.train_set.in_channels()*10,  self.train_set.out_channels()*10,
                                                            self.helper.opt.layer_sizes)

    def create_model_interface(self):
        self.model_interface = FFTupleArithmeticDatasetInterface(self.model, 2, self.helper.opt.n_digits)

    def get_half_mask_masked_layer_names(self, masks: Masks) -> List[Set[str]]:
        names = list(sorted(masks.keys()))
        return [set(names[:len(names) // 2])]

    def inv_mask_test_get_exluded(self) -> Set[str]:
        return {k for k in self.model.masks[0].keys() if k.startswith("layers_0_") or
                k.startswith(f"layers_{len(self.helper.opt.layer_sizes)}_")}
