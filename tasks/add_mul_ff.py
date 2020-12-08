from .add_mul import AddMulTask
from interfaces import FFAddMulInterface
from models import FeedforwardModel
import torch
from masked_model import Masks
from typing import List, Set
import math


class AddMulFeedforward(AddMulTask):
    def create_model(self):
        return FeedforwardModel(self.train_set.in_channels(),  self.train_set.out_channels(),
                                                            self.helper.opt.layer_sizes)

    def create_model_interface(self) -> torch.nn.Module:
        self.model_interface = FFAddMulInterface(self.model)

    def get_half_mask_masked_layer_names(self, masks: Masks) -> List[Set[str]]:
        names = list(sorted(masks.keys()))
        return [set(names[:math.ceil(len(names)/2)])]
