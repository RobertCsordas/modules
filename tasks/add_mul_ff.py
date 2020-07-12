from .add_mul import AddMulTask
from interfaces import FFAddMulInterface
from models import FeedforwardModel
import torch


class AddMulFeedforward(AddMulTask):
    def create_model(self):
        return FeedforwardModel(self.train_set.in_channels(),  self.train_set.out_channels(),
                                                            self.helper.opt.layer_sizes)

    def create_model_interface(self) -> torch.nn.Module:
        self.model_interface = FFAddMulInterface(self.model)
