from .tuple import TupleTask
from interfaces import FFTupleArithmeticDatasetInterface
from models import FeedforwardModel
import torch


class TupleTaskFeedforward(TupleTask):
    model_interface: FFTupleArithmeticDatasetInterface

    def create_model(self) -> torch.nn.Module:
        return FeedforwardModel(self.train_set.in_channels()*10,  self.train_set.out_channels()*10,
                                                            self.helper.opt.layer_sizes)

    def create_model_interface(self):
        self.model_interface = FFTupleArithmeticDatasetInterface(self.model, 2, self.helper.opt.n_digits)
