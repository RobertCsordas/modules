from .task import Task
import torch
from models import TransformerEncDecModel
from interfaces import TransformerEncDecInterface
import math


class TransformerTask(Task):
    def create_model(self) -> torch.nn.Module:
        return TransformerEncDecModel(len(self.train_set.vocabulary),
                                    len(self.train_set.vocabulary), self.helper.opt.state_size,
                                    nhead=self.helper.opt.transformer.n_heads,
                                    num_encoder_layers=self.helper.opt.transformer.encoder_n_layers,
                                    num_decoder_layers=self.helper.opt.transformer.decoder_n_layers,
                                    ff_multipiler=self.helper.opt.transformer.ff_multiplier,
                                    tied_embedding=True)

    def create_model_interface(self):
        self.model_interface = TransformerEncDecInterface(self.model)

    def set_lr(self):
        if self.helper.opt.transformer.use_paper_lr_schedule:
            i = self.helper.state.iter + 1
            lr = self.helper.opt.lr * math.sqrt(4000) * min(i * 4000**(-1.5), i**(-0.5))
            self.set_optimizer_lr(lr)
        else:
            super().set_lr()