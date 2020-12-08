from .task import Task, TaskDataset
import dataset
from models import EncoderDecoder
from interfaces.recurrent import EncoderDecoderInterface
import torch
from .scan_transformer import TransformerScanTask

class ScanTask(TransformerScanTask):
    def create_model(self) -> torch.nn.Module:
        return EncoderDecoder(len(self.train_set.in_vocabulary),
                                    len(self.train_set.out_vocabulary), self.helper.opt.state_size,
                                    self.helper.opt.n_layers,
                                    self.helper.opt.encoder_decoder.embedding_size,
                                    self.helper.opt.dropout,
                                    self.train_set.max_out_len)

    def create_model_interface(self):
        self.model_interface = EncoderDecoderInterface(self.model, self.model.model.decoder.eos_token)

    def prepare_model_for_analysis(self):
        # The model is not trained anymore. Dropouts are not needed.
        self.model.model.set_dropout(False)
