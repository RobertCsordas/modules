from .task import Task
import dataset
from models import EncoderDecoder
from interfaces.recurrent import EncoderDecoderInterface
import torch


class ScanTask(Task):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_datasets(self):
        self.batch_dim = 1
        self.train_set = dataset.Scan(["train"])
        self.valid_sets.iid = dataset.Scan(["test"])

        for s in self.helper.opt.scan.analyze_splits:
            self.valid_sets[s] = dataset.Scan(["test"], split_type=[s])

    def create_model(self) -> torch.nn.Module:
        return EncoderDecoder(len(self.train_set.in_vocabulary),
                                    len(self.train_set.out_vocabulary), self.helper.opt.state_size,
                                    self.helper.opt.n_layers,
                                    self.helper.opt.encoder_decoder.embedding_size,
                                    self.helper.opt.dropout,
                                    self.train_set.max_out_len)

    def create_model_interface(self):
        self.model_interface = EncoderDecoderInterface(self.model)

    def get_n_masks(self) -> int:
        return 1+len(self.helper.opt.scan.analyze_splits)

    def post_train(self):
        # The model is not trained anymore. Dropouts are not needed.
        self.model.model.set_dropout(False)

        for stage, split in enumerate(["simple"] + self.helper.opt.scan.analyze_splits):
            print(f"Scan: Training on {split}.")
            self.mask_grad_norm.clear()
            self.model.set_active(stage)
            self.set_optimizer(torch.optim.Adam(self.model.masks[stage].parameters(), self.helper.opt.mask_lr or
                                                self.helper.opt.lr))

            set = dataset.Scan(["train"], split_type=[split]) if stage > 0 else self.train_set
            loader = self.create_train_loader(set, 1234)
            start = self.helper.state.iter

            self.create_validate_on_train(set)

            for d in loader:
                if self.helper.state.iter - start > self.helper.opt.step_per_mask:
                    plots = {f"analysis_results/{split}/{k}": v for k, v in self.validate().items()}
                    self.helper.summary.log(plots)
                    self.export_masks(stage)
                    break

                res = self.train_step(d)

                plots = self.plot(res)
                plots.update({f"analyzer/{split}/{k}": v for k, v in plots.items()})

                if self.helper.state.iter % 1000 == 0:
                    plots.update({f"analyzer/{split}/{k}": v for k, v in \
                                  self.plot_selected_masks([0] if stage == 0 else [0, stage]).items()})

                    if stage > 0:
                        plots.update({f"analyzer_remaining/{split}/{k}": v for k, v in
                                      self.plot_remaining_stat(0, [stage]).items()})

                self.helper.summary.log(plots)

        self.helper.summary.log(self.plot_remaining_stat(0, range(1, len(self.model.masks))))
        self.helper.summary.log(self.plot_mask_sharing(range(1, len(self.model.masks))))

