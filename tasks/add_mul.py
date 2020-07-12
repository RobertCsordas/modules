from .task import Task
import dataset
from models import LSTMModel
from interfaces.recurrent import RecurrentAddMulInterface
import torch


class AddMulTask(Task):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_datasets(self):
        self.batch_dim = 0
        self.train_set = dataset.AddMul("train", 1000000, self.helper.opt.n_digits)
        self.valid_sets.iid = dataset.AddMul("valid", 10000, self.helper.opt.n_digits)
        self.valid_sets.add = dataset.AddMul("valid", 10000, self.helper.opt.n_digits, restrict=["add"])
        self.valid_sets.mul = dataset.AddMul("valid", 10000, self.helper.opt.n_digits, restrict=["mul"])

    def create_model(self) -> torch.nn.Module:
        return LSTMModel(self.train_set.in_channels(),  self.train_set.out_channels(),
                  self.helper.opt.state_size, self.helper.opt.n_layers)

    def create_model_interface(self):
        self.model_interface = RecurrentAddMulInterface(self.model, 3)

    def get_n_masks(self) -> int:
        return 3

    def post_train(self):
        for stage, split in enumerate(["baseline", "add", "mul"]):
            start = self.helper.state.iter

            self.mask_grad_norm.clear()
            self.model.set_active(stage)
            self.set_optimizer(torch.optim.Adam(self.model.masks[stage].parameters(), self.helper.opt.mask_lr or
                                                self.helper.opt.lr))

            set = dataset.AddMul("train", 1000000, self.helper.opt.n_digits, restrict=[split]) if stage!=0 else\
                  self.train_set

            self.create_validate_on_train(set)

            loader = self.create_train_loader(set, 1234)

            for d in loader:
                if self.helper.state.iter - start > self.helper.opt.step_per_mask:
                    test, _ = self.validate_on(self.valid_sets.iid, self.valid_loaders.iid)
                    self.export_tensor(f"confusion/{split}", test.confusion)
                    self.export_masks(stage)
                    self.helper.summary.log(self.do_inverse_mask_test(stage, split))
                    break

                res = self.train_step(d)

                plots = self.plot(res)
                plots.update({f"analyzer/{split}/{k}": v for k, v in plots.items()})

                if self.helper.state.iter % 1000 == 0:
                    plots.update(self.plot_masks(stage))

                self.helper.summary.log(plots)

        self.helper.summary.log(self.plot_remaining_stat(0, range(1, len(self.model.masks))))
        self.helper.summary.log(self.plot_mask_sharing(range(1, len(self.model.masks))))
