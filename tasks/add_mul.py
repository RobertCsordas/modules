from .task import Task, TaskDataset
import dataset
from models import LSTMModel
from interfaces.recurrent import RecurrentAddMulInterface
import torch
from typing import Dict, Any


class AddMulTask(Task):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_datasets(self):
        self.batch_dim = 0
        self.train_set = dataset.AddMul("train", 1000000, self.helper.opt.n_digits)
        self.valid_sets.iid = dataset.AddMul("valid", 10000, self.helper.opt.n_digits)
        self.valid_sets.add = dataset.AddMul("valid", 10000, self.helper.opt.n_digits, restrict=["add"])
        self.valid_sets.mul = dataset.AddMul("valid", 10000, self.helper.opt.n_digits, restrict=["mul"])

        for n in ["add", "mul"]:
            self.tasks.append(
                TaskDataset(n, (lambda name: lambda: dataset.AddMul("train", 1000000,
                                self.helper.opt.n_digits, restrict=[name]))(n), self.valid_sets.iid)
            )

    def create_model(self) -> torch.nn.Module:
        return LSTMModel(self.train_set.in_channels(),  self.train_set.out_channels(),
                  self.helper.opt.state_size, self.helper.opt.n_layers)

    def create_model_interface(self):
        self.model_interface = RecurrentAddMulInterface(self.model, 3)

    def analysis_stage_finished(self, index: int, name: str):
        test, _ = self.validate_on(self.valid_sets.iid, self.valid_loaders.iid)
        self.export_tensor(f"confusion/{name}", test.confusion)
        self.export_masks(index)
        log = self.do_inverse_mask_test(index, name)
        if index==0:
            log.update(self.do_half_mask_test(index, name))
        self.helper.summary.log(log)

    def analysis_periodic_plot(self, index: int, name: str) -> Dict[str, Any]:
        return self.plot_masks(index)

    def post_train(self):
        super().post_train()
        self.helper.summary.log(self.plot_mask_sharing(range(1, len(self.model.masks))))
