from .task import Task, TaskDataset
import dataset
from models import LSTMModel
from interfaces.recurrent import TupleArithmeticDatasetInterface
import torch
from typing import Dict, Any


class TupleTask(Task):
    ANALYZE_TRAIN_SET = False
    model_interface: TupleArithmeticDatasetInterface

    def create_datasets(self):
        self.batch_dim = 0
        self.train_set = dataset.TupleArithmetic("train", self.helper.opt.n_digits, 2, n_samples=1000000)
        self.valid_sets.iid = dataset.TupleArithmetic("valid", self.helper.opt.n_digits, 2, n_samples=10000)

        for i in range(3):
            self.tasks.append(TaskDataset(str(i), self.train_set, self.valid_sets.iid))

    def calculate_output_size(self):
        return (2 if self.helper.opt.tuple.mode not in ["same_output", "same_io"] else 1) * \
               10 * self.helper.opt.n_digits

    def calculate_input_size(self):
        return (2 if self.helper.opt.tuple.mode not in ["same_input", "same_io"] else 1) * \
               10 * 2 * self.helper.opt.n_digits

    def create_model(self) -> torch.nn.Module:
        return LSTMModel(self.calculate_input_size(),  self.calculate_output_size(),
                               self.helper.opt.state_size, self.helper.opt.n_layers)

    def create_model_interface(self):
        self.model_interface = TupleArithmeticDatasetInterface(self.model, 2, mode=self.helper.opt.tuple.mode)

    def set_mask_stage(self, index: int, name: str):
        super().set_mask_stage(index, name)
        self.model_interface.restrict(index - 1)

    def train_step_reconfig(self):
        if self.helper.opt.task=="tuple":
            if self.helper.state.iter == 0 and self.helper.opt.tuple.tuple2_delay>0:
                print("Disabling training of 2nd tuple.")
                self.model_interface.restrict(0)
            elif self.helper.state.iter == self.helper.opt.tuple.tuple2_delay:
                print(f"Iteration {self.helper.state.iter}: Enabling 2nd tuple...")
                self.model_interface.restrict(-1)

    def analysis_stage_finished(self, index: int, name: str):
        self.model_interface.restrict(-1)
        log = self.do_inverse_mask_test(index)
        log.update({f"final_accuracy/{index}/{k}": v for k, v in self.validate().items()})
        self.export_masks(index)
        self.helper.summary.log(log)

    def analysis_periodic_plot(self, index: int, name: str) -> Dict[str, Any]:
        return self.plot_masks(index)

    def post_train(self):
        super().post_train()
        self.helper.summary.log(self.plot_mask_sharing(range(1, len(self.model.masks))))
