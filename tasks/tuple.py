from .task import Task
import dataset
from models import LSTMModel
from interfaces.recurrent import TupleArithmeticDatasetInterface
import torch


class TupleTask(Task):
    model_interface: TupleArithmeticDatasetInterface

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_datasets(self):
        self.batch_dim = 0
        self.train_set = dataset.TupleArithmetic("train", self.helper.opt.n_digits, 2, n_samples=1000000)
        self.valid_sets.iid = dataset.TupleArithmetic("valid", self.helper.opt.n_digits, 2, n_samples=10000)

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

    def get_n_masks(self) -> int:
        return 3

    def set_tuple_post_train_stage(self, stage: int):
        self.model.set_active(stage)
        self.set_optimizer(torch.optim.Adam(self.model.masks[stage].parameters(), self.helper.opt.mask_lr or
                                            self.helper.opt.lr))

        self.model_interface.restrict(stage - 1)

    def train_step_reconfig(self):
        if self.helper.opt.task=="tuple":
            if self.helper.state.iter == 0 and self.helper.opt.tuple.tuple2_delay>0:
                print("Disabling training of 2nd tuple.")
                self.model_interface.restrict(0)
            elif self.helper.state.iter == self.helper.opt.tuple.tuple2_delay:
                print(f"Iteration {self.helper.state.iter}: Enabling 2nd tuple...")
                self.model_interface.restrict(-1)

    def post_train(self):
        for stage in range(3):
            start = self.helper.state.iter

            loader = self.create_train_loader(self.train_set, 1234)
            self.set_tuple_post_train_stage(stage)

            for d in loader:
                if self.helper.state.iter - start > self.helper.opt.step_per_mask:
                    self.model_interface.restrict(-1)
                    log = self.do_inverse_mask_test(stage)
                    log.update({f"final_accuracy/{stage}/{k}": v for k, v in self.validate().items()})
                    self.export_masks(stage)
                    self.helper.summary.log(log)
                    break

                res = self.train_step(d)

                plots = self.plot(res)
                if self.helper.state.iter % 1000 == 0:
                    plots.update(self.plot_masks(stage))

                self.helper.summary.log(plots)

        self.helper.summary.log(self.plot_remaining_stat(0, range(1, len(self.model.masks))))
        self.helper.summary.log(self.plot_mask_sharing(range(1, len(self.model.masks))))

