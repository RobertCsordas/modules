from .task import Task
import dataset
from models import ConvModel
from interfaces import ConvClassifierInterface
import torch
import framework
from framework.visualize import plot


class Cifar10ClassRemovalTask(Task):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_datasets(self):
        self.batch_dim = 0
        self.train_set = dataset.image.CIFAR10("train")
        self.valid_sets.iid = dataset.image.CIFAR10("valid")

    def create_model(self):
        return ConvModel(self.train_set.in_channels(), self.train_set.out_channels())

    def create_model_interface(self):
        self.model_interface = ConvClassifierInterface(self.model)

    def get_n_masks(self) -> int:
        return 1+len(self.train_set.class_names)

    def class_removal_init_masks_and_optim(self, stage: int):
        if self.helper.opt.class_removal.keep_last_layer and stage > 0:
            names = list(sorted(self.model.masks[stage].keys()))
            last_layer_prefix = "out_layer_"

            copy_names = [n for n in names if n.startswith(last_layer_prefix)]
            optimize_names = [n for n in names if not n.startswith(last_layer_prefix)]

            print("Optimizing: ", optimize_names)

            assert len(copy_names) >= 2

            with torch.no_grad():
                for cn in copy_names:
                    self.model.masks[stage][cn].copy_(self.model.masks[0][cn])

            params = [self.model.masks[stage][n] for n in optimize_names]
        else:
            params = self.model.masks[stage].parameters()

        self.set_optimizer(torch.optim.Adam(params, self.helper.opt.mask_lr or self.helper.opt.lr))

    def draw_confusion_heatmap(self, hm: torch.Tensor) -> framework.visualize.plot.Heatmap:
        return plot.Heatmap(hm, "predicted", "real", round_decimals=2, x_marks=self.train_set.class_names,
                     y_marks=self.train_set.class_names)

    def post_train(self):
        for stage, split in enumerate(["baseline"] + self.train_set.class_names):
            start = self.helper.state.iter

            self.mask_grad_norm.clear()
            self.model.set_active(stage)
            self.class_removal_init_masks_and_optim(stage)

            set = dataset.image.CIFAR10("train",
                                        restrict=[i for i in range(self.train_set.n_classes) if i != (stage - 1)])
            self.create_validate_on_train(set)
            loader = self.create_train_loader(set, 1234)

            for d in loader:
                if self.helper.state.iter - start > self.helper.opt.step_per_mask:
                    test, _ = self.validate_on(self.valid_sets.iid, self.valid_loaders.iid)
                    confusion = test.confusion.type(torch.float32)
                    confusion = (confusion / confusion.sum(dim=0, keepdim=True)).transpose(1, 0)
                    if stage == 0:
                        confusion_ref = confusion
                        self.export_tensor("class_removal/confusion_reference", confusion_ref)
                        log = {"class_removal/confusion_reference": self.draw_confusion_heatmap(confusion_ref)}
                    else:
                        diff = confusion - confusion_ref
                        log_name = f"class_removal/confusion_difference/{split}"
                        self.export_tensor(log_name, diff)
                        log = {log_name: self.draw_confusion_heatmap(diff)}
                        log.update({f"class_removal/mask_remaining/{split}/{k}": v for k, v in
                                    self.plot_remaining_stat(0, [stage]).items()})

                    self.helper.summary.log(log)
                    self.export_masks(stage)
                    break

                res = self.train_step(d)

                plots = self.plot(res)
                plots.update({f"analyzer/{split}/{k}": v for k, v in plots.items()})

                if self.helper.state.iter % 1000 == 0:
                    plots.update({f"class_removal_masks/{split}/{k}": v for k, v in
                                  self.plot_selected_masks([0, stage] if stage > 0 else [0]).items()})

                self.helper.summary.log(plots)