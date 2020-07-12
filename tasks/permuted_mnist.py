from .task import Task
import dataset
from models import FeedforwardModel
from interfaces import FeedforwardImageClassifierInterface
import torch
from masked_model import Masks
from typing import Tuple, Any, Dict


class PermutedMnistTask(Task):
    def __init__(self, *args, **kwargs):
        self.n_prems = 11
        super().__init__(*args, **kwargs)
        assert not self.helper.opt.bias_no_mask

    def create_train_set(self, permutation: int):
        return dataset.image.PermutedMNIST(permutation, "train")

    def create_datasets(self):
        self.batch_dim = 0
        self.train_set = self.create_train_set(0)
        for perm in range(self.n_prems):
            self.valid_sets[f"perm_{perm}"] = dataset.image.PermutedMNIST(perm, "valid")

    def create_model(self) -> torch.nn.Module:
        return FeedforwardModel(28*28,  10, self.helper.opt.layer_sizes)

    def create_model_interface(self):
        self.model_interface = FeedforwardImageClassifierInterface(self.model)

    def create_masked_model(self):
        model = self.create_model()
        model.reset_parameters()
        self.create_masked_model_from(model)


    def get_n_masks(self) -> int:
        return self.n_prems

    def create_used_masks(self, perm: int) -> Masks:
        res = Masks()
        for i in range(perm):
            res = res | self.model.get_masks(i)
        return res

    def reinit_weights(self, used_masks: Masks):
        active = self.model.active
        self.model.set_active(-1)

        with torch.no_grad():
            backup = {k: v.clone() for k, v in self.model.model_parameters.items()}
            self.model.model.reset_parameters()

            for name, mask in used_masks.items():
                param = self.model.model_parameters[name]
                param.set_(torch.where(mask, backup[name], param))

        self.model.set_active(active)

    def init_masks(self, used_masks: Masks, index: int):
        with torch.no_grad():
            for name, used in used_masks.items():
                self.model.masks[index][name].masked_fill_(used, self.helper.opt.transfer.mask_used_init)
                self.model.masks[index][name].masked_fill_(~used, self.helper.opt.transfer.mask_new_init)

    def set_perm(self, perm: int):
        self.curr_perm = perm
        self.model.set_active(perm)
        self.train_set = self.create_train_set(perm)
        self.train_loader = self.create_train_loader(self.train_set)

        self.weight_masks = self.create_used_masks(perm)
        self.reinit_weights(self.weight_masks)
        self.init_masks(self.weight_masks, perm)

        self.set_optimizer(torch.optim.Adam(list(self.model.model_parameters.values())
                                            + list(self.model.masks[perm].parameters()), self.helper.opt.lr))

    def post_backward(self):
        for k, mask in self.weight_masks.items():
            self.model.model_parameters[k].grad.masked_fill_(mask, 0)

    def train(self):
        pass

    def get_n_mask_samples(self):
        return 8

    def validate_on_name(self, name: str) -> Tuple[Any, float]:
        perm = int(name.split("_")[1]) if name!="train" else self.curr_perm
        active = self.model.active
        self.model.set_active(perm)
        res = super().validate_on_name(name)
        self.model.set_active(active)
        return res

    def log_plots(self, plots: Dict[str, Any]):
        filtered = {k: v for k, v in plots.items() if not k.endswith("/confusion")}
        self.helper.summary.log(filtered)

    def plot_mask_stats(self, perm: int) -> Dict[str, Any]:
        used = self.create_used_masks(perm)
        this_masks = self.model.get_masks(perm)

        count = {k: v.float().sum() for k, v in this_masks.items()}
        shared = {k: v.float().sum() for k, v in (this_masks & used).items()}

        res = {"mask_stat/shared/total": sum(shared.values())/sum(count.values())}
        for k in shared.keys():
            res[f"mask_stat/counts/{k}"] = count[k]
            res[f"mask_stat/shared/{k}"] = shared[k]/max(count[k],1)

        return res

    def post_train(self):
        for perm in range(self.n_prems):
            print(f"Permutation {perm}")
            self.set_perm(perm)

            start = self.helper.state.iter

            for d in self.train_loader:
                if self.helper.state.iter - start > self.helper.opt.stop_after:
                    self.export_masks(perm)

                    plots = self.plot_masks(perm, None)
                    if perm > 0:
                        plots.update(self.plot_mask_stats(perm))

                    plots.update(self.validate_on_names([f"perm_{i}" for i in range(perm)]))

                    plots = {f"permuted_mnist/perm_{perm}/{k}": v for k, v in plots.items()}
                    self.log_plots(plots)
                    break

                res = self.train_step(d)
                plots = self.plot(res)
                self.log_plots(plots)
