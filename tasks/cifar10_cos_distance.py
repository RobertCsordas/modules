import framework
from .cifar10_class_removal import Cifar10ClassRemovalTask
from typing import Dict, Any
from tqdm import tqdm
import torch
import dataset


class Cifar10GradCosDistanceTask(Cifar10ClassRemovalTask):
    def get_n_masks(self) -> int:
        return 1

    def measure_grad(self) -> Dict[str, torch.Tensor]:
        # Zero grads without optimizer
        for p in self.model.model_parameters.values():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

        # Calculate grads on the full IID test set
        for data in tqdm(self.valid_loaders["iid"]):
            data = self.helper.to_device(data)
            res = self.model_interface(data)
            res.loss.backward()

        # Return a copy of the grads
        return {k: v.grad.clone() for k, v in self.model.model_parameters.items()}

    def calc_cos_distance(self, grads1: Dict[str, torch.Tensor], grads2: Dict[str, torch.Tensor],
                          masks: Dict[str, torch.Tensor]) -> Dict[str, float]:

        def cos_sim(a, b):
            return ((a * b).sum() / (a.norm() * b.norm())).item()

        cos_distances = {}
        all_ref = []
        all_new = []
        for k, grad_ref in grads1.items():
            if k in masks:
                grad_ref = grad_ref[masks[k]]
                grad_new = grads2[k][masks[k]]
            else:
                print(f"Param {k} doesn't have a mask.")
                grad_ref = grad_ref.view(-1)
                grad_new = grads2[k].view(-1)

            all_ref.append(grad_ref)
            all_new.append(grad_new)
            cos_distances[k] = cos_sim(grad_new, grad_ref)

        cos_distances["total"] = cos_sim(torch.cat(all_ref), torch.cat(all_new))
        return cos_distances

    def post_train(self):
        # Disable dropout
        self.model.set_model_to_eval()

        # Measure gradient of the fully trained network without masks
        baseline_grads = self.measure_grad()

        # Train masks
        start = self.helper.state.iter

        self.mask_grad_norm.clear()
        self.model.set_active(0)
        self.set_optimizer(torch.optim.Adam(self.model.masks[0].parameters(), self.get_mask_lr()))

        set = dataset.image.CIFAR10("train",
                                    restrict=list(range(1,10)))
        self.create_validate_on_train(set)
        loader = self.create_train_loader(set, 1234)

        layer_order = ['out_layer', 'features_10', 'features_6', 'features_3', 'features_0']
        def check_name(name_list, name):
            for n in name_list:
                if n in name:
                    return True
            return False

        for d in loader:
            if self.helper.state.iter - start > self.helper.opt.step_per_mask:
                break

            res = self.train_step(d)

            plots = self.plot(res)
            plots.update({f"analyzer/0/{k}": v for k, v in plots.items()})

            if self.helper.state.iter % 1000 == 0 and self.helper.opt.analysis.plot_masks:
                plots.update({f"analysis/0/{k}": v for k, v in self.plot_selected_masks([0]).items()})

            self.helper.summary.log(plots)


        plots = {}

        for i in range(len(layer_order)+1):
            print("Measuring with masking on layers ", layer_order[:i])
            # Measure grads of the masked network. 0 is the control.
            if i>0:
                masks = self.model.get_masks(0)

                self.model.set_active(0)
                self.model.set_temporary_masks({k: v if check_name(layer_order[:i], k) else torch.ones_like(v)
                                                for k, v in masks.items()})
            else:
                masks = {}
                self.model.set_active(-1)
                self.model.set_temporary_masks(None)

            masked_grads = self.measure_grad()

            # Calculate cos distances of the grads of the remaining weights
            plots.update({f"cos_distance/{i}/{k}": v for k, v in self.calc_cos_distance(baseline_grads, masked_grads,
                                                                                        masks).items()})

        # Plot
        self.helper.summary.log(plots)