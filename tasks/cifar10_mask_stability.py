import framework
from .cifar10_class_removal import Cifar10ClassRemovalTask
from typing import Iterable
import dataset
import torch
import numpy as np


class Cifar10MaskStabilityTask(Cifar10ClassRemovalTask):
    def get_n_masks(self) -> int:
        return 2*len(self.train_set.class_names)

    def get_mask_iou(self, stages: Iterable[int]) -> float:
        i = {k: torch.ones_like(v) for k, v in self.model.get_masks(0).items()}
        u = {k: torch.zeros_like(v) for k, v in self.model.get_masks(0).items()}

        for j in stages:
            masks = self.model.get_masks(j)
            u = {k: v | masks[k] for k, v in u.items()}
            i = {k: v & masks[k] for k, v in i.items()}

        i = sum(a.long().sum().item() for a in i.values())
        u = sum(a.long().sum().item() for a in u.values())
        return i/u

    def post_train(self):
        # Disable dropout
        self.model.set_model_to_eval()

        rnd = framework.utils.seed.get_randstate(1234)

        all_ious = []

        def measure(set, stage: int, split: str):
            self.create_validate_on_train(set)

            for substage in range(2):
                print(f"Training on {split}, substage {substage}")
                start = self.helper.state.iter

                self.mask_grad_norm.clear()
                self.model.set_active(2 * stage + substage)
                self.class_removal_init_masks_and_optim(2 * stage + substage)

                loader = self.create_train_loader(set, rnd.randint(0x7FFFFFFF))

                for d in loader:
                    if self.helper.state.iter - start > self.helper.opt.step_per_mask:
                        if substage == 1:
                            iou = self.get_mask_iou([2 * stage, 2 * stage + 1])
                            all_ious.append(iou)
                            self.helper.summary.log({f"masks_stability/{split}/iou": iou})
                        break

                    res = self.train_step(d)

                    plots = self.plot(res)
                    plots.update({f"analyzer/{split}/{k}": v for k, v in plots.items()})

                    if self.helper.state.iter % 1000 == 0 and self.helper.opt.analysis.plot_masks:
                        plots.update({f"analysis/{split}/{k}": v for k, v in
                                      self.plot_selected_masks([0, stage] if stage > 0 else [0]).items()})

                    self.helper.summary.log(plots)

        if self.helper.opt.mask_stability.measure_on == "minimal":
            c = np.random.choice(90, 10, replace=False).tolist()
            c = [(a // 10, a % 10) for a in c]
            c = [(a[0]+int(a[1]<=a[0]), a[1]) for a in c]

            for stage, pair in enumerate(c):
                measure(dataset.image.CIFAR10("train", restrict=pair), stage, f"pair_{pair[0]}_{pair[1]}")
        else:
            for stage, split in enumerate(self.train_set.class_names):
                set = dataset.image.CIFAR10("train", restrict=[i for i in range(self.train_set.n_classes)
                                                               if i != stage])

                measure(set, stage, split)

        self.helper.summary.log({f"masks_stability/average_iou": sum(all_ious)/len(all_ious)})
