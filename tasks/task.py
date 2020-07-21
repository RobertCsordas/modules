import framework
from masked_model import MaskedModel
from interfaces import Result, ModelInterface
import torch
import torch.utils.data
from tqdm import tqdm
from typing import Dict, Any, Iterable, Tuple, Optional
from grad_norm import GradNormTracker
import functools
from draw import draw_mask, draw_mask_histogram
import os


class Task:
    train_loader: torch.utils.data.DataLoader
    valid_loaders: framework.data_structures.DotDict
    model_interface: ModelInterface
    batch_dim: int
    train_set: torch.utils.data.Dataset
    model: MaskedModel

    def __init__(self, helper: framework.helpers.TrainingHelper):
        self.helper = helper
        self.valid_sets = framework.data_structures.DotDict()
        self.loss_average = framework.utils.Average()

        self.create_grad_norm_tracker()
        self.create_datasets()
        self.create_loaders()
        self.create_masked_model()
        self.create_model_interface()
        self.create_optimizer()
        self.create_validate_on_train(self.train_set)

    def create_loaders(self):
        self.train_loader = self.create_train_loader(self.train_set)
        self.valid_loaders = framework.data_structures.DotDict()
        self.valid_loaders.update({k: torch.utils.data.DataLoader(v, batch_size=self.helper.opt.batch_size,
                                   collate_fn=framework.loader.collate.VarLengthCollate(batch_dim=self.batch_dim),
                                   num_workers=1) for k, v in self.valid_sets.items()})


    def create_train_loader(self, loader: torch.utils.data.Dataset, seed: Optional[int] = None) -> \
            torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(loader, batch_size=self.helper.opt.batch_size,
                                           sampler=framework.loader.sampler.InfiniteSampler(
                                               loader, seed = seed),
                                           collate_fn=framework.loader.collate.VarLengthCollate(
                                               batch_dim=self.batch_dim),
                                           num_workers=1, pin_memory=True)

    def create_validate_on_train(self, set: torch.utils.data.Dataset):
        self.valid_sets.train = set
        self.valid_loaders.train = torch.utils.data.DataLoader(set, batch_size=self.helper.opt.batch_size,
                                   collate_fn=framework.loader.collate.VarLengthCollate(batch_dim=self.batch_dim),
                                   sampler=framework.loader.sampler.SubsetSampler(set, len(self.valid_sets.iid)
                                                                          if "iid" in self.valid_sets else 1000),
                                   num_workers=1)


    def create_grad_norm_tracker(self):
        self.param_grad_norm = GradNormTracker()
        self.mask_grad_norm = GradNormTracker()

    def mask_filter(self, name: str) -> bool:
        return (not "embedding" in name) and ((not self.helper.opt.bias_no_mask) or ("bias" not in name))

    def get_n_mask_samples(self):
        return 4

    def create_masked_model_from(self, model: torch.nn.Module):
        self.model = MaskedModel(model, self.get_n_masks(), self.get_n_mask_samples(),
                                 self.helper.opt.mask_loss_weight, mask_filter=self.mask_filter,
                                 empty_init=self.helper.opt.mask_init).to(self.helper.device)
        self.model.set_active(-1)
        self.helper.saver["model"] = self.model

    def create_masked_model(self):
        self.create_masked_model_from(self.create_model())

    def set_optimizer(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer
        self.helper.saver.register("optimizer", optimizer, replace=True)

    def create_optimizer(self):
        self.set_optimizer(torch.optim.Adam(self.model.model_parameters.values(), self.helper.opt.lr))

    def clip_gradients(self):
        if self.helper.opt.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.helper.opt.grad_clip)

    def validate_on(self, set: torch.utils.data.Dataset, loader: torch.utils.data.DataLoader) -> Tuple[Any, float]:
        self.model.eval()

        with torch.no_grad():
            loss_sum = 0

            test = set.start_test()
            for d in tqdm(loader):
                d = self.helper.to_device(d)
                res = self.model_interface(d)
                digits = self.model_interface.decode_outputs(res)
                loss_sum += res.loss.item() * res.batch_size

                test.step(digits, d)

        self.model.train()
        return test, loss_sum / len(set)

    def validate_on_name(self, name: str) -> Tuple[Any, float]:
        return self.validate_on(self.valid_sets[name], self.valid_loaders[name])

    def validate_on_names(self, name_it: Iterable[str]) -> Dict[str, Any]:
        charts = {}
        sum_accuracy = 0
        sum_all_losses = 0

        for name in name_it:
            test, loss = self.validate_on_name(name)

            print(f"Validation accuracy on {name}: {test.accuracy}")
            charts[f"{name}/loss"] = loss
            sum_all_losses += loss
            charts.update({f"{name}/{k}": v for k, v in test.plot().items()})
            sum_accuracy += test.accuracy

        charts["mean_accuracy"] = sum_accuracy / len(self.valid_sets)
        charts["mean_loss"] = sum_all_losses / len(self.valid_sets)
        return charts

    def validate(self) -> Dict[str, Any]:
        return self.validate_on_names(self.valid_sets.keys())

    def plot_grad_norms(self) -> Dict[str, float]:
        if self.model.masking_enabled:
            return {f"masks/{k}": v for k, v in self.mask_grad_norm.get().items()}
        else:
            return {f"params/{k}": v for k, v in self.param_grad_norm.get().items()}

    def plot(self, res: Result) -> Dict[str, Any]:
        plots = {}

        self.loss_average.add(res.loss)

        if self.helper.state.iter % 200 == 0:
            plots.update(res.plot())

        if self.helper.state.iter % 20 == 0:
            plots["train/loss"] = self.loss_average.get()
            plots.update({f"grad_norms/{k}": v for k, v in self.plot_grad_norms().items()})

        if self.helper.state.iter % self.helper.opt.test_interval == 0:
            plots.update({f"validation/{k}": v for k, v in self.validate().items()})

        return plots

    def train(self):
        self.loss_average.reset()

        for d in self.train_loader:
            if (self.helper.opt.stop_after or 10e10) <= self.helper.state.iter:
                break

            self.train_step_reconfig()
            res = self.train_step(d)
            self.helper.summary.log(self.plot(res))

    def track_grad_norms(self):
        self.param_grad_norm.add_dict(self.model.model_parameters)
        if self.model.masking_enabled:
            self.mask_grad_norm.add_dict(self.model.active_masks)

    def train_step(self, data: Dict[str, torch.Tensor]) -> Result:
        data = self.helper.to_device(data)

        self.optimizer.zero_grad()

        res = self.model_interface(data)
        res.loss.backward()

        self.track_grad_norms()

        if self.model.masking_enabled:
            (self.model.mask_loss(self.mask_grad_norm.get() if self.helper.opt.scale_mask_loss else None) /
                                  self.helper.opt.batch_size).backward()

        self.post_backward()

        self.clip_gradients()
        self.optimizer.step()

        self.helper.state.iter += 1

        return res


    def plot_remaining_stat(self, ref: int, others: Iterable[int]) -> Dict[str, float]:
        mask_indices = [ref]+list(others)
        masks = [self.model.get_masks(c) for c in mask_indices]

        res = {}

        n_total = 0
        n_per_mask = [0 for _ in range(len(masks)-1)]

        for k in masks[0].keys():
            this_masks = [m[k] for m in masks]

            n = [(m & this_masks[0]).long().sum().item() for m in this_masks]
            res[f"mask_remaining/layers/{k}/n_ref"] = n[0]

            n_per_mask = [t+m for t, m in zip(n_per_mask, n[1:])]
            n_total += n[0]

            for i, m in enumerate(n[1:]):
                res[f"mask_remaining/layers/{k}/n_{mask_indices[i+1]}"] = m
                res[f"mask_remaining/layers/{k}/remaining_{mask_indices[i+1]}"] = float(m) / n[0]

        res[f"mask_remaining/all/n_total"] = n_total
        res.update({f"mask_remaining/masks/n_{i}": v for i, v in enumerate(n_per_mask)})
        res.update({f"mask_remaining/masks/remaining_{i}": float(v)/n_total for i, v in enumerate(n_per_mask)})

        return res

    def plot_mask_sharing(self, mask_inidices: Iterable[int]) -> Dict[str, float]:
        mask_inidices = list(mask_inidices)
        masks = [self.model.get_masks(c) for c in mask_inidices]

        res = {}
        t_total = 0
        t_per_mask = [0 for _ in masks]

        for k in masks[0].keys():
            this_masks = [m[k] for m in masks]
            share_maps = [m1 & functools.reduce(torch.logical_or, [m2 for j, m2 in enumerate(this_masks) if i != j])
                          for i, m1 in enumerate(this_masks)]

            n_total = float(functools.reduce(torch.logical_or, this_masks).long().sum().item())
            n_shared_per_mask = [float(s.long().sum().item()) for s in share_maps]
            n_per_mask = [float(m.long().sum().item()) for m in this_masks]

            t_total += n_total
            t_per_mask = [a+b for a, b in zip(n_shared_per_mask, t_per_mask)]

            res[f"mask_stat/mask_{k}/n_total"] = n_total
            res.update({f"mask_stat/mask_{k}/n_{mask_inidices[i]}": n for i, n in enumerate(n_per_mask)})
            res.update({f"mask_stat/mask_{k}/shared_{mask_inidices[i]}": n_s/max(n_t,1)
                        for i, (n_s, n_t) in enumerate(zip(n_shared_per_mask, n_per_mask))})

        res[f"mask_stat/all/n_total"] = t_total
        res.update({f"mask_stat/all/n_{mask_inidices[i]}": n for i, n in enumerate(t_per_mask)})
        res.update({f"mask_stat/all/shared_{mask_inidices[i]}": n/t_total for i, n in enumerate(t_per_mask)})

        return res

    def plot_masks(self, upto_channels: int, presence_channel: Optional[int] = 0) -> Dict[str, Any]:
        presence = self.model.get_masks(presence_channel) if presence_channel is not None and \
                                                             upto_channels > presence_channel else {}

        masks = [self.model.get_masks(c) for c in range(0, upto_channels+1) if c != presence_channel]
        if not masks:
            return {}
        res = {f"masks/{k}": framework.visualize.plot.Image(draw_mask([m[k] for m in masks],
                               len(self.model.masks)-1, presence_mask=presence.get(k))) for k in masks[0].keys()}
        res[f"mask_histogram/{upto_channels}"] = draw_mask_histogram(self.model.get_mask_probs(upto_channels))
        res[f"mask_histogram_nonzero/{upto_channels}"] = draw_mask_histogram(self.model.get_mask_probs(upto_channels),
                                                                             threshold=1e-2)
        return res

    def plot_selected_masks(self, indices: Iterable[int]) -> Dict[str, Any]:
        masks = [self.model.get_masks(i) for i in indices]
        res = {f"masks/{k}": framework.visualize.plot.Image(draw_mask([m[k] for m in masks]))
                              for k in masks[0].keys()}
        res.update({f"mask_histogram/{i}": draw_mask_histogram(self.model.get_mask_probs(m))
                    for i, m in enumerate(indices)})
        res.update({f"mask_histogram_nonzero/{i}": draw_mask_histogram(self.model.get_mask_probs(m),
                                                                       threshold=1e-2)
                    for i, m in enumerate(indices)})
        return res

    def do_inverse_mask_test(self, stage: int, split: Optional[str] = None) -> Dict[str, Any]:
        def inverse_mask_test_run(prefix: str) -> Dict[str, Any]:
            plots = self.validate()

            test, _ = self.validate_on(self.valid_sets.iid, self.valid_loaders.iid)
            if hasattr(test, "confusion"):
                self.export_tensor(f"{prefix}/confusion", test.confusion)

            return {f"{prefix}/{k}": v for k, v in plots.items()}

        split = split or f"split_{stage}"

        self.model.set_temporary_masks(self.model.get_masks(stage).invert())
        res = inverse_mask_test_run(f"inverse_mask_test/{split}")
        self.model.set_temporary_masks(None)
        return res

    def export_tensor(self, rel_path: str, data: torch.Tensor):
        path = os.path.join(self.helper.dirs.export, rel_path + ".pth")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(data.detach().cpu().numpy(), path)

    def export_masks(self, stage: int):
        for k, v in self.model.masks[stage].items():
            self.export_tensor(f"stage_final_masks/stage_{stage}/{k}", v)

    def train_step_reconfig(self):
        pass

    def post_backward(self):
        pass

    def create_model_interface(self):
        raise NotImplementedError()

    def create_datasets(self):
        raise NotImplementedError()

    def create_model(self) -> torch.nn.Module:
        raise NotImplementedError()

    def get_n_masks(self) -> int:
        raise NotImplementedError()
