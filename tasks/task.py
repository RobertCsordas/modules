import framework
from masked_model import MaskedModel
from interfaces import Result, ModelInterface
import torch
import torch.utils.data
from tqdm import tqdm
from typing import Dict, Any, Iterable, Tuple, Optional, List, Union, Callable, Set
from grad_norm import GradNormTracker
import functools
from draw import draw_mask, draw_mask_histogram
import os
import re
from dataclasses import dataclass
import optimizer
from masked_model import Masks
import math
import numpy as np


@dataclass
class TaskDataset:
    name: str
    get_train_set: Union[torch.utils.data.Dataset, Callable[[], torch.utils.data.Dataset]]
    get_valid_set: Union[torch.utils.data.Dataset, Callable[[], torch.utils.data.Dataset], None]

    @staticmethod
    def _callable_to_dataset(ds: Union[torch.utils.data.Dataset, Callable[[], torch.utils.data.Dataset]]) -> \
            torch.utils.data.Dataset:
        if isinstance(ds, torch.utils.data.Dataset):
            return ds
        else:
            return ds()

    @property
    def train_set(self) -> torch.utils.data.Dataset:
        return self._callable_to_dataset(self.get_train_set)

    @property
    def valid_set(self) -> Optional[torch.utils.data.Dataset]:
        if self.get_valid_set is None:
            return None
        else:
            return self._callable_to_dataset(self.get_valid_set)


class Task:
    train_loader: torch.utils.data.DataLoader
    valid_loaders: framework.data_structures.DotDict
    model_interface: ModelInterface
    batch_dim: int
    train_set: torch.utils.data.Dataset
    model: MaskedModel
    TRAIN_NUM_WORKERS = 1
    ANALYZE_TRAIN_SET = True

    def __init__(self, helper: framework.helpers.TrainingHelper):
        self.helper = helper
        self.valid_sets = framework.data_structures.DotDict()
        self.loss_average = framework.utils.Average()
        self.forward_time_meter = framework.utils.ElapsedTimeMeter()
        self.load_time_meter = framework.utils.ElapsedTimeMeter()
        self.plot_time_meter = framework.utils.ElapsedTimeMeter()

        self.tasks: List[TaskDataset] = []

        self.step_lr_scheduler = optimizer.StepLrSched(self.helper.opt.lr, self.helper.opt.lr_sched.steps,
                                                       self.helper.opt.lr_sched.gamma)

        self.create_grad_norm_tracker()
        self.create_datasets()
        self.create_loaders()
        self.create_masked_model()
        self.create_model_interface()
        self.create_optimizer()
        self.create_validate_on_train(self.train_set)


    def create_valid_loader(self, vset: torch.utils.data.Dataset) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(vset, batch_size=self.helper.opt.test_batch_size or
                                                            self.helper.opt.batch_size,
                                   collate_fn=framework.loader.collate.VarLengthCollate(batch_dim=self.batch_dim),
                                   num_workers=1)


    def create_loaders(self):
        self.train_loader = self.create_train_loader(self.train_set, mask = False)
        self.valid_loaders = framework.data_structures.DotDict()
        self.valid_loaders.update({k: self.create_valid_loader(v) for k, v in self.valid_sets.items()})

    def replace_valid_set(self, name: str, vset: torch.utils.data.Dataset):
        self.valid_sets[name] = vset
        self.valid_loaders[name] = self.create_valid_loader(vset)

    def create_train_loader(self, loader: torch.utils.data.Dataset, seed: Optional[int] = None,
                            mask: bool = True) -> torch.utils.data.DataLoader:

        batch_size = (self.helper.opt.mask_batch_size or self.helper.opt.batch_size) if mask else \
                     self.helper.opt.batch_size

        return torch.utils.data.DataLoader(loader, batch_size=batch_size,
                                           sampler=framework.loader.sampler.InfiniteSampler(
                                               loader, seed = seed),
                                           collate_fn=framework.loader.collate.VarLengthCollate(
                                               batch_dim=self.batch_dim),
                                           num_workers=self.TRAIN_NUM_WORKERS, pin_memory=True)

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
        return (not "embedding" in name) and ((not self.helper.opt.bias_no_mask) or ("bias" not in name)) and \
               (not name.endswith("rezero_alpha")) and ((not self.helper.opt.analysis.skip_layernorm) or
               (not re.match(".+_norm[0-9]+_(beta|gamma)$", name)))

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
        if self.helper.opt.optimizer == "adam":
            self.set_optimizer(torch.optim.Adam(self.model.model_parameters.values(), self.helper.opt.lr,
                                                weight_decay=self.helper.opt.wd))
        elif self.helper.opt.optimizer == "sgd":
            self.set_optimizer(torch.optim.SGD(self.model.model_parameters.values(), self.helper.opt.lr,
                                                weight_decay=self.helper.opt.wd, momentum=0.9))
        else:
            assert False, f"Unsupported optimizer: {self.helper.opt.optimizer}"

    def clip_gradients(self):
        if self.helper.opt.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.helper.opt.grad_clip)

    def set_optimizer_lr(self, lr: float):
        framework.utils.set_lr(self.optimizer, lr)
        if self.helper.state.iter % 100 == 0:
            self.helper.summary.log({"lr": lr})

    def set_linear_warmup(self, curr_step: int, n_steps: int, final: float) -> float:
        if curr_step >= n_steps:
            lr = final
        else:
            lr = final / n_steps * (curr_step+1)

        self.set_optimizer_lr(lr)
        return lr

    def set_mask_lr(self, start_step: int) -> float:
        # Linear warmup for mask learning rate
        return self.set_linear_warmup(self.helper.state.iter - start_step,
                                      self.helper.opt.masking.warmup.nsteps,
                                      self.get_mask_lr())

    def set_lr(self):
        self.set_linear_warmup(self.helper.state.iter, self.helper.opt.lr_warmup,
                               self.step_lr_scheduler.get(self.helper.state.iter))

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
            plots["timing/ms_per_iter"] = self.forward_time_meter.get(True)*1000/20
            plots["timing/ms_per_load"] = self.load_time_meter.get(True)*1000/20
            plots["timing/ms_per_plot"] = self.plot_time_meter.get(True)*1000/20
            plots.update({f"grad_norms/{k}": v for k, v in self.plot_grad_norms().items()})

        if self.helper.state.iter % self.helper.opt.test_interval == 0:
            plots.update({f"validation/{k}": v for k, v in self.validate().items()})

        return plots

    def train(self):
        self.loss_average.reset()

        for d in self.train_loader:
            self.load_time_meter.stop()
            if (self.helper.opt.stop_after or 10e10) <= self.helper.state.iter:
                break

            self.train_step_reconfig()
            self.set_lr()
            res = self.train_step(d)

            with self.plot_time_meter:
                self.helper.summary.log(self.plot(res))

            self.load_time_meter.start()

    def track_grad_norms(self):
        self.param_grad_norm.add_dict(self.model.model_parameters)
        if self.model.masking_enabled:
            self.mask_grad_norm.add_dict(self.model.active_masks)

    def train_step(self, data: Dict[str, torch.Tensor]) -> Result:
        with self.forward_time_meter:
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
                res[f"mask_remaining/layers/{k}/remaining_{mask_indices[i+1]}"] = float(m) / max(n[0], 1)

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

    def run_final_iid_validation(self, prefix: str) -> Dict[str, Any]:
        plots = self.validate()

        test, _ = self.validate_on(self.valid_sets.iid, self.valid_loaders.iid)
        if hasattr(test, "confusion"):
            self.export_tensor(f"{prefix}/confusion", test.confusion)

        return {f"{prefix}/{k}": v for k, v in plots.items()}

    def get_half_mask_masked_layer_names(self, masks: Masks) -> List[Set[str]]:
        layers = list(sorted(masks.keys()))
        k = math.ceil(len(layers)/2)
        n_pos = math.factorial(len(layers)) / (math.factorial(k) * math.factorial(len(layers)-k))
        m = min(n_pos, 5)

        res = set()
        while len(res) < m:
            c=np.random.choice(len(layers), k, replace=False)
            c=list(sorted(c))
            res.add(sum([v * (len(layers) ** i) for i,v in enumerate(c)]))

        return [{layers[((r//(len(layers) ** i)) % len(layers))] for i in range(k)} for r in res]

    def do_half_mask_test(self, stage: int, split: Optional[str] = None) ->  Dict[str, Any]:
        split = split or f"split_{stage}"

        masks = self.model.get_masks(stage)

        res = {}
        for i, layers in enumerate(self.get_half_mask_masked_layer_names(masks)):
            temp_masks = Masks({k: masks[k] for k in layers})
            inverse_temp_masks = Masks({k: v for k, v in masks.items() if k not in layers})

            print(f"Half-mask test, stage: {split}, iteration {i}: keeping masks for the following layers "
                  f"({len(layers)} out of {len(masks)}): {layers}")
            print(f"Inverse: masking {len(inverse_temp_masks)} out of {len(masks)}: {set(inverse_temp_masks.keys())}")

            self.model.set_temporary_masks(temp_masks)

            res[f"half_mask_test/normal/{split}/iter_{i}/iid/accuracy"] = \
                self.validate_on(self.valid_sets.iid, self.valid_loaders.iid)[0].accuracy

            self.model.set_temporary_masks(inverse_temp_masks)
            res[f"half_mask_test/inverse/{split}/iter_{i}/iid/accuracy"] = \
                self.validate_on(self.valid_sets.iid, self.valid_loaders.iid)[0].accuracy

        self.model.set_temporary_masks(None)
        return res

    def inv_mask_test_get_exluded(self) -> Set[str]:
        raise NotImplementedError()

    def do_inverse_mask_test(self, stage: int, split: Optional[str] = None) -> Dict[str, Any]:
        split = split or f"split_{stage}"

        m = self.model.get_masks(stage).invert()
        u = {}
        if self.helper.opt.inv_mask_exclude_io == "fullmask":
            print("Inv mask exclude mode: copy full masks")
            ctrl = self.model.get_masks(0)
            excluded = self.inv_mask_test_get_exluded()
            u = {k: ctrl[k] for k in m.keys() if k in excluded}
        elif self.helper.opt.inv_mask_exclude_io == "ones":
            print("Inv mask exclude mode: fill with ones")
            excluded = self.inv_mask_test_get_exluded()
            u = {k: torch.ones_like(v) for k, v in m.items() if k in excluded}

        for k in u.keys():
            assert k in m
        m.update(u)

        print(f"Inv mask test. Masks: {list(m.keys())}, fallback: {list(u.keys())}")
        self.model.set_temporary_masks(m)
        res = self.run_final_iid_validation(f"inverse_mask_test/{split}")
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
        if self.tasks:
            return len(self.tasks)+int(self.ANALYZE_TRAIN_SET)
        else:
            raise NotImplementedError()

    def set_iid_dataset(self, dataset: Optional[torch.utils.data.Dataset]):
        if "iid" in self.valid_sets:
            if dataset is None:
                del self.valid_sets["iid"]
                del self.valid_loaders["iid"]
            elif self.valid_sets.iid is not dataset:
                self.replace_valid_set("iid", dataset)
        elif dataset is not None:
            self.replace_valid_set("iid", dataset)

    def plot_stage_results(self, index: int, name: str) -> Dict[str, Any]:
        return {f"analysis_results/{name}/{k}": v for k, v in self.validate().items()}

    def analysis_stage_finished(self, index: int, name: str):
        self.helper.summary.log(self.plot_stage_results(index, name))
        self.export_masks(index)

    def analysis_periodic_plot(self, index: int, name: str) -> Dict[str, Any]:
        plots = {}

        if self.helper.opt.analysis.plot_masks:
            plots.update({f"analyzer/{name}/{k}": v for k, v in \
                          self.plot_selected_masks([0] if index == 0 else [0, index]).items()})

        if index > 0:
            plots.update({f"analyzer_remaining/{name}/{k}": v for k, v in
                          self.plot_remaining_stat(0, [index]).items()})

        return plots

    def prepare_model_for_analysis(self):
        # The model is not trained anymore. Dropouts are not needed.
        self.model.set_model_to_eval()

    def set_train_set(self, ds: torch.utils.data.Dataset):
        self.train_set = ds
        self.train_loader = self.create_train_loader(self.train_set, mask = False)
        self.create_validate_on_train(self.train_set)

    def set_baseline_mode(self):
        # In case there is only 1 set to analyze, then we know what to do, otherwise it is task-specific
        if len(self.tasks)!=1 or not self.ANALYZE_TRAIN_SET:
            raise NotImplementedError

        self.set_train_set(self.tasks[0].train_set)
        self.set_iid_dataset(self.tasks[0].valid_set)

    def get_mask_lr(self) -> float:
        return self.helper.opt.mask_lr or self.helper.opt.lr

    def set_mask_stage(self, index: int, name: str):
        self.model.set_active(index)
        self.set_optimizer(torch.optim.Adam(self.model.masks[index].parameters(), self.get_mask_lr()))

    def post_train(self):
        assert self.tasks

        self.prepare_model_for_analysis()

        task_list = ([TaskDataset("verify", self.train_set, None)] if self.ANALYZE_TRAIN_SET else []) + self.tasks

        for stage, task in enumerate(task_list):
            if self.helper.opt.analysis.only_verify_masks and stage > 0:
                break

            print(f"Analysis: Training on {task.name}.")
            self.mask_grad_norm.clear()
            self.set_mask_stage(stage, task.name)

            loader = self.create_train_loader(task.train_set, 1234)
            start = self.helper.state.iter

            self.create_validate_on_train(task.train_set)

            if stage >= 1 or not self.ANALYZE_TRAIN_SET:
                self.set_iid_dataset(task.valid_set)

            for d in loader:
                if self.helper.state.iter - start > self.helper.opt.step_per_mask:
                    self.analysis_stage_finished(stage, task.name)
                    break

                lr = self.set_mask_lr(start)
                res = self.train_step(d)

                plots = self.plot(res)
                plots.update({f"analyzer/{task.name}/{k}": v for k, v in plots.items()})

                if self.helper.state.iter % 1000 == 0:
                    plots.update(self.analysis_periodic_plot(stage, task.name))

                if self.helper.state.iter % 10 == 0:
                    plots["mask_lr"] = lr
                self.helper.summary.log(plots)

        self.helper.summary.log(self.plot_remaining_stat(0, range(1, len(self.model.masks))))
