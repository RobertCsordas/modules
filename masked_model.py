import torch
from typing import Dict, Tuple, Any, Union, Optional, Callable
from dataclasses import dataclass
from layers import MaskedModule
from framework.layers import gumbel_sigmoid

@dataclass
class ParameterPointer:
    parent: torch.nn.Module
    name: str
    multimask_support: bool

    def set(self, data: torch.Tensor):
        self.parent.__dict__[self.name] = data


def append_update(target: Dict[str, Any], src: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    target.update({f"{prefix}_{k}": v for k, v in src.items()})


class Masks(dict):
    def invert(self, filter = lambda k: True):
        return Masks({k: ~v if filter(k) else v for k, v in self.items()})

    def __or__(self, other):
        res = Masks()
        res.update({k: torch.logical_or(other[k], v) if k in other else v for k, v in self.items()})
        res.update({k: v for k, v in other.items() if k not in self})
        return res

    def __and__(self, other):
        return Masks({k: torch.logical_and(other[k], v) for k, v in self.items() if k in other})


class MaskedModel(torch.nn.Module):
    def update_rnn_params(self, module: torch.nn.Module):
        if isinstance(module, (torch.nn.LSTM, torch.nn.GRU)):
            module._flat_weights = [getattr(module, weight) for weight in module._flat_weights_names]
            module.flatten_parameters()

        for m in module.children():
            self.update_rnn_params(m)

    def gather_and_remove_params(self, module: torch.nn.Module) -> Tuple[Dict[str, ParameterPointer],
                                                                         Dict[str, torch.nn.Parameter]]:
        res_ptrs, res_params = {}, {}

        multimask_support = isinstance(module, MaskedModule)
        for name, m in module.named_children():
            ptrs, params = self.gather_and_remove_params(m)
            append_update(res_ptrs, ptrs, name)
            append_update(res_params, params, name)

        for name, param in module._parameters.items():
            res_ptrs[name] = ParameterPointer(module, name, multimask_support)
            res_params[name] = param

        module._parameters.clear()

        return res_ptrs, res_params

    def sample_mask(self, mask: torch.Tensor, n_samples: int) -> torch.Tensor:
        if n_samples > 0:
            if n_samples > 1:
                mask = mask.unsqueeze(0).expand(n_samples, *mask.shape)
            return gumbel_sigmoid(mask, hard=True)
        else:
            return (mask >= 0).float()

    def __init__(self, model: torch.nn.Module, n_mask_sets: int, n_mask_samples: int, mask_loss_weight: float,
                 mask_filter: Callable[[str], bool] = lambda x: True, empty_init: float = 1):
        super().__init__()
        self.pointers, params = self.gather_and_remove_params(model)
        self.model_parameters = torch.nn.ParameterDict(params)
        self.masks = torch.nn.ModuleList([torch.nn.ParameterDict({k: torch.nn.Parameter(torch.full_like(v, empty_init))
                                         for k, v in self.model_parameters.items() if mask_filter(k)})
                                         for _ in range(n_mask_sets)])

        self.masked_params = set(self.masks[0].keys())

        self.n_mask_samples = n_mask_samples
        self.mask_loss_weight = mask_loss_weight
        self.active = -1
        self.temporary_masks: Optional[Masks] = None

        print(f"Found module parameters: {list(self.model_parameters.keys())}")
        print(f"Masking is applied to paramteres: {self.masked_params}")

        single_sample_params = [k for k in self.masked_params if not self.pointers[k].multimask_support]

        if single_sample_params and n_mask_samples != 1:
            print("!!!!!!!!!!!!!!!!!!!!!!!! ERROR !!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"The following parameters support only single masks {single_sample_params}.")
            assert False

        self.model = model

    @property
    def masking_enabled(self):
        return self.active >= 0

    @property
    def active_masks(self) -> torch.nn.ParameterDict:
        assert self.masking_enabled
        return self.masks[self.active]

    def get_mask(self, name: str) -> torch.Tensor:
        if self.temporary_masks is not None:
            return self.temporary_masks[name].float()
        else:
            return self.sample_mask(self.active_masks[name], self.n_mask_samples if self.training else 0)

    def update_params(self):
        for name, ptr in self.pointers.items():
            if self.masking_enabled and name in self.masked_params:
                ptr.set(self.model_parameters[name] * self.get_mask(name))
            else:
                ptr.set(self.model_parameters[name])

        self.update_rnn_params(self.model)

    def set_active(self, mask_set: int):
        self.active = mask_set
        self.update_params()

    def __call__(self, *args, **kwargs):
        if self.masking_enabled:
            self.update_params()

        return self.model(*args, **kwargs)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.update_params()
        return self

    def normalize_and_clamp_scales(self, scales: Optional[Dict[str, float]]) -> Dict[str, float]:
        if not scales:
            return {}

        norm = sum(scales.values())/len(scales)
        # maxval = 10
        # clamp = lambda x: max(min(x, maxval), 1/maxval)
        clamp = lambda x: max(x, 0.0001)
        return {k: clamp(v/norm) for k, v in scales.items()}

    def mask_loss(self, scales: Optional[Dict[str, float]]) -> Union[torch.Tensor, float]:
        scales = self.normalize_and_clamp_scales(scales)

        if not self.masking_enabled:
            return 0.0

        res = 0.0
        for n, p in self.active_masks.named_parameters():
            res = res + p.sum() * scales.get(n, 1.0)

        return self.mask_loss_weight * res

    def get_masks(self, channel: int) -> Masks:
        return Masks({k: v > 0 for k, v in self.masks[channel].items()})

    def get_mask_probs(self, channel: int) -> Masks:
        return Masks({k: torch.sigmoid(v.detach()) for k, v in self.masks[channel].items()})

    def get_sampled_masks(self, channel: int, n_samples: int = 1) -> Masks:
        return Masks({k: self.sample_mask(v.detach(), n_samples).bool() for k, v in self.masks[channel].items()})

    def set_temporary_masks(self, temporary_masks: Optional[Masks]):
        self.temporary_masks = temporary_masks
