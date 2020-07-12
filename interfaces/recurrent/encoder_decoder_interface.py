import torch
import torch.nn.functional as F
from typing import Dict, Tuple, List
from models.encoder_decoder import add_eos
from dataclasses import dataclass
import random
from ..result import Result
from ..model_interface import ModelInterface

@dataclass
class EncoderDecoderResult(Result):
    outputs: torch.Tensor
    out_lengths: torch.Tensor
    loss: torch.Tensor

    @property
    def batch_size(self) -> int:
        return self.outputs.shape[1]


class EncoderDecoderInterface(ModelInterface):
    def __init__(self, model):
        self.model = model

    def loss(self, outputs: torch.Tensor, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        mask = torch.arange(outputs.shape[0], device=data["out_len"].device).unsqueeze(1) <=\
                data["out_len"].unsqueeze(0)

        ref = add_eos(data["out"], data["out_len"], self.model.model.decoder.eos_token)

        l = F.cross_entropy(outputs.flatten(end_dim=-2), ref.long().flatten(), reduction='none')
        l = l.reshape_as(ref) * mask
        l = l.sum() / outputs.shape[1]
        return l

    def decode_outputs(self, outputs: EncoderDecoderResult) -> Tuple[torch.Tensor, torch.Tensor]:
        return outputs.outputs.argmax(-1), outputs.out_lengths

    def slice_input(self, data: Dict[str, torch.Tensor], n_slices: int) -> List[Dict[str, torch.Tensor]]:
        batch_size = data["in"].shape[1]
        assert batch_size % n_slices == 0

        new_batch = batch_size // n_slices
        return [{k: v.narrow(1 if v.ndim >= 2 else 0, s*new_batch, new_batch) for k, v in data.items()}
                for s in range(n_slices)]

    def join_results(self, results: List[EncoderDecoderResult]) -> EncoderDecoderResult:
        return EncoderDecoderResult(
            torch.cat([r.outputs for r in results], dim=1),
            torch.cat([r.out_lengths for r in results], dim=0),
            sum([r.loss.detach() for r in results])
        )

    def __call__(self, data: Dict[str, torch.Tensor]) -> EncoderDecoderResult:
        teacher_forcing = self.model.training and random.random() < 0.5
        outs, lens = self.model(data["in"], data["in_len"].long(), data["out"], data["out_len"].long(), teacher_forcing)
        loss = self.loss(outs, data)

        return EncoderDecoderResult(outs, lens, loss)

