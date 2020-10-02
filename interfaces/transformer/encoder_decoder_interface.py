import torch
import torch.nn.functional as F
from typing import Dict, Tuple
from models.encoder_decoder import add_eos
from models.transformer_enc_dec import TransformerResult
from ..model_interface import ModelInterface

from ..recurrent.encoder_decoder_interface import EncoderDecoderResult


class TransformerEncDecInterface(ModelInterface):
    def __init__(self, model: torch.nn.Module):
        self.model = model

    def loss(self, outputs: TransformerResult, ref: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        l = F.cross_entropy(outputs.data.flatten(end_dim=-2), ref.long().flatten(), reduction='none')
        l = l.reshape_as(ref) * mask
        l = l.sum() / outputs.data.shape[1]
        return l

    def decode_outputs(self, outputs: EncoderDecoderResult) -> Tuple[torch.Tensor, torch.Tensor]:
        return outputs.outputs.argmax(-1), outputs.out_lengths

    def __call__(self, data: Dict[str, torch.Tensor]) -> EncoderDecoderResult:
        in_len = data["in_len"].long()
        out_len = data["out_len"].long()
        in_with_eos = add_eos(data["in"], data["in_len"], self.model.model.encoder_eos)
        out_with_eos = add_eos(data["out"], data["out_len"], self.model.model.decoder_sos_eos)
        in_len += 1
        out_len += 1

        res = self.model(in_with_eos.transpose(0,1), in_len, out_with_eos.transpose(0,1),
                         out_len, teacher_forcing=self.model.training, max_len=out_len.max().item())

        res.data = res.data.transpose(0,1)
        len_mask = ~self.model.model.generate_len_mask(out_with_eos.shape[0], out_len).transpose(0,1)

        loss = self.loss(res, out_with_eos, len_mask)
        return EncoderDecoderResult(res.data, res.length, loss)
