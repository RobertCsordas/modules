import torch
import torch.nn
import torch.nn.functional as F
import math
from layers import Linear
from typing import Optional, Callable, List
from dataclasses import dataclass


@dataclass
class AttentionMask:
    src_length_mask: Optional[torch.Tensor]
    position_mask: Optional[torch.Tensor]


class MultiHeadAttentionBase(torch.nn.Module):
    def __init__(self, state_size: int, n_heads: int, dropout: float=0.1):
        assert state_size % n_heads == 0
        super().__init__()
        self.state_size = state_size
        self.projection_size = state_size // n_heads
        self.n_heads = n_heads
        self.scale = 1.0 / math.sqrt(self.projection_size)

        self.dropout = torch.nn.Dropout(dropout)
        self.multi_head_merge = Linear(n_heads * self.projection_size, state_size, bias=False)

    def _masked_softmax(self, logits: torch.Tensor, mask: Optional[AttentionMask]) -> torch.Tensor:
        if mask is None or (mask.src_length_mask is None and mask.position_mask is None):
            return F.softmax(logits, -1)

        # Output shape: [n_batch * n_heads, n_time_dest, n_time_src]
        bb, n_time_dest, n_time_src = logits.shape

        logits = logits.view(bb//self.n_heads, self.n_heads, n_time_dest, n_time_src)

        if mask.position_mask is not None:
            logits = logits.masked_fill(mask.position_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        if mask.src_length_mask is not None:
            logits = logits.masked_fill(mask.src_length_mask.unsqueeze(1).unsqueeze(1), float("-inf"))

        logits = F.softmax(logits, -1)
        return logits.view(bb, n_time_dest, n_time_src)

    def _attention(self, mask: Optional[torch.Tensor], q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> \
                   torch.Tensor:
        # all inputs should have a shape of [n_batch, n_steps, data_size]
        # Output shape [n_batch * n_heads, n_time_dest, data_size]
        logits = torch.bmm(q, k.transpose(1,2)) * self.scale
        scores = self._masked_softmax(logits, mask)
        scores = self.dropout(scores)
        return torch.bmm(scores, v)

    def transform_data(self, input: torch.Tensor, proj: Callable[[torch.Tensor], torch.Tensor],
                       n_projs: int) -> List[torch.Tensor]:
        # Input shape: [n_batch, n_steps, n_channels]
        # Output: Tuple of n_projs tensors of dimension: [n_batch * n_heads, n_steps, projection_size]
        n_batch, n_steps, _ = input.shape
        transformed = proj(input).view(n_batch, n_steps, self.n_heads, n_projs, self.projection_size). \
            permute(0, 2, 1, 3, 4).contiguous().view(n_batch * self.n_heads, n_steps, n_projs, self.projection_size)
        return transformed.unbind(dim=2)

    def merged_attention(self, n_batch: int, mask: Optional[AttentionMask], q: torch.Tensor, k: torch.Tensor,
                         v: torch.Tensor) -> torch.Tensor:
        _, n_steps, _ = q.shape

        data = self._attention(mask, q, k, v).view(n_batch, self.n_heads, n_steps, -1).permute(0, 2, 1, 3). \
            contiguous().view(n_batch, n_steps, -1)

        return self.multi_head_merge(data)

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.multi_head_merge.weight)


class MultiHeadAttention(MultiHeadAttentionBase):
    def __init__(self, state_size: int, n_heads: int, dropout: float=0.1, input_size: Optional[torch.Tensor]=None):
        super().__init__(state_size, n_heads, dropout)
        self.data_to_kv = Linear(state_size, 2 * n_heads * self.projection_size, bias=False)
        self.data_to_q = Linear(state_size if input_size is None else input_size,
                                                            n_heads * self.projection_size, bias=False)
        self.reset_parameters()

    def forward(self, curr_state: torch.Tensor, attend_to: torch.Tensor, mask: Optional[AttentionMask]):
        # Input and output shape: [n_batch, n_steps, data_size]
        k, v = self.transform_data(attend_to, self.data_to_kv, 2)
        q, = self.transform_data(curr_state, self.data_to_q, 1)

        return self.merged_attention(curr_state.shape[0], mask, q, k, v)

    def reset_parameters(self):
        super().reset_parameters()

        torch.nn.init.xavier_uniform_(self.data_to_q.weight)
        torch.nn.init.xavier_uniform_(self.data_to_kv.weight[:self.data_to_kv.weight.shape[0]//2])
        torch.nn.init.xavier_uniform_(self.data_to_kv.weight[self.data_to_kv.weight.shape[0]//2:])
