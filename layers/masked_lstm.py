import torch
import torch.nn
from typing import Tuple, List, Optional, Union
import math
from .batch_ops import batch_matmul, batch_bias_add
from .masked_module import MaskedModule
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence

class LSTMCell(MaskedModule):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = torch.nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = torch.nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        self.bias = torch.nn.Parameter(torch.Tensor(4 * hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        # Init forget gate bias to 1, rest to 0
        self.bias.data.fill_(0)
        self.bias.data[self.hidden_size:2*self.hidden_size] = 1

        # Xavier init for the gates
        std = math.sqrt(1/self.hidden_size) / math.sqrt(2) * torch.nn.init.calculate_gain("sigmoid")
        self.weight_hh.data.normal_(0, std)
        std = math.sqrt(2 / (self.input_size + self.hidden_size)) / math.sqrt(2) * torch.nn.init.calculate_gain("sigmoid")
        self.weight_ih.data.normal_(0, std)

        # Orthogonal init for the state
        torch.nn.init.orthogonal_(self.weight_hh[2*self.hidden_size:3*self.hidden_size].data,
                                  gain=torch.nn.init.calculate_gain("tanh") / math.sqrt(2))

        self.weight_ih[2 * self.hidden_size:3 * self.hidden_size].data *= torch.nn.init.calculate_gain("tanh") / \
                                                                          torch.nn.init.calculate_gain("sigmoid")

    def forward(self, input: torch.Tensor, state:Tuple[torch.Tensor, torch.Tensor]) -> \
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = state
        gates = batch_matmul(input, self.weight_ih.transpose(-1,-2)) + batch_matmul(hx, self.weight_hh.transpose(-1,-2))
        gates = batch_bias_add(gates, self.bias)

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)


class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, batch_first=False, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_first = batch_first
        self.dropout = dropout
        self.timedim = 1 if self.batch_first else 0
        self.cells = torch.nn.ModuleList([LSTMCell(input_size if i==0 else hidden_size, hidden_size) for i in range(n_layers)])

    def do_forward_tensor(self, inputs: List[torch.Tensor], state: List[Tuple[torch.Tensor, torch.Tensor]]) ->\
            Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]], List[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        outputs = torch.jit.annotate(List[torch.Tensor], [])
        all_states = []
        for t in range(len(inputs)):
            states = torch.jit.annotate(List[Tuple[torch.Tensor, torch.Tensor]], [])
            input = inputs[t]

            for i, c in enumerate(self.cells):
                if self.dropout>0 and i!=0:
                    input = F.dropout(input, p=self.dropout, training=self.training)
                input, new_state = c(input, state[i])
                states += [new_state]

            outputs += [input]
            all_states.append(state)
            state = states
        return torch.stack(outputs, dim=self.timedim), state, all_states

    def permute_state(self, state: List[Tuple[torch.Tensor, torch.Tensor]], order: torch.Tensor) -> \
            List[Tuple[torch.Tensor, torch.Tensor]]:

        return [tuple(t.index_select(0, order) for t in s) for s in state]

    def cat_states(self, states: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        return tuple(torch.cat([s[t] for s in states], dim=0) for t in range(2))

    def state_split_batches(self, state: Tuple[torch.Tensor, torch.Tensor], n: int,
                            discarded: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:

        if n == state[0].shape[0]:
            return state

        discarded.append(tuple(s[n:] for s in state))
        return tuple(s[:n] for s in state)

    def do_foward_packed_sequence(self, input: PackedSequence, state: List[Tuple[torch.Tensor, torch.Tensor]]) -> \
            Tuple[PackedSequence, List[Tuple[torch.Tensor, torch.Tensor]]]:

        # DO NOT run the RNN on a shorter, sorted tensor. Masks require IID sampling of the data, and if one would sort
        # ont mask would the long squences and the other the short ones.

        unpacked, seq_len = torch.nn.utils.rnn.pad_packed_sequence(input)
        unpacked, _, all_states = self.do_forward_tensor(unpacked.unbind(self.timedim), state)

        ended_in_step_per_layer = [[] for _ in self.cells]
        batch_sizes = input.batch_sizes.cpu().numpy().tolist() + [0]

        for t, bs in enumerate(batch_sizes):
            if bs == batch_sizes[t-1]:
                continue

            state = all_states[t-1]
            indices = input.sorted_indices[bs:batch_sizes[t-1]]

            for li, l in enumerate(state):
                ended_in_step_per_layer[li].append(tuple(t.index_select(0, indices) for t in l))

        state = self.permute_state([self.cat_states(list(reversed(ll))) for ll in ended_in_step_per_layer],
                                   input.unsorted_indices)
        return torch.nn.utils.rnn.pack_padded_sequence(unpacked, seq_len.cpu(), enforce_sorted=False), state


    def forward(self, input: Union[torch.Tensor, PackedSequence], state:Optional[List[torch.Tensor]]=None) -> \
            Tuple[Union[torch.Tensor, PackedSequence], List[Tuple[torch.Tensor, torch.Tensor]]]:

        if isinstance(input, PackedSequence):
            device = input.data.device
            dtype = input.data.dtype
            batch_size = input.batch_sizes[0].item()
        else:
            device = input.device
            dtype = input.dtype
            batch_size = input.shape[1-self.timedim]

        if state is None:
            state = [(torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype) \
                      for _ in range(2)) for _ in range(self.n_layers)]

        if isinstance(input, PackedSequence):
            return self.do_foward_packed_sequence(input, state)
        else:
            inputs = input.unbind(self.timedim)
            return self.do_forward_tensor(inputs, state)[:2]
