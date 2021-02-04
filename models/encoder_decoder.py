import torch
import torch.nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
from layers import LSTM
from layers import Linear


def add_eos(input: torch.Tensor, lengths: torch.Tensor, eos_id: int):
    input = torch.cat((input, torch.zeros_like(input[0:1])), dim=0)
    input.scatter_(0, lengths.unsqueeze(0).long(), value=eos_id)
    return input


def add_sos(input: torch.Tensor, sos_id: Optional[int]) -> torch.Tensor:
    # input shape: [T, B, C]
    return torch.cat((torch.full_like(input[0:1], fill_value=sos_id), input), dim=0)


class Encoder(torch.nn.Module):
    def __init__(self, vocabulary_size: int, hidden_size: int, n_layers: int, embedding_size: int, dropout: float):
        super().__init__()

        self.vocabulary_size = vocabulary_size
        self.dropout = dropout
        self.embedding = torch.nn.Embedding(vocabulary_size+1, embedding_size)
        self.lstm = LSTM(embedding_size, hidden_size, n_layers, dropout=dropout)

    def set_dropout(self, dropout: float):
        self.dropout = dropout
        self.lstm.dropout = dropout

    def __call__(self, inputs: torch.Tensor, lengths: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        inputs = add_eos(inputs, lengths, self.vocabulary_size)
        lengths = lengths + 1

        net = self.embedding(inputs.long())
        net = F.dropout(net, self.dropout, training=self.training)

        ps = torch.nn.utils.rnn.pack_padded_sequence(net, lengths.cpu(), enforce_sorted=False)
        _, state = self.lstm(ps)
        return state


class Decoder(torch.nn.Module):
    def __init__(self, vocabulary_size: int, hidden_size: int, n_layers: int, embedding_size: int,
                 dropout: float, max_out_len: int):
        super().__init__()
        self.dropout = dropout
        self.vocabulary_size = vocabulary_size

        self.embedding = torch.nn.Embedding(vocabulary_size+1, embedding_size)
        self.lstm = LSTM(embedding_size, hidden_size, n_layers, dropout=dropout)
        self.output_projection = Linear(hidden_size, vocabulary_size+1)

        self.max_out_len = max_out_len
        self.sos_token = self.vocabulary_size
        self.eos_token = self.vocabulary_size

    def set_dropout(self, dropout: float):
        self.dropout = dropout
        self.lstm.dropout = dropout

    def teacher_forcing(self, ref_output: torch.Tensor, lengths: torch.Tensor,
                        hidden: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:

        ref_output = add_sos(ref_output, self.sos_token)
        lengths = lengths + 1
        net = F.dropout(self.embedding(ref_output.long()), self.dropout, training=self.training)

        ps = torch.nn.utils.rnn.pack_padded_sequence(net, lengths.cpu(), enforce_sorted=False)
        ps, _ = self.lstm(ps, hidden)
        net, _ = torch.nn.utils.rnn.pad_packed_sequence(ps, total_length=ref_output.shape[0])
        return self.output_projection(net)

    def self_output(self, force_length: Optional[int], hidden: Tuple[torch.Tensor, torch.Tensor]) ->\
            Tuple[torch.Tensor, torch.Tensor]:

        batch_size = hidden[0][0].shape[0]
        curr_token = torch.full([1, batch_size], self.sos_token, dtype=torch.long, device=hidden[0][0].device)
        outs = []

        running = torch.ones_like(curr_token, dtype=torch.bool)
        len = torch.zeros_like(curr_token, dtype=torch.long)

        n_max_steps = force_length if force_length is not None else self.max_out_len+1

        for s in range(n_max_steps):
            net = F.dropout(self.embedding(curr_token), self.dropout)
            net, hidden = self.lstm(net, hidden)
            outs.append(self.output_projection(net))

            curr_token = outs[-1].argmax(-1)

            running = running & (curr_token != self.eos_token)
            len.masked_fill_(running, s+1)

            if force_length is None and not running.any():
                break

        return torch.cat(outs, 0), len.squeeze(0)

    def __call__(self, ref_output: Optional[torch.Tensor], lengths: Optional[torch.Tensor],
                 hidden: Tuple[torch.Tensor, torch.Tensor], teacher_forcing: bool) -> Tuple[torch.Tensor, torch.Tensor]:

        if teacher_forcing:
            assert self.training
            return self.teacher_forcing(ref_output, lengths, hidden), lengths
        else:
            return self.self_output((ref_output.shape[0]+1) if ref_output is not None else None, hidden)


class EncoderDecoder(torch.nn.Module):
    def __init__(self, in_vocabulary_size: int, out_vocabulary_size: int, hidden_size: int, n_layers: int,
                 embedding_size: int, dropout: float, max_out_length: int):

        super().__init__()
        self.dropout = dropout
        self.encoder = Encoder(in_vocabulary_size, hidden_size, n_layers, embedding_size, dropout)
        self.decoder = Decoder(out_vocabulary_size, hidden_size, n_layers, embedding_size, dropout, max_out_length)

    def set_dropout(self, enabled: bool):
        dropout_f = self.dropout if enabled else 0
        self.encoder.set_dropout(dropout_f)
        self.decoder.set_dropout(dropout_f)

    def __call__(self, input: torch.Tensor, in_lengths: torch.Tensor,
                 ref_outputs: Optional[torch.Tensor], out_lengts: Optional[torch.Tensor], teacher_forcing: bool):

        hidden = self.encoder(input, in_lengths)
        return self.decoder(ref_outputs, out_lengts, hidden, teacher_forcing)
