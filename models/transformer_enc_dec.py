import torch
import framework
from layers import Transformer, Linear
import math
from typing import Optional


# Cannot be dataclass, because that won't work with gather
class TransformerResult(framework.data_structures.DotDict):
    data: torch.Tensor
    length: torch.Tensor

    @staticmethod
    def create(data: torch.Tensor, length: torch.Tensor):
        return TransformerResult({"data": data, "length": length})


class TransformerEncDecModel(torch.nn.Module):
    def __init__(self, n_input_tokens: int, n_out_tokens: int, state_size: int = 512, ff_multipiler: float = 4,
                 max_len: int=5000, transformer = Transformer, tied_embedding: bool=False, **kwargs):
        '''
        Transformer encoder-decoder.

        :param n_input_tokens: Number of channels for the input vectors
        :param n_out_tokens: Number of channels for the output vectors
        :param state_size: The size of the internal state of the transformer
        '''
        super().__init__()

        assert (not tied_embedding) or (n_input_tokens == n_out_tokens)

        self.tied_embedding = tied_embedding

        self.decoder_sos_eos = n_out_tokens
        self.encoder_eos = n_input_tokens
        self.state_size = state_size

        self.pos = framework.layers.PositionalEncoding(state_size, max_len=max_len, batch_first=True)
        self.input_embedding = torch.nn.Embedding(n_input_tokens+1, state_size)
        self.output_embedding = self.input_embedding if tied_embedding else \
                                torch.nn.Embedding(n_out_tokens+1, state_size)

        if not tied_embedding:
            self.output_map = Linear(state_size, n_out_tokens+1)
        else:
            self.out_scale = math.sqrt(1/state_size)

        self.register_buffer('int_seq', torch.arange(max_len, dtype=torch.long))

        self.trafo = transformer(d_model=state_size, dim_feedforward=int(ff_multipiler*state_size), **kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        if not self.tied_embedding:
            torch.nn.init.xavier_uniform_(self.output_map.weight)

    def generate_len_mask(self, max_len: int, len: torch.Tensor) -> torch.Tensor:
        return self.int_seq[: max_len] >= len.unsqueeze(-1)

    def map_output(self, res: torch.Tensor) -> torch.Tensor:
        if self.tied_embedding:
            return (res @ self.output_embedding.weight.transpose(0,1)) * self.out_scale
        else:
            return self.output_map(res)

    def run_greedy(self, src: torch.Tensor, src_len: torch.Tensor, max_len: int) -> TransformerResult:
        batch_size = src.shape[0]
        n_steps = src.shape[1]

        in_len_mask = self.generate_len_mask(n_steps, src_len)
        memory = self.trafo.encoder(src, mask=in_len_mask)

        running = torch.ones([batch_size], dtype=torch.bool, device=src.device)
        out_len = torch.zeros_like(running, dtype=torch.long)

        next_tgt = self.pos(self.output_embedding(torch.full([batch_size,1], self.decoder_sos_eos, dtype=torch.long,
                                                   device=src.device)))

        all_outputs = []
        state = self.trafo.decoder.create_state(src.shape[0], max_len, src.device)

        for i in range(max_len):
            output = self.trafo.decoder.one_step_forward(state, next_tgt, memory, memory_key_padding_mask=in_len_mask)

            output = self.map_output(output)
            all_outputs.append(output)

            out_token = torch.argmax(output[:,-1], -1)
            running &= out_token != self.decoder_sos_eos

            out_len[running] = i + 1
            next_tgt = self.pos(self.output_embedding(out_token).unsqueeze(1), offset=i+1)

        return TransformerResult.create(torch.cat(all_outputs, 1), out_len)

    def run_teacher_forcing(self, src: torch.Tensor, src_len: torch.Tensor, target: torch.Tensor,
                            target_len: torch.Tensor) -> TransformerResult:
        target = self.output_embedding(torch.cat((torch.full_like(target[:, :1], self.decoder_sos_eos),
                                                 target[:,:-1]), 1).long())
        target = self.pos(target)

        in_len_mask = self.generate_len_mask(src.shape[1], src_len)

        res = self.trafo(src, target, src_length_mask=in_len_mask,
                          tgt_mask=self.trafo.generate_square_subsequent_mask(target.shape[1], src.device))

        return TransformerResult.create(self.map_output(res), target_len)

    def forward(self, src: torch.Tensor, src_len: torch.Tensor, target: torch.Tensor,
                target_len: torch.Tensor, teacher_forcing: bool, max_len: Optional[int] = None) -> TransformerResult:
        '''
        Run transformer encoder-decoder on some input/output pair

        :param src: source tensor. Shape: [N, S], where S in the in sequence length, N is the batch size
        :param src_len: length of source sequences. Shape: [N], N is the batch size
        :param target: target tensor. Shape: [N, S], where T in the in sequence length, N is the batch size
        :param target_len: length of target sequences. Shape: [N], N is the batch size
        :param teacher_forcing: use teacher forcing or greedy decoding
        :param max_len: overwrite autodetected max length. Useful for parallel execution
        :return: prediction of the target tensor. Shape [N, T, C_out]
        '''

        src = self.pos(self.input_embedding(src.long()))

        if teacher_forcing:
            return self.run_teacher_forcing(src, src_len, target, target_len)
        else:
            return self.run_greedy(src, src_len, max_len or target_len.max().item())
