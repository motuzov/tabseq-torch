from collections.abc import Sequence
from enum import Enum, auto
from typing import Protocol

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

type Lengths = torch.Tensor | Sequence[int]


def _valid_steps_mask(lengths: torch.Tensor, seq_len: int) -> torch.Tensor:
    """[B, T] bool mask, True for real (non-padded) time steps."""
    steps = torch.arange(seq_len, device=lengths.device)
    return steps[None, :] < lengths[:, None]


def _last_step(
    ho_t: torch.Tensor, lengths: torch.Tensor | None, bidirectional: bool
) -> torch.Tensor:
    """
    Final hidden state of each sequence: the forward direction at its last real
    step, and (if bidirectional) the backward direction at step 0.
    """
    if lengths is None:
        last = ho_t[:, -1]
    else:
        last = ho_t[torch.arange(ho_t.size(0), device=ho_t.device), lengths - 1]
    if bidirectional:
        h_size = ho_t.size(2) // 2
        last = torch.cat((last[:, :h_size], ho_t[:, 0, h_size:]), dim=1)
    return last


class RNNPooling:
    def __init__(self, h_size: int, bidirectional: bool = False):
        self._h_size = h_size
        self._bidirectional = bidirectional

    def __call__(
        self, ho_t: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        raise NotImplementedError()

    @property
    def out_featurs(self) -> int:
        return self._h_size * (2 if self._bidirectional else 1)


class AvgPooling(RNNPooling):
    def __init__(self, h_size: int, bidirectional: bool):
        super().__init__(h_size=h_size, bidirectional=bidirectional)

    @property
    def out_featurs(self) -> int:
        return super().out_featurs * 2

    def __call__(self, ho_t: torch.Tensor, lengths: torch.Tensor | None = None):
        ho_n = _last_step(ho_t, lengths, self._bidirectional)
        if lengths is None:
            avg_ho_t = torch.mean(ho_t, dim=1)
        else:
            mask = _valid_steps_mask(lengths, ho_t.size(1))
            avg_ho_t = (ho_t * mask[..., None]).sum(dim=1) / lengths[:, None]
        out = torch.cat((avg_ho_t, ho_n), dim=1)
        return out


class Hn(RNNPooling):
    def __init__(self, h_size: int, bidirectional: bool):
        super().__init__(h_size=h_size, bidirectional=bidirectional)

    def __call__(self, ho_t: torch.Tensor, lengths: torch.Tensor | None = None):
        return _last_step(ho_t, lengths, self._bidirectional)


class Attention(nn.Module):
    def __init__(self, h_size, bidirectional: bool):
        super().__init__()
        # Linear layer to calculate attention scores
        self._o_size = h_size * (2 if bidirectional else 1)
        self.s = nn.Linear(in_features=self._o_size, out_features=1)
        self._bidirectional = bidirectional
        self.a: torch.Tensor

    def forward(self, ho_t: torch.Tensor, lengths: torch.Tensor | None = None):
        s = self.s(ho_t)
        if lengths is not None:
            mask = _valid_steps_mask(lengths, ho_t.size(1))
            s = s.masked_fill(~mask[..., None], float("-inf"))
        self.a = F.softmax(s, dim=1)
        ctx = torch.sum(self.a * ho_t, dim=1)
        return ctx

    @property
    def out_featurs(self) -> int:
        return self._o_size


class Ctx(Protocol):
    @property
    def out_featurs(self) -> int: ...
    def __call__(
        self, ho_t: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> torch.Tensor: ...


class PoolingType(Enum):
    AVG = auto()
    LAST = auto()


def _create_pooling(type: PoolingType, h_size: int, bidirectional: bool) -> RNNPooling:
    match type:
        case PoolingType.LAST:
            return Hn(h_size=h_size, bidirectional=bidirectional)
        case PoolingType.AVG:
            return AvgPooling(h_size=h_size, bidirectional=bidirectional)


class TabLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        h_size: int,
        num_classes: int,
        pooling_type: PoolingType = PoolingType.AVG,
        attention: bool = False,
        bidirectional: bool = False,
        num_layers=1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=h_size,
            batch_first=True,
            num_layers=num_layers,
            bidirectional=bidirectional,
        )
        self.attention = attention
        self.ctx: Ctx
        if self.attention:
            self.ctx: Ctx = Attention(h_size=h_size, bidirectional=bidirectional)
        else:
            self.ctx: Ctx = _create_pooling(
                pooling_type, h_size=h_size, bidirectional=self.lstm.bidirectional
            )
        self.h2o = nn.Linear(in_features=self.ctx.out_featurs, out_features=num_classes)

    def forward(self, batch: torch.Tensor, lengths: Lengths | None = None):
        """
        batch: [B, T, input_size] padded sequences.
        lengths: real length of each sequence (e.g. ``PaddedTabBatch.lengths``).
            When given, the LSTM runs on packed sequences and pooling/attention
            ignore padded steps, so a sequence's output does not depend on the
            other sequences in the batch. When omitted, every step is treated
            as real.
        """
        if lengths is None:
            ho_t, _ = self.lstm(batch)
            lengths_t = None
        else:
            lengths_cpu = torch.as_tensor(lengths, dtype=torch.long).cpu()
            packed = pack_padded_sequence(
                batch, lengths_cpu, batch_first=True, enforce_sorted=False
            )
            packed_ho_t, _ = self.lstm(packed)
            ho_t, _ = pad_packed_sequence(
                packed_ho_t, batch_first=True, total_length=batch.size(1)
            )
            lengths_t = lengths_cpu.to(ho_t.device)
        h_ctx: torch.Tensor = self.ctx(ho_t, lengths_t)
        o = self.h2o(h_ctx)
        return o
