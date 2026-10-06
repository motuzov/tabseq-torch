from enum import Enum, auto
from typing import Protocol

import torch
import torch.nn.functional as F
from torch import nn


class RNNPooling:
    def __init__(self, h_size: int, bidirectional: bool = False):
        self._h_size = h_size
        self._bidirectional = bidirectional

    def __call__(self, ho_t: torch.Tensor):
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

    def __call__(self, ho_t: torch.Tensor):
        ho_n = ho_t[:, -1]
        avg_ho_t = torch.mean(ho_t, dim=1)
        out = torch.cat((avg_ho_t, ho_n), dim=1)
        return out


class Hn(RNNPooling):
    def __init__(self, h_size: int, bidirectional: bool):
        super().__init__(h_size=h_size, bidirectional=bidirectional)

    def __call__(self, ho_t: torch.Tensor):
        ho_n = ho_t[:, -1]
        return ho_n


class Attention(nn.Module):
    def __init__(self, h_size, bidirectional: bool):
        super().__init__()
        # Linear layer to calculate attention scores
        self._o_size = h_size * (2 if bidirectional else 1)
        self.s = nn.Linear(in_features=self._o_size, out_features=1)
        self._bidirectional = bidirectional
        self.a: torch.Tensor

    def forward(self, ho_t):
        s = self.s(ho_t)
        self.a = F.softmax(s, dim=1)
        ctx = torch.sum(self.a * ho_t, dim=1)
        return ctx

    @property
    def out_featurs(self) -> int:
        return self._o_size


class Ctx(Protocol):
    @property
    def out_featurs(self) -> int: ...
    def __call__(self, ho_t) -> torch.Tensor: ...


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

    def forward(self, batch: torch.Tensor):
        ho_t, _ = self.lstm(batch)
        h_ctx: torch.Tensor = self.ctx(ho_t)
        o = self.h2o(h_ctx)
        return o
