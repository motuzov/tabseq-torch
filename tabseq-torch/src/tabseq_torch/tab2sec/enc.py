from tabseq_torch.dl import PaddedTabBatch
from torch import nn
import torch
from abc import abstractmethod


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, padded_batch: PaddedTabBatch) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def output_size(self) -> int:
        raise NotImplementedError


class Tab2Seq(nn.Module):
    def __init__(self, encoders: list[Encoder]) -> None:
        super().__init__()
        self.encoders: nn.ModuleList = torch.nn.ModuleList(encoders)

    @property
    def output_size(self) -> int:
        return sum([e.output_size for e in self.encoders])

    def forward(self, padded_batch: PaddedTabBatch) -> torch.Tensor:
        return torch.cat([e(padded_batch) for e in self.encoders], 2)
