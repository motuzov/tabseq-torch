from collections.abc import Hashable
from enum import Enum
from typing import Annotated

from numpy.typing import ArrayLike
from torch import Tensor

# 1D Tensor torch.tensor([1, 3, 7])
Seq = Annotated[Tensor, "1D"]
Seqs = list[Seq]


TbItem = int | float
ColumnGroupItems = list[TbItem] | ArrayLike
TbGroupItems = dict[Hashable, ColumnGroupItems]

# Seqs – list of variable length sequences to passes to torch.nn.utils.rnn.pad_sequence
ColumnSeq = dict[Hashable, Seq]
Seqs = list[Seq]
ColumnSeqs = dict[Hashable, Seqs]

INDEX = "index"


class ColumnType(Enum):
    CAT = 1
    NUM = 2
    TARGET = 3


ColumnsByType = dict[ColumnType, list[str]]
