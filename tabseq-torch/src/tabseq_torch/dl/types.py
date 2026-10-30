from collections.abc import Hashable
from enum import Enum
from typing import Annotated

from numpy.typing import ArrayLike
from torch import Tensor

# 1D Tensor torch.tensor([1, 3, 7])
type ColTensor = Annotated[Tensor, "1D"]
type Seqs = list[ColTensor]


type TbItem = int | float
type ColumnGroupItems = list[TbItem] | ArrayLike
type TbGroupItems = dict[Hashable, ColumnGroupItems]

# Seqs – list of variable length sequences to passes to torch.nn.utils.rnn.pad_sequence
type ColumnSeq = dict[Hashable, ColTensor]
type Seqs = list[ColTensor]
type ColumnSeqs = dict[Hashable, Seqs]

INDEX = "index"


class ColumnType(Enum):
    CAT = 1
    NUM = 2
    TARGET = 3


type ColumnsByType = dict[ColumnType, list[str]]
