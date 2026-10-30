from collections.abc import Hashable
from enum import Enum
from typing import Annotated

from numpy.typing import ArrayLike
from torch import Tensor

type TbItem = int | float
type ColumnGroupItems = list[TbItem] | ArrayLike
type TbGroupItems = dict[Hashable, ColumnGroupItems]


class ColumnType(Enum):
    CAT = 1
    NUM = 2
    TARGET = 3


type ColumnsByType = dict[ColumnType, list[str]]

# 1D Tensor torch.tensor([1, 3, 7])
type ColTensor = Annotated[Tensor, "1D"]
# List of tensors length sequences to passes to torch.nn.utils.rnn.pad_sequence
type ColBatch = list[ColTensor]


INDEX = "index"
