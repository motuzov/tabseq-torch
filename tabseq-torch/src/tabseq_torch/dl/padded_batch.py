from collections import defaultdict
from collections.abc import Hashable
from typing import Annotated

from torch import Tensor, device, long, tensor
from torch.nn.utils.rnn import pad_sequence

from tabseq_torch.dl.seqdata import SeqData
from tabseq_torch.dl.tb_dataset import ColumnsByType, ColumnType
from tabseq_torch.dl.types import ColBatch, ColTensor

type Batch = Annotated[Tensor, "2D"]


class PaddedTabBatch:
    def __init__(self, batch_data: list[SeqData]):
        # save lengths befor padding
        self._lengths: list[int] = []
        column_sequences: defaultdict[Hashable, ColBatch] = defaultdict(list)
        self._batche_by_column: dict[Hashable, Batch] = defaultdict(Tensor)
        self._type_columns: ColumnsByType = defaultdict(list[str])
        targets: list = []
        self._batch_size = len(batch_data)
        self._columns_by_type = batch_data[0].type_columns
        for seq_data in batch_data:
            self._lengths.append(len(seq_data))
            targets.append(seq_data.target)
            for column in seq_data:
                colseq: ColTensor = seq_data[column]
                column_sequences[column].append(colseq)
        self._targets: Tensor = tensor(targets, dtype=long)

        for column in column_sequences:
            self._batche_by_column[column] = pad_sequence(
                sequences=column_sequences[column], batch_first=True, padding_value=0
            )

    def __getitem__(self, column: Hashable) -> Tensor:
        return self._batche_by_column[column]

    def __len__(self):
        return self._batch_size

    def get_columns_iter(self, column_type: ColumnType = ColumnType.CAT):
        return iter(self._columns_by_type[column_type])

    @property
    def targets(self) -> Tensor:
        return self._targets

    @property
    def lengths(self) -> list[int]:
        return self._lengths

    def to(self, device: device):
        self._batche_by_column = {
            column: batches.to(device=device)
            for column, batches in self._batche_by_column.items()
        }
        self._targets = self._targets.to(device=device)


def dummy_collate_fn(samples_from_tb: list[SeqData]) -> list[SeqData]:
    return samples_from_tb


def collate_padded_batch_fn(samples_from_tb: list[SeqData]) -> PaddedTabBatch:
    # batch of padded tensors
    return PaddedTabBatch(samples_from_tb)
