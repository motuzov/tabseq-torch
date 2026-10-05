from collections import defaultdict
from collections.abc import Hashable
from typing import Annotated

from torch import Tensor, device, long, tensor
from torch.nn.utils.rnn import pad_sequence

from tabseq_torch.dl.group_data import GroupData
from tabseq_torch.dl.tb_dataset import ColumnsByType, ColumnType
from tabseq_torch.dl.types import Seq, Seqs

Batch = Annotated[Tensor, "2D"]


class PaddedTabBatch:
    def __init__(self, batch_of_groups: list[GroupData]):
        # save lengths befor padding
        self._lengths: list[int] = []
        column_sequences = defaultdict(Seqs)
        self._batche_by_column: dict[Hashable, Batch] = defaultdict(Tensor)
        self._type_columns: ColumnsByType = defaultdict(list[str])
        targets: list = []
        self._batch_size = len(batch_of_groups)
        self._columns_by_type = batch_of_groups[0].type_columns
        for group_data in batch_of_groups:
            self._lengths.append(len(group_data))
            targets.append(group_data.target)
            for column in group_data:
                seq: Seq = group_data[column]
                column_sequences[column].append(seq)
        self._targets: Tensor = tensor(targets, dtype=long)

        for column in column_sequences:
            self._batche_by_column[column] = pad_sequence(
                column_sequences[column], batch_first=True, padding_value=0
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


def dummy_collate_fn(samples_from_tb: list[GroupData]) -> list[GroupData]:
    return samples_from_tb


def collate_padded_batch_fn(samples_from_tb: list[GroupData]) -> PaddedTabBatch:
    # batch of padded tensors
    return PaddedTabBatch(samples_from_tb)
