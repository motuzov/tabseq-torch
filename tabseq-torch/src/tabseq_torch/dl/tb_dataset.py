from typing import cast

from torch import LongTensor, tensor

from tabseq_torch.dl.group_data import GroupData
from tabseq_torch.dl.types import (
    ColumnGroupItems,
    ColumnsByType,
    ColumnType,
    Seq,
    TbGroupItems,
)


class TbDataset:
    def __init__(self) -> None:
        self._column2type: dict[str, ColumnType] = {}
        self._type_columns: ColumnsByType = {}
        self.target_column = ""

    @property
    def num_classes(self) -> int:
        return 0

    @property
    def num_columns(self) -> list[str]:
        return self._type_columns.get(ColumnType.NUM, [])

    @property
    def cat_columns(self) -> list[str]:
        return self._type_columns.get(ColumnType.CAT, [])

    def get_columns_by_type(self, column_type: ColumnType) -> list[str]:
        return self._type_columns.get(column_type, [])

    def embedding_dims_template(self, embedding_dim: int) -> dict:
        return dict.fromkeys(
            self.get_columns_by_type(column_type=ColumnType.CAT), embedding_dim
        )

    @property
    def type_columns(self) -> ColumnsByType:
        return self._type_columns

    def update_type_columns(self, column_type: ColumnType, columns: list[str]):
        self._type_columns[column_type] = columns
        for column_type, columns in self.type_columns.items():
            self._column2type |= zip(columns, [column_type] * len(columns))

    def _items2seq(self, col_items: ColumnGroupItems, column: str) -> Seq:
        column_type: ColumnType = self._column2type[column]
        if column_type == ColumnType.CAT:
            return LongTensor(col_items)
        elif column_type == ColumnType.NUM:
            return tensor(col_items)

    def create_group_data(self, group_items: TbGroupItems, gid) -> GroupData:
        target = None
        if self.target_column:
            target = group_items[self.target_column]
            # del group_items[self.target_column]
        return GroupData(
            column_seq={
                column: self._items2seq(group_items[column], cast(str, column))
                for column in self._column2type
            },
            type_columns=self._type_columns,
            target=target,
            gid=gid,
        )
