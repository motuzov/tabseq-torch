from collections.abc import Hashable, Iterator

from tabseq_torch.dl.types import ColumnsByType, Seq


class GroupData:
    def __init__(
        self,
        column_seq: dict[Hashable, Seq],
        type_columns: ColumnsByType,
        gid: int,
        target: int | None = None,
    ):
        self.gid = gid
        self._type_columns = type_columns
        self.column_seq: dict[Hashable, Seq] = column_seq
        self._target: int | None = target
        # The group's sequences must be the same size.
        self._sec_len = len(self.column_seq[next(iter(self.column_seq.keys()))])

    def __getitem__(self, column_name: Hashable) -> Seq:
        return self.column_seq[column_name]

    def __iter__(self) -> Iterator[Hashable]:
        return iter(self.column_seq.keys())

    def __str__(self) -> str:
        return (
            f"gid:\n {self.gid} \ntb seqs: \n{self.column_seq} \n target: {self.target}"
        )

    def __len__(self):
        return self._sec_len

    @property
    def target(self):
        return self._target

    @property
    def type_columns(self) -> ColumnsByType:
        return self._type_columns
