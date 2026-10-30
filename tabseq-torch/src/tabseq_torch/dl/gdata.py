from collections.abc import Hashable, Iterator

from tabseq_torch.dl.types import ColTensor, ColumnsByType


class GroupTensors:
    """
    GroupTensors represents the group data as a set of tensors;
    each tensor corresponds to the group and column in the input table.
    All tensors belonging to the same group must have the same length(a slice of the table),
    but can be of different types.
    """

    def __init__(
        self,
        column_tensor: dict[Hashable, ColTensor],
        type_columns: ColumnsByType,
        gid: int,
        target: int | None = None,
    ):
        self.gid = gid
        self._type_columns = type_columns
        self.column_tensor: dict[Hashable, ColTensor] = column_tensor
        self._target: int | None = target
        # The group's tensors must be the same size.
        self._sec_len = len(self.column_tensor[next(iter(self.column_tensor.keys()))])

    def __getitem__(self, column_name: Hashable) -> ColTensor:
        return self.column_tensor[column_name]

    def __iter__(self) -> Iterator[Hashable]:
        return iter(self.column_tensor.keys())

    def __str__(self) -> str:
        return f"gid:\n {self.gid} \n input table data: \n{self.column_tensor} \n target: {self.target}"

    def __len__(self):
        return self._sec_len

    @property
    def target(self):
        return self._target

    @property
    def type_columns(self) -> ColumnsByType:
        return self._type_columns
