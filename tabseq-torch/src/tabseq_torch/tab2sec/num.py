from tabseq_torch.dl import PaddedTabBatch, ColumnType
from tabseq_torch.tab2sec import Encoder
import torch


class NumColumnsEncoder(Encoder):
    def __init__(self, num_dim: int) -> None:
        super().__init__()
        self.num_dim: int = num_dim

    def forward(self, padded_batch: PaddedTabBatch) -> torch.Tensor:
        tensors = [
            padded_batch[col_name].float().unsqueeze(-1)
            for col_name in padded_batch.get_columns_iter(ColumnType.NUM)
        ]
        return torch.cat(tensors, dim=-1)

    @property
    def output_size(self) -> int:
        return self.num_dim
