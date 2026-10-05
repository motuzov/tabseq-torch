import torch
from torch import nn

from tabseq_torch.dl import CatColEmbeddingParams, PaddedTabBatch
from tabseq_torch.tab2sec import Encoder


class CatColumnsEncoder(Encoder):
    def __init__(self, params: dict[str, CatColEmbeddingParams]) -> None:
        super().__init__()
        self.embeddings = nn.ModuleDict(
            {
                col: nn.Embedding(
                    num_embeddings=col_embedings_params.num_embeddings,
                    embedding_dim=col_embedings_params.embedding_dim,
                    padding_idx=0,
                )
                for col, col_embedings_params in params.items()
            }
        )

    def forward(self, padded_batch: PaddedTabBatch) -> torch.Tensor:
        return torch.cat(
            [
                self.embeddings[col_name](padded_batch[col_name])
                for col_name in self.embeddings
            ],
            2,
        )

    @property
    def output_size(self) -> int:
        return sum([e.embedding_dim for e in self.embeddings.values()])
