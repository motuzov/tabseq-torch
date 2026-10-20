from tabseq_torch.dl import LazyDataset
from tabseq_torch.dl.create_lzdataset import ChunkReaderNames
from tabseq_torch.dl.create_lzdataset import (
    create_lzdataset as create,
)

from .etl import dc, meta


def create_lzdataset(
    chunk_reader: ChunkReaderNames = "DDChunkReader",
    num_chunk_per_worker=10,
    colset_name="all",
    num_columns=meta.num_columns,
) -> LazyDataset:

    return create(
        dc=dc,
        cat_columns=meta.cat_columns,
        num_columns=num_columns,
        groupby=meta.groupby,
        target_column=meta.target_column,
        chunk_reader=chunk_reader,
        num_chunk_per_worker=num_chunk_per_worker,
        colset_name=colset_name,
    )
