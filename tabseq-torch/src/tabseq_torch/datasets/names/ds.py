from tabseq_torch.dl import LazyDataset, PdDataset
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
        groupby=meta.g_col_name,
        target_column=meta.labels_col_name,
        chunk_reader=chunk_reader,
        num_chunk_per_worker=num_chunk_per_worker,
        colset_name=colset_name,
    )


def create_gpddataset(num_columns=meta.num_columns) -> PdDataset:

    return PdDataset(
        tbts_path=dc.num,
        tbts_groupby_column=meta.g_col_name,
        cat_columns=meta.cat_columns,
        num_columns=num_columns,
        targets_path=dc.targets,
        target_column=meta.labels_col_name,
    )
