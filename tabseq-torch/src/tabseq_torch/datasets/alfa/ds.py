from typing import Literal

from tabseq_torch.dl import (
    DDChunkReader,
    LazyDataset,
    TargetsMetadata,
)

from .etl import dc, meta

type ChunkReader = Literal["DDChunkReader"]


def create_lzdataset(
    chunk_reader: ChunkReader = "DDChunkReader",
    num_chunk_per_worker=10,
    colset_name="all",
    num_columns=meta.num_columns,
) -> LazyDataset:
    num_chunk_per_worker = 10
    match chunk_reader:
        case "DDChunkReader":
            chunk_reader = DDChunkReader(
                groupby="app_id",
                num_chunk_per_worker=num_chunk_per_worker,
                num_groups=TargetsMetadata.num_groups(dc.targets),
                memlimit_gb=0.2,
            )

        case "_":
            raise ValueError("wrong chunk reader")

    lazy_dataset = LazyDataset(chunk_reader=chunk_reader)
    lazy_dataset.set_catdata(
        data_path=dc.cat(colset_name=colset_name),
        columns_to_select=meta.cat_columns,
        catnum_json=dc.catnum_json(colset_name=colset_name),
    )
    lazy_dataset.set_numdata(
        data_path=dc.num,
        columns_to_select=num_columns,
    )
    lazy_dataset.set_targetmeta(dc.targets, target_column="flag")
    return lazy_dataset
