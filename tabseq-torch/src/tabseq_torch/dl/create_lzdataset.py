from pathlib import Path
from typing import Literal

from .chunk_reader import DDChunkReader
from .lazy_dataset import LazyDataset, TargetsMetadata
from .lazy_dataset_catalog import DataCatalog

type ChunkReaderNames = Literal["DDChunkReader"]


def create_lzdataset(
    dc: DataCatalog,
    cat_columns: tuple[str],
    num_columns: tuple[str],
    groupby: str,
    target_column: str,
    chunk_reader: ChunkReaderNames = "DDChunkReader",
    num_chunk_per_worker=10,
    colset_name="all",
) -> DataCatalog:
    match chunk_reader:
        case "DDChunkReader":
            chunk_reader = DDChunkReader(
                groupby=groupby,
                num_chunk_per_worker=num_chunk_per_worker,
                num_groups=TargetsMetadata.num_groups(dc.targets),
                memlimit_gb=0.2,
            )

        case "_":
            raise ValueError("wrong chunk reader")

    lazy_dataset = LazyDataset(chunk_reader=chunk_reader)
    lazy_dataset.set_catdata(
        data_path=dc.cat(colset_name=colset_name),
        columns_to_select=list(cat_columns),
        catnum_json=dc.catnum_json(colset_name=colset_name),
    )
    lazy_dataset.set_numdata(
        data_path=dc.num,
        columns_to_select=list(num_columns),
    )
    lazy_dataset.set_targetmeta(dc.targets, target_column=target_column)
    return lazy_dataset
