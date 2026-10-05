from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from torch.utils.data import DataLoader

from tabseq_torch.dl import (
    DataCatalog,
    DDChunkReader,
    LazyDataset,
    ds_memory_usage,
    read_num_grous,
)
from tabseq_torch.dl.group_data import GroupData
from tabseq_torch.dl.padded_batch import PaddedTabBatch


def ds_memory_usage_tx(
    tx_dc: DataCatalog,
    num_chunk_per_worker: int,
    memlimit_gb: float,
    cat_columns_to_select: list[str] = ["mcc"],
    num_columns_to_select: list[str] = ["amnt"],
    dryrun: bool = True,
    read_groups=True,
    setname="test",
) -> int:
    num_grous = read_num_grous(tx_dc.targets, groupby="app_id")

    chunk_reader = DDChunkReader(
        groupby="app_id",
        num_chunk_per_worker=num_chunk_per_worker,
        num_groups=num_grous,
        memlimit_gb=memlimit_gb,
    )
    lazy_dataset = LazyDataset(chunk_reader=chunk_reader)
    lazy_dataset.set_catdata(
        data_path=tx_dc.cat(setname),
        columns_to_select=cat_columns_to_select,
        catnum_json=tx_dc.catnum_json(setname=setname),
    )
    lazy_dataset.set_numdata(
        data_path=tx_dc.src, columns_to_select=num_columns_to_select
    )
    lazy_dataset.set_targetmeta(data_path=tx_dc.targets, target_column="flag")
    return ds_memory_usage(
        lazy_dataset=lazy_dataset,
        num_chunk_per_worker=num_chunk_per_worker,
        memlimit_gb=memlimit_gb,
        dryrun=dryrun,
        read_groups=read_groups,
    )


def mcc_loading_test(
    dl: DataLoader,
    mcc: pd.DataFrame,
) -> str:
    ptb: PaddedTabBatch
    for ptb in iter(dl):
        for sidx, mcc_seq in enumerate(ptb["mcc"]):
            gid = ptb._groups[sidx]
            sl = ptb.lengths[sidx]
            if not mcc_seq.sum() == mcc_seq[:sl].sum():
                return f"boken len group={gid}: {sl}"
            ptb.lengths[sidx]
            seq = mcc.loc[gid]["mcc"]
            expected_seq = mcc_seq[:sl].numpy()
            if not np.array_equal(seq, expected_seq):
                return f"boken group gid:{gid}\n{seq}\n{expected_seq}"
    return "Ok!"


def n_rows_expected(data_path: Path) -> int:
    return pl.scan_parquet(data_path).select(pl.len()).collect().item()


def read_column_goups(
    data_path: Path,
    groupby: str,
    cl: str,
) -> pd.DataFrame:
    return (
        pl.scan_parquet(data_path)
        .select(groupby, cl)
        .group_by(groupby)
        .agg(pl.col(cl))
        .collect()
        .to_pandas()
        .set_index(groupby)
    )


def check_column_reading(
    lazy_dataset,
    data_path: Path,
    groupby,
    cl: str,
) -> None:
    expected_names = read_column_goups(data_path=data_path, groupby=groupby, cl=cl)
    gd: GroupData
    for gd in lazy_dataset:
        seq = gd.column_seq[cl]
        expected_names.loc[gd.gid][cl]
        if not np.array_equal(expected_names.loc[gd.gid][cl], seq.numpy()):
            print(gd.gid)
            print(gd.column_seq[cl])
            print(expected_names.loc[gd.gid])
            break
    print("Ok!")
