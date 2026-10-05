import json
import os
import random
import shutil
from copy import copy
from pathlib import Path
from typing import cast

import duckdb as dd
import pandas as pd
import polars as pl
from torch.utils.data import IterableDataset, get_worker_info

from tabseq_torch.dl.padded_batch import GroupData
from tabseq_torch.dl.tb_dataset import INDEX, ColumnsByType, ColumnType, TbDataset


def encode_categorical_cl(
    lf: pl.LazyFrame, cat_col_name: str, is_labels: bool = False
) -> tuple[pl.LazyFrame, dict[int, str]]:
    encoded_col_name = f"e_{cat_col_name}"
    q = lf.with_columns(pl.col(cat_col_name).rank("dense").alias(encoded_col_name))
    if is_labels:
        q = q.with_columns(pl.col(encoded_col_name) - 1)
    # lf_ecoded = q.select(pl.col("*").exclude(cat_col_name))
    lf_ecoded = q.drop(cat_col_name).rename({encoded_col_name: cat_col_name})
    q = q.select([cat_col_name, encoded_col_name])
    cat_code_tb_dict = q.group_by(encoded_col_name).first().collect().to_dict()
    code2cat_dict = dict(
        zip(cat_code_tb_dict[cat_col_name], cat_code_tb_dict[encoded_col_name])
    )
    return lf_ecoded, code2cat_dict


class EncodedDataCatalog:
    def __init__(self, src_tabts_path: Path, labels_col_name: str = ""):
        self._src = src_tabts_path
        self._encoded_data_dir: Path = src_tabts_path.parent / "encoded"
        self._labels_col_name = labels_col_name
        self._labels_file_name = f"{self._labels_col_name}.parquet"

    @property
    def labels_col_name(self) -> str:
        return self._labels_col_name

    @property
    def cat(self):
        return self._encoded_data_dir / "cat.parquet"

    @property
    def labeled(self) -> bool:
        return not self._labels_col_name == ""

    @property
    def labels(self) -> Path:
        return self._encoded_data_dir / self._labels_file_name

    @property
    def cat2code(self) -> Path:
        return self._encoded_data_dir / "cat2code.json"

    @property
    def label2code(self) -> Path:
        return self._encoded_data_dir / "label2code.json"

    @property
    def stats(self) -> Path:
        return self._encoded_data_dir / "stats.json"

    @property
    def meta(self) -> Path:
        return self._encoded_data_dir / "meta.json"

    @property
    def src(self) -> Path:
        return self._src

    @property
    def src_labels(self) -> Path:
        return self._src.parent / self._labels_file_name

    @property
    def data(self) -> Path:
        return self._encoded_data_dir


def _rm_encoded_data(encoded_data_dir):
    if os.path.exists(encoded_data_dir) and os.path.isdir(encoded_data_dir):
        shutil.rmtree(encoded_data_dir)


def encode_cat_data(
    dc: EncodedDataCatalog, cat_columns: list[str], T_column: str, G_column: str
) -> None:
    lf = pl.scan_parquet(dc.src).select(cat_columns + [T_column, G_column, INDEX])
    columns_decoder_dict = {}
    for cat_column in cat_columns:
        # The dict starts from 1 because 0 is used for the padded element
        lf, code2cat_dict = encode_categorical_cl(lf, cat_column)
        columns_decoder_dict[cat_column] = code2cat_dict

    os.makedirs(dc.data, exist_ok=True)
    lf.collect().write_parquet(dc.cat)

    with open(dc.cat2code, mode="w") as outfile:
        json.dump(columns_decoder_dict, outfile, indent=4)

    if dc.labeled:
        lf_labels: pl.LazyFrame = pl.scan_parquet(dc.src_labels)
        lf_labels, code2cat_dict = encode_categorical_cl(
            lf_labels, cat_col_name=dc.labels_col_name, is_labels=True
        )
        lf_labels.collect().write_parquet(dc.labels)
        with open(dc.label2code, mode="w") as outfile:
            json.dump(code2cat_dict, outfile, indent=4)


_groupby = "groupby"
_cat_columns = "cat_columns"
_num_groups_stat = "num_groups"


def data_prep(
    src_parquet: Path,
    cat_columns: list[str],
    group_cl_name: str,
    T_cl_name: str,
    labels_col_name: str,
    overwrite: bool = True,
):
    dc: EncodedDataCatalog = EncodedDataCatalog(
        src_parquet, labels_col_name=labels_col_name
    )

    if overwrite:
        _rm_encoded_data(dc.data)
    elif os.path.exists(dc.data):
        return

    encode_cat_data(
        dc=dc, cat_columns=cat_columns, T_column=T_cl_name, G_column=group_cl_name
    )
    lf = pl.scan_parquet(src_parquet)
    stats = {}
    stats[_num_groups_stat] = lf.select(pl.col(group_cl_name).n_unique()).collect()[
        group_cl_name
    ][0]
    with open(dc.stats, mode="w") as outfile:
        json.dump(stats, outfile, indent=4)

    with open(dc.meta, mode="w") as outfile:
        json.dump({_groupby: group_cl_name, _cat_columns: cat_columns}, outfile)


class ChunkIdGenerator:
    def __init__(self, num_partitions: int):
        self._num_partitions = num_partitions

    def __iter__(self):
        # worker_info = torch.utils.data.get_worker_info()
        worker_info = get_worker_info()
        num_chanks = self._num_partitions
        if worker_info:
            self.num_chanks = num_chanks * worker_info.num_workers
            for i in range(0, self.num_chanks, worker_info.num_workers):  # type: ignore
                worker_chank = i + worker_info.id  # type: ignore
                yield worker_chank
        else:
            for i in range(num_chanks):
                yield i


class LazyDataset(TbDataset, IterableDataset):
    # Dataset backed by grouped Polars LazyFrame
    def __init__(
        self,
        src_parquet: Path,
        num_partitions: int,
        columns_by_type: ColumnsByType,
        label_cl_name="labels",
        timeline_cl_name="T",
    ) -> None:
        self.dc = EncodedDataCatalog(src_parquet, labels_col_name=label_cl_name)
        with open(self.dc.cat2code) as infile:
            cat2code = json.load(infile)
        TbDataset.__init__(
            self, columns_by_type=columns_by_type, columns_decoder=cat2code
        )
        self._timeline_cl_name = timeline_cl_name

        with open(self.dc.meta, mode="r") as infile:
            self.meta = json.load(infile)
        with open(self.dc.stats, mode="r") as infile:
            self.stats = json.load(infile)
        self.label2code = {}
        if self.dc.label2code.exists():
            with open(self.dc.label2code, mode="r") as infile:
                self.label2code = json.load(infile)
        self._num_partitions = num_partitions
        self._label_cl_name = label_cl_name
        self._chunk_control_sum = 0
        self._maintain_batch_order_in_chunks = False
        self._is_valid_chank: bool = False
        self._group_filter_items: list[int] = []
        self._belong_group = True
        self.print_debug_info = False
        # TODO: Validation of n_partitions, cat_columns, etc.

    def to_subset(self, group_filter_items: list[int], belong_group: bool):
        self._group_filter_items = group_filter_items
        self._belong_group = belong_group

    @property
    def groupby(self) -> str:
        return self.meta[_groupby]

    def read_chunk_dd(
        self, path: Path, chunk: int, columns: list | None = None, groupby=True
    ) -> pd.DataFrame:
        rel = dd.read_parquet(
            str(path),
        )
        if columns:
            rel = rel.select(*columns)
        rel = rel.filter(f"hash({self.groupby}) % {self._num_partitions} == {chunk}")
        if self._group_filter_items:
            group_filter_exp = "G IN" if self._belong_group else "G NOT IN"
            group_filter_exp = group_filter_exp + f" {self._group_filter_items}"
            rel = rel.filter(group_filter_exp)
        if groupby:
            columns = [c for c in rel.columns if c != self.groupby]
            df = rel.apply(
                function_name="list",
                function_aggr=f"{','.join(columns)}",
                group_expr=self.groupby,
                projected_columns=self.groupby,
            ).df()
            df.set_index(self.groupby, inplace=True)
            df.columns = columns
        else:
            df = rel.df()
            df.set_index(self.groupby, inplace=True)
        return df

    def __len__(self):
        if not self._group_filter_items:
            return self.stats[_num_groups_stat]
        elif self._belong_group:
            return len(self._group_filter_items)
        else:
            return self.stats[_num_groups_stat] - len(self._group_filter_items)

    def __iter__(self):
        for chunk in ChunkIdGenerator(self._num_partitions):
            df_label_chank: pd.DataFrame = self.read_chunk_dd(
                path=self.dc.labels, chunk=chunk, groupby=False
            )
            dfs = []
            df_chank: pd.DataFrame = self.read_chunk_dd(
                path=self.dc.cat,
                chunk=chunk,
            )
            dfs.append(df_chank)
            self._is_valid_chank = True
            if self.num_columns:
                df_num_chank: pd.DataFrame = self.read_chunk_dd(
                    path=self.dc.src,
                    chunk=chunk,
                    columns=self.num_columns + [self.groupby, INDEX],
                )
                # df_num_chank = df_num_chank.drop(self.groupby, axis=1)
                dfs.append(df_num_chank)
                # TODO: try to concatenate and check the T order
                # rel_chank = rel_chank.join(rel_num_chank, condition=INDEX, how="inner")

            gindeces = df_label_chank.index.to_list()
            gidx: int
            for gidx in random.sample(gindeces, len(gindeces)):
                name: str = self._label_cl_name
                lable = df_label_chank.loc[gidx][name]
                group_items = {}
                for df in dfs:
                    gdf = df.loc[gidx]
                    group_items |= gdf.to_dict()
                yield GroupData(
                    column_items=group_items,
                    columns_by_type=self.columns_by_type,
                    label=cast(int, lable),
                )

            chunk_memory_usage = df_chank.memory_usage().sum()
            if self.print_debug_info:
                print(f"\nchunk memory: {chunk_memory_usage / (1024**2):.2f} MB")

    @property
    def num_classes(self) -> int:
        return len(self.label2code)

    @property
    def num_categories(self) -> dict[str, int]:
        return {
            str(column_name): len(code2cat_dict)
            for column_name, code2cat_dict in self._columns_decoder.items()
        }


class DebugCtx:
    def __init__(self, lazyds: LazyDataset):
        self._lazyds = lazyds

    def __enter__(self):
        self._lazyds.print_debug_info = True

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self._lazyds.print_debug_info = False


class Subset(IterableDataset):
    def __init__(
        self, dataset: LazyDataset, group_filter_items: list[int], belong_group: bool
    ):
        self._dataset = copy(dataset)
        self._dataset.to_subset(group_filter_items, belong_group)

    def __len__(self):
        return len(self._dataset)

    def __iter__(self):
        for group_data in self._dataset:
            yield group_data


def random_split(dataset: LazyDataset, test_size: float = 0.2) -> tuple[Subset, Subset]:
    """
    output: train, test
    """
    group_filter_items = (
        pl.scan_parquet(dataset.dc.labels)
        .select(pl.col(dataset.groupby))
        .collect()[dataset.groupby]
        .sample(fraction=test_size, shuffle=True)
    ).to_list()

    return Subset(dataset, group_filter_items, belong_group=False), Subset(
        dataset, group_filter_items, belong_group=True
    )
