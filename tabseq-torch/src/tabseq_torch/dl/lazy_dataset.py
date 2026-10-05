import random
from collections import defaultdict
from copy import deepcopy
from functools import cached_property
from pathlib import Path
from typing import cast

import polars as pl
from torch.utils.data import IterableDataset

from tabseq_torch.dl import CatNum
from tabseq_torch.dl.chunk_reader import ChunkReader, DDChunkReader
from tabseq_torch.dl.encoder import read_json
from tabseq_torch.dl.padded_batch import GroupData
from tabseq_torch.dl.tb_dataset import ColumnType, TbDataset
from tabseq_torch.dl.types import TbGroupItems


class TbMetadata:
    def __init__(
        self,
        data_path: Path,
        groupby_column: str,
        columns_to_select: list[str],
        column_type: ColumnType,
        do_groupby: bool = True,
    ):
        self.data_path: Path = data_path
        self._columns_to_select: list[str] = columns_to_select
        self.groupby_column: str = groupby_column
        self._column_type: ColumnType = column_type
        self._do_groupby = do_groupby

    @cached_property
    def schema(self) -> pl.Schema:
        return pl.scan_parquet(self.data_path).collect_schema()

    def validate_groupby(self) -> bool:
        return self.groupby_column in self.schema

    @property
    def chunk_columns_to_select(self) -> list[str]:
        return self._columns_to_select + [self.groupby_column]

    @property
    def column_type(self) -> ColumnType:
        return self._column_type

    @property
    def columns_to_select(self) -> list[str]:
        return self._columns_to_select

    @property
    def do_groupby(self) -> bool:
        return self._do_groupby

    def __len__(self) -> int:
        return (
            pl.scan_parquet(self.data_path)
            .select(pl.col(self.groupby_column).n_unique())
            .collect()
            .item()
        )


def _validation_status_msg(isvalid: bool):
    return "Ok!:" if isvalid else "Ivalid!:"


class CatMetadata(TbMetadata):
    def __init__(
        self,
        data_path: Path,
        groupby_column: str,
        columns_to_select: list[str],
        json_path: Path,
    ):
        self._catnum_json_path: Path = json_path
        super().__init__(
            data_path=data_path,
            groupby_column=groupby_column,
            columns_to_select=columns_to_select,
            column_type=ColumnType.CAT,
        )

    @property
    def catnum_json_path(self) -> Path:
        return self._catnum_json_path

    def validate(self) -> bool:
        catnum: CatNum = read_json(self.catnum_json_path)
        self._encoded_columns: list[str] = list(catnum.keys())
        print("Validation report:")
        validation_result = True
        isvalid = set(self.columns_to_select) <= set(self._encoded_columns)
        validation_result = isvalid
        print(
            f"{_validation_status_msg(isvalid)} 'categorical column filters'={self.columns_to_select} <= 'encoded columns'={self._encoded_columns}"
        )
        data_columns = [a for a in self.schema]
        isvalid = set(self._encoded_columns) <= set(data_columns)
        validation_result &= isvalid
        print(
            f"{_validation_status_msg(isvalid)} 'encoded columns'={self._encoded_columns} <= data schema columns={data_columns}"
        )
        return validation_result


class NumMetadata(TbMetadata):
    def __init__(self, data_path: Path, groupby_column: str, columns: list[str]):
        super().__init__(
            data_path=data_path,
            groupby_column=groupby_column,
            columns_to_select=columns,
            column_type=ColumnType.NUM,
        )


class TargetsMetadata(TbMetadata):
    @staticmethod
    def num_groups(data_path: Path) -> int:
        return pl.scan_parquet(data_path).select(pl.len()).collect().item()

    def __init__(
        self, data_path: Path, colname: str, groupby_column: str, type_: str = "binary"
    ):
        super().__init__(
            data_path=data_path,
            groupby_column=groupby_column,
            columns_to_select=[colname],
            column_type=ColumnType.TARGET,
            do_groupby=False,
        )
        self.type = type_
        self.colname = colname

    def __len__(self) -> int:
        return TargetsMetadata.num_groups(self.data_path)

    @property
    def num_classes(self):
        return (
            pl.scan_parquet(self.data_path)
            .select(pl.col(self.colname).n_unique())
            .collect()
            .item()
        )


class LazyDatasetException(Exception):
    def __init__(self, data_path: Path, message: str = ""):
        self.message = message
        self.data_path = data_path
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message}: {self.data_path}"


class LazyDataset(TbDataset, IterableDataset):
    # Dataset backed by grouped Polars LazyFrame

    def __init__(
        self,
        chunk_reader: ChunkReader,
    ) -> None:
        TbDataset.__init__(
            self,
        )
        self._groupby_column: str = chunk_reader.groupby
        self._chunk_reader: ChunkReader = chunk_reader

        self._groups_to_filter: list[int] = []

        self._len = chunk_reader._num_groups
        self._data_paths: list[Path] = []

        self.metadata: dict[ColumnType, list[TbMetadata]] = defaultdict(
            list[TbMetadata]
        )
        self.targetmeta: TargetsMetadata | None = None

    @property
    def chunk_reader(self) -> ChunkReader:
        return self._chunk_reader

    def set_data_cource(
        self,
        meta: TbMetadata,
    ):
        self.metadata[meta.column_type].append(meta)
        self._data_paths.append(meta.data_path)
        if meta.column_type != ColumnType.TARGET:
            self.update_type_columns(
                column_type=meta.column_type, columns=meta.columns_to_select
            )
        self._chunk_reader.add_chunk_info(
            data_path=meta.data_path,
            columns_to_select=meta.chunk_columns_to_select,
            do_groupby=meta.do_groupby,
        )

    def set_catdata(
        self, data_path: Path, columns_to_select: list[str], catnum_json: Path
    ) -> None:
        meta = CatMetadata(
            data_path=data_path,
            json_path=catnum_json,
            groupby_column=self._groupby_column,
            columns_to_select=columns_to_select,
        )
        self.set_data_cource(meta=meta)

    @property
    def catnum(self) -> CatNum:
        catnum: CatNum = {}
        for cmeta in self.metadata.get(ColumnType.CAT, []):
            cmeta = cast(CatMetadata, cmeta)
            catnum |= {
                cat: num
                for cat, num in read_json(cmeta.catnum_json_path).items()
                if cat in cmeta.columns_to_select
            }
        if catnum:
            return catnum
        raise Exception("catdata is not set, see set_catdata")

    def set_numdata(self, data_path: Path, columns_to_select: list[str]) -> None:
        self.set_data_cource(
            meta=NumMetadata(
                data_path=data_path,
                groupby_column=self._groupby_column,
                columns=columns_to_select,
            )
        )

    def set_targetmeta(self, data_path: Path, target_column: str) -> None:
        self.targetmeta = TargetsMetadata(
            data_path=data_path,
            colname=target_column,
            groupby_column=self._groupby_column,
        )
        self.set_data_cource(meta=self.targetmeta)
        self.target_column = self.targetmeta.colname

    def validate(self) -> None:
        pass

    @cached_property
    def num_groups(self) -> int:
        return self._chunk_reader.num_groups

    def split(self, test_size: float = 0.2, is_test=True) -> None:
        if not self._data_paths:
            raise Exception(
                "The data not set! You must set data first see: set_numdata, set_targetmeta, set_catdata"
            )
        self._len = self.chunk_reader.split(
            test_size=test_size, is_test=is_test, data_path=self._data_paths[0]
        )

    def __len__(self):
        return self._len

    def __iter__(self):
        for chunk_id in self._chunk_reader:
            self._chunk_reader.materialize(chunk_id)
            gids = self._chunk_reader.gids
            for gid in random.sample(gids, len(gids)):
                group_items: TbGroupItems = self._chunk_reader.get_group_items(gid)
                yield self.create_group_data(group_items, gid=gid)

    @property
    def num_classes(self) -> int:
        if self.targetmeta:
            return self.targetmeta.num_classes
        return -1


class Subset(IterableDataset):
    def __init__(self, dataset: LazyDataset, test_size: float, is_test: bool):
        self._dataset: LazyDataset = deepcopy(dataset)
        self._dataset.split(test_size=test_size, is_test=is_test)

    def __len__(self):
        return len(self._dataset)

    def __iter__(self):
        yield from self._dataset

    @property
    def chunk_reader(self) -> ChunkReader:
        return self._dataset._chunk_reader


def random_split(dataset: LazyDataset, test_size: float = 0.2) -> tuple[Subset, Subset]:
    """
    output: train, test
    """

    return Subset(dataset, test_size=test_size, is_test=False), Subset(
        dataset, test_size=test_size, is_test=True
    )


def read_num_grous(data_path: Path, groupby: str) -> int:
    return (
        pl.scan_parquet(data_path).select(pl.col(groupby).n_unique()).collect().item()
    )


def ds_memory_usage(
    lazy_dataset: LazyDataset,
    num_chunk_per_worker: int,
    memlimit_gb: float,
    dryrun: bool = True,
    read_groups=True,
) -> int:
    chunk_reader: DDChunkReader = cast(DDChunkReader, lazy_dataset.chunk_reader)
    chunk_reader._num_chunk_per_worker = num_chunk_per_worker
    chunk_reader._memlimit_gb = memlimit_gb
    n_rows = 0
    for chunk_id in chunk_reader:
        chunk_reader.materialize(chunk_id)
        chunk_reader.print_meme_usage()
        gindeces = lazy_dataset.chunk_reader.gids
        if read_groups:
            for gidx in gindeces:
                gd: GroupData = lazy_dataset.create_group_data(
                    group_items=chunk_reader.get_group_items(gidx),
                    gid=gidx,
                )
                n_rows += len(gd)
        if dryrun:
            break
    print(f"num rows: {n_rows}")
    return n_rows


def lazy_dataset_test(
    lazy_dataset: LazyDataset,
    chunk_dd_memlimit_gb: float,
    num_chunks: int,
    dryrun=False,
) -> int:
    """
    return row count
    """
    chunk_reader: DDChunkReader = cast(DDChunkReader, lazy_dataset.chunk_reader)
    chunk_reader._memlimit_gb = chunk_dd_memlimit_gb
    chunk_reader._num_chunks = num_chunks
    rows = 0
    groups = 0
    gd: GroupData
    for gd in lazy_dataset:
        if dryrun:
            print(f"random seq len={len(gd)}")
            chunk_reader.print_meme_usage()
            break
        groups += 1
        rows += len(gd)
    if not dryrun:
        print(f"row count={rows}")
        print(f"group count={groups}")
    return rows
