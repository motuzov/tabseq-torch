from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import duckdb as dd
import pandas as pd
from torch.utils.data import get_worker_info

from tabseq_torch.dl.types import TbGroupItems


class ChunkIdGenerator:
    def __init__(
        self,
        num_chunk_per_worker: int,
    ):
        self._num_chunk_per_worker = num_chunk_per_worker

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info:
            num_chunks = self._num_chunk_per_worker * worker_info.num_workers
            for i in range(0, num_chunks, worker_info.num_workers):  # type: ignore
                worker_chank = worker_info.id + i  # type: ignore
                yield worker_chank
        else:
            for i in range(self._num_chunk_per_worker):
                yield i


@dataclass
class ChunkInfo:
    data_path: Path
    columns_to_select: list[str]
    do_groupby: bool


class ChunkReader:
    # Reads all data chunk by chunk and group by group

    def __init__(
        self,
        groupby: str,
        num_chunk_per_worker: int,
        num_groups: int,
    ):
        self._groupby = groupby
        self._num_chunk_per_worker: int = num_chunk_per_worker
        self._num_groups: int = num_groups

        self._subset_idx_lower_bound = 0
        self._subset_idx_upper_bound = self._num_groups

    @property
    def groupby(self) -> str:
        return self._groupby

    def __len__(self):
        return self._subset_idx_upper_bound - self._subset_idx_lower_bound

    def __iter__(self):
        chunk_indeces = ChunkIdGenerator(self._num_chunk_per_worker)
        yield from chunk_indeces

    @property
    def num_groups(self) -> int:
        return self._num_groups

    @abstractmethod
    def add_chunk_info(
        self,
        data_path: Path,
        columns_to_select: list[str],
        do_groupby: bool = True,
    ) -> None:
        raise NotImplementedError()

    @abstractmethod
    def materialize(self, chunk_id: int) -> None:
        raise NotImplementedError()

    @property
    @abstractmethod
    def gids(self) -> list[int]:
        raise NotImplementedError()

    @abstractmethod
    def get_group_items(self, gidx: int) -> TbGroupItems:
        raise NotImplementedError()

    @abstractmethod
    def split(self, test_size: float, is_test: bool, data_path: Path) -> int:
        """
        Do split int 2 part: test/train and return the len of test
        if is_test=True the reader set to read the test data else the train data
        data_path is the path to data is used to compute the len
        """
        raise NotImplementedError()


class MemUnit(Enum):
    KB = 1024
    MB = 1024**2
    GB = 1024**3


def _group_filter_exp(
    group_ids: list[int],
    groupby: str,
    in_condition=True,
) -> str:
    group_filter_exp = f"{groupby} IN" if in_condition else f"{groupby} NOT IN"
    group_filter_exp = group_filter_exp + f" {group_ids}"
    return group_filter_exp


class DDChunkReader(ChunkReader):
    # ChunkReader backed by duckdb(DD)

    def __init__(
        self,
        groupby: str,
        num_chunk_per_worker: int,
        num_groups: int,
        memlimit_gb: float = 1.5,
    ) -> None:
        super().__init__(
            groupby=groupby,
            num_chunk_per_worker=num_chunk_per_worker,
            num_groups=num_groups,
        )
        self._num_chunks = num_chunk_per_worker
        self._memlimit_gb: float = memlimit_gb
        self._chunk_loading_memory_usage = 0

        self._num_chunk_columns = []
        self._target_chunk_columns = []

        self._chunk_infos: list[ChunkInfo] = []

        self._groups_to_filter: list[int] = []
        self._is_subset = False

        self._chunk_dfs: list[pd.DataFrame] = []
        self.df_targets_chunk: pd.DataFrame = pd.DataFrame()
        self.memunit: MemUnit = MemUnit.MB

    def add_chunk_info(
        self,
        data_path: Path,
        columns_to_select: list[str],
        do_groupby: bool = True,
    ) -> None:
        self._chunk_infos.append(
            ChunkInfo(
                data_path=data_path,
                columns_to_select=columns_to_select,
                do_groupby=do_groupby,
            )
        )

    def chunk_loading_memory_usage(
        self,
        memunit: MemUnit = MemUnit.MB,
    ) -> str:
        return f"{(self._chunk_loading_memory_usage / memunit.value):.1f}{memunit.name}"

    def chunk_memory_usage(
        self,
        memunit: MemUnit = MemUnit.MB,
    ) -> str:
        memory_usage = 0
        df: pd.DataFrame
        for df in self._chunk_dfs:
            memory_usage += df.memory_usage().sum()
        return f"{(memory_usage / memunit.value):.1f}{memunit.name}"

    def select_subset(self, rel: dd.DuckDBPyRelation) -> dd.DuckDBPyRelation:
        return (
            rel.select(f"hash(app_id) % {self.num_groups} as gidx, *")
            .filter(
                f" {self._subset_idx_lower_bound} <= gidx  AND gidx < {self._subset_idx_upper_bound}"
            )
            .select("* exclude (gidx)")
        )

    def _read_chunk(
        self,
        chunk_id: int,
        info: ChunkInfo,
    ) -> pd.DataFrame:

        conn = dd.connect()
        conn.execute("SET enable_progress_bar = false;")
        conn.execute(f"SET memory_limit = '{self._memlimit_gb}GB'")
        rel: dd.DuckDBPyRelation = conn.read_parquet(path_or_buffer=info.data_path)
        rel = rel.select(*info.columns_to_select)
        rel = rel.filter(f"hash({self.groupby}) % {self._num_chunks} == {chunk_id}")

        if self._is_subset:
            rel = self.select_subset(rel)

        if self._groups_to_filter:
            rel = rel.filter(
                _group_filter_exp(self._groups_to_filter, groupby=self.groupby)
            )
        df = rel.df()
        memory_usage = df.memory_usage().sum() + self._memlimit_gb * (1024**3)
        self._chunk_loading_memory_usage = max(
            self._chunk_loading_memory_usage, memory_usage
        )
        if info.do_groupby:
            gdf = df.groupby(self.groupby).agg(list)
            del df
            df = gdf
        else:
            df.set_index(self.groupby, inplace=True)
        conn.close()
        return df

    def materialize(self, chunk_id: int) -> None:
        for d in self._chunk_dfs:
            del d
        self._chunk_dfs.clear()
        chunk_infos: ChunkInfo
        for chunk_infos in self._chunk_infos:
            self._chunk_dfs.append(
                self._read_chunk(chunk_id=chunk_id, info=chunk_infos)
            )

    def print_meme_usage(self):
        print(f"loading: memory usage: {self.chunk_loading_memory_usage()}")
        print(f"chunk memorys: {self.chunk_memory_usage()}")

    @property
    def gids(self) -> list[int]:
        if not self._chunk_dfs:
            return []
        return self._chunk_dfs[0].index.unique().to_list()

    def get_target(
        self,
        gidx: int,
        column: str,
    ):
        return self.df_targets_chunk.loc[gidx][column]

    def get_group_items(self, gidx: int) -> TbGroupItems:
        group_items: TbGroupItems = {}
        df: pd.DataFrame
        for df in self._chunk_dfs:
            gdf = df.loc[gidx]
            group_items |= gdf.to_dict()
        return group_items

    def set_groups_filter(
        self,
        groups_to_filter: list[int],
        belong_group: bool,
    ):
        self._groups_to_filter = groups_to_filter
        self._belong_group = belong_group

    def split(self, test_size: float, is_test: bool, data_path: Path) -> int:
        conn = dd.connect()
        if data_path.is_dir():
            data_path = data_path / "*.parquet"
        if is_test:
            train_size = 1 - test_size
            self._subset_idx_upper_bound -= round(self.__len__() * train_size)
        else:
            self._subset_idx_lower_bound += round(self.__len__() * test_size)
        len_ = (
            self.select_subset(conn.read_parquet(str(data_path)))
            .aggregate("count (distinct app_id) as test_len")
            .fetchall()[0][0]
        )
        self._is_subset = True
        conn.close()
        return len_
