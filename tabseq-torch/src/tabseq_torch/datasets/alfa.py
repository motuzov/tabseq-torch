from dataclasses import dataclass

import git
import polars as pl

from tabseq_torch.dl import (
    ChunkReader,
    DDChunkReader,
    LazyDataset,
    LzDatasetCatalog,
    TargetsMetadata,
)

from .data_catalog import DC


class Meta:
    def __init__(self):
        self._name = "alfa"

    @property
    def name(self):
        return self._name

    @property
    def num_columns(self):
        return ["amnt", "days_before", "transaction_number", "hour_diff"]

    @property
    def cat_columns(self):
        col_filter = self.num_columns + ["__index_level_0__", "app_id"]
        q = pl.scan_parquet(dc.src)
        return [c for c in q.collect_schema() if c not in col_filter]


meta = Meta()


class RawDataCatalog:
    def __init__(self):
        self._raw = DC.raw / meta.name

    @property
    def raw(self):
        return self._raw

    @property
    def scoring_trx(self):
        return self.raw / "alfa-scoring-trx"

    @property
    def train_tx(self):
        return self.scoring_trx / "train_transactions"

    @property
    def train_target(self):
        return self.scoring_trx / "train_target.csv.gz"


raw_data_catalog = RawDataCatalog()


def load():
    git.Repo.clone_from(
        "https://huggingface.co/datasets/dllllb/alfa-scoring-trx",
        raw_data_catalog.scoring_trx,
        depth=1,
    )


dc = LzDatasetCatalog(
    num=raw_data_catalog.scoring_trx,
    cat=DC.datasets / meta.name / "cat",
    targets=raw_data_catalog.train_target,
)


def create_lzdataset(chunk_reader, loader_name: str) -> LazyDataset:
    num_chunk_per_worker = 10
    lazy_dataset = LazyDataset(
        chunk_reader=DDChunkReader(
            groupby="app_id",
            num_chunk_per_worker=num_chunk_per_worker,
            num_groups=TargetsMetadata.num_groups(DC.targets),
            memlimit_gb=0.2,
        )
    )
    # lazy_dataset.set_catdata(
    #     data_path=dc.cat(encoded_columns_setname),
    #     columns_to_select=cat_columns,
    #     catnum_json=dc.catnum_json(setname=encoded_columns_setname),
    # )
    # lazy_dataset.set_numdata(
    #     data_path=dc.src,
    #     columns_to_select=num_columns,
    # )
    # lazy_dataset.set_targetmeta(dc.targets, target_column="flag")
    # return lazy_dataset
