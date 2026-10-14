from dataclasses import dataclass

import git
import polars as pl
from tqdm import tqdm

from tabseq_torch.dl import (
    DDChunkReader,
    LazyDataset,
    LzDatasetCatalog,
    TargetsMetadata,
)
from tabseq_torch.dl.encoder import encode_cat

from .data_catalog import DC

_NAME = "alfa"


class RawDataCatalog:
    def __init__(self):
        self._raw = DC.raw / _NAME

    @property
    def raw(self):
        return self._raw

    @property
    def scoring_trx(self):
        return self.raw / "alfa-scoring-trx"

    @property
    def train_tx(self):
        return self.scoring_trx / "train_transactions/train_transactions"

    @property
    def train_target(self):
        return self.scoring_trx / "train_target.csv.gz"


raw_data_catalog = RawDataCatalog()


class Meta:
    def __init__(self):
        self._name = _NAME

    @property
    def name(self):
        return self._name

    @property
    def num_columns(self):
        return ["amnt", "days_before", "transaction_number", "hour_diff"]

    @property
    def cat_columns(self):
        col_filter = self.num_columns + ["__index_level_0__", "app_id"]
        q = pl.scan_parquet(raw_data_catalog.train_tx)
        return [c for c in q.collect_schema() if c not in col_filter]


meta = Meta()


def load():

    giturl = "https://huggingface.co/datasets/dllllb/alfa-scoring-trx"
    print(f"load data from: {giturl}")
    print(f"to: {raw_data_catalog.raw}")
    raw_data_catalog.raw.mkdir(parents=True, exist_ok=True)
    git.Repo.clone_from(
        giturl,
        raw_data_catalog.train_tx,
        depth=1,
    )


dc = LzDatasetCatalog(
    num=raw_data_catalog.train_tx,
    cat=DC.datasets / meta.name / "cat",
    targets=raw_data_catalog.train_target,
)


def prep(setname: str = "allcats"):
    """
    encode categorical columns
    """
    encode_cat(
        data_path=raw_data_catalog.train_tx,
        dc=dc,
        columns=meta.cat_columns,
        groupby="app_id",
        setname=setname,
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
