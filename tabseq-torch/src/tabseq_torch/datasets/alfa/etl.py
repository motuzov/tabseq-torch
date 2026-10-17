import git
import polars as pl
from tabseq_torch.dl import (
    LzDatasetCatalog,
)
from tabseq_torch.dl.encoder import encode_cat

from ..data_catalog import DC


class RawDataCatalog:
    def __init__(self):
        self._raw = DC.raw

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


class Meta:
    @property
    def name(self):
        return raw_data_catalog.scoring_trx.stem

    @property
    def num_columns(self):
        return ["amnt", "days_before", "transaction_number", "hour_diff"]

    @property
    def cat_columns(self):
        col_filter = self.num_columns + ["__index_level_0__", "app_id"]
        q = pl.scan_parquet(raw_data_catalog.train_tx)
        return [c for c in q.collect_schema() if c not in col_filter]

    @property
    def groupby(self) -> str:
        return "app_id"

    @property
    def target_column(self) -> str:
        return "flags"


meta = Meta()


def load():

    giturl = "https://huggingface.co/datasets/dllllb/alfa-scoring-trx"
    print(f"load data from: {giturl}")
    print(f"to: {raw_data_catalog.raw}")
    if raw_data_catalog.scoring_trx.exists():
        return
    raw_data_catalog.raw.mkdir(parents=True, exist_ok=True)
    git.Repo.clone_from(
        giturl,
        raw_data_catalog.scoring_trx,
        depth=1,
    )


dc = LzDatasetCatalog(
    num=raw_data_catalog.train_tx,
    cat=DC.datasets / meta.name / "cat",
    targets=DC.datasets / meta.name / "targets.parquet",
)


def prep(setname: str = "all"):
    """
    encode categorical columns
    """
    if not dc.targets.exists():
        pl.scan_csv(raw_data_catalog.train_target).sink_parquet(dc.targets)
    if not dc.cat(setname).exists():
        encode_cat(
            data_path=raw_data_catalog.train_tx,
            dc=dc,
            columns=meta.cat_columns,
            groupby=meta.groupby,
            setname=setname,
        )
    print("numerical columns:")
    print(",".join(meta.num_columns))
    dc.show()
