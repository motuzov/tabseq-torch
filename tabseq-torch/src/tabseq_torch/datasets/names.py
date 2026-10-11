import os
import urllib.request
from dataclasses import dataclass

from dotenv import load_dotenv
from tabseq_torch.dl import LzDatasetCatalog

from .data_catalog import DC

load_dotenv()


@dataclass
class Mata:
    name_chars: str = "name_chars"
    g_col_name: str = "G"
    T: str = "T"
    quantity: int = 0  # Default values are supported


labels_col_name = "labels"
_name = "names"


def load():
    urllib.request.urlretrieve(
        "https://download.pytorch.org/tutorial/data.zip", DC.raw / _name
    )


dc = LzDatasetCatalog(
    num=DC.datasets / _name / "tb.parquet",
    cat=DC.datasets / _name / "cat",
    targets=DC.datasets / f"{labels_col_name}.parquet",
)


def prep():
    pass


def create_lzdataset(
    encodeed_cat_columns: list[str], encodeed_cat_columns_setname: str
):
    pass
