import json
import os
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tabseq_torch.dl import LzDatasetCatalog, PdDataset

from .data_catalog import DC

load_dotenv()


@dataclass
class Mata:
    name: str = "names"
    name_chars: str = "name_chars"
    g_col_name: str = "G"
    T: str = "T"
    labels_col_name: str = "labels"


meta = Mata()

_raw_data = DC.raw / meta.name


@dataclass
class RawDataCatalog:
    raw = _raw_data
    data_zip = _raw_data / "data.zip"
    names = _raw_data / "data" / "names"


raw_data_catalog = RawDataCatalog()


def load():
    print("load to:", raw_data_catalog.raw)
    raw_data_catalog.raw.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(
        "https://download.pytorch.org/tutorial/data.zip", raw_data_catalog.data_zip
    )
    with zipfile.ZipFile(raw_data_catalog.data_zip, "r") as zip_ref:
        zip_ref.extractall(raw_data_catalog.raw)
    print("is exist", raw_data_catalog.names, ":", raw_data_catalog.names.exists())


dc = LzDatasetCatalog(
    num=DC.datasets / meta.name / "tb.parquet",
    cat=DC.datasets / meta.name / "cat",
    targets=DC.datasets / f"{meta.labels_col_name}.parquet",
)


def load_labeled_names(
    names_data_path: Path, file_names: list[str]
) -> tuple[list[list[str]], list[str]]:
    labels: list[str] = []
    name_cahrs: list[list[str]] = []
    for fnanme in file_names:
        with open(names_data_path / fnanme, mode="r", encoding="utf-8") as f:
            for line in f.read().strip().split("\n"):
                name_cahrs.append(list(line))
                labels.append(fnanme)
    return name_cahrs, labels


def prep():
    if not raw_data_catalog.names.exist():
        load()

    path = raw_data_catalog.names

    file_names = sorted([f_path.name for f_path in path.glob("*.txt")])

    df = pd.DataFrame(
        dict(
            zip(
                [meta.name_chars, meta.labels_col_name],
                load_labeled_names(names_data_path=path, file_names=file_names),
            )
        )
    )
    print("raw data:")
    print(df.head(3))

    labels_path = dc.targets
    df_labels = df[[meta.labels_col_name]]
    df_labels[meta.g_col_name] = df.index
    df_labels.to_parquet(labels_path, engine="pyarrow")
    print("labels")
    print(df_labels.head())

    df_tb = (
        df.loc[:, df.columns != meta.labels_col_name]
        .explode(meta.name_chars)
        .reset_index()
        .rename(columns={"index": meta.g_col_name})
    )

    print(df_tb.head(3))

    df_tb[meta.T] = df_tb.index

    df_tb[meta.T] = df_tb.groupby(meta.g_col_name)[meta.T].rank(method="dense")
    df_tb["F"] = (
        df_tb[[meta.name_chars]]
        .groupby(meta.name_chars)[meta.name_chars]
        .transform("count")
    )

    print(df_tb.head(3))

    ch_rank = (
        df_tb[[meta.name_chars]]
        .groupby(meta.name_chars)[meta.name_chars]
        .agg("count")
        .sort_values()
        .rank(method="dense")
    )

    df_tb_wiht_num_features = pd.merge(
        df_tb,
        ch_rank.to_frame().rename(columns={meta.name_chars: "F_Rank"}),
        left_on=meta.name_chars,
        right_index=True,
    )

    with open(Path(__file__).resolve().parent / "data" / "vowels.json", "r") as f:
        vowels = json.load(f)

    df_vc = pd.DataFrame({meta.name_chars: vowels.keys(), "vc": vowels.values()})
    df_vc.head(3)
    df_tb_wiht_num_features = pd.merge(
        df_tb_wiht_num_features,
        df_vc,
        left_on=meta.name_chars,
        right_on=meta.name_chars,
    )
    df_tb_wiht_num_features["index"] = df_tb_wiht_num_features.index
    df_tb_wiht_num_features.to_parquet(path=dc.num, engine="pyarrow")


def create_lzdataset(
    encodeed_cat_columns: list[str], encodeed_cat_columns_setname: str
):
    pass
    # pd_dataset = PdDataset(
    #     tbts_path=src_parquet,
    #     tbts_groupby_column=_GROUP,
    #     cat_columns=_CAT_COLUMNS,
    #     num_columns=_NUM_COLUMNS,
    #     targets_path=data_path / f"{filename}-labels.parquet",
    #     target_column=_LABELES_COL,
    # )
