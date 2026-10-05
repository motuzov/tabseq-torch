import json
import os
import shutil
import sys
from collections.abc import Hashable
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from polars._typing import SizeUnit

from tabseq_torch.dl.lazy_dataset_catalog import DataCatalog

CatNum = dict[str, int]


def read_json(path: Path) -> dict:
    with open(path, "r") as json_file:
        return json.load(json_file)


def write_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    print(path)
    with open(path, "w") as json_file:
        json.dump(data, json_file, indent=4)


Cat2Code = dict[str, dict[Hashable, int]]


def encode_targets(src: Path, dst: Path, src_column: str, dst_column=""):
    if not dst_column:
        dst_column = src_column
    pl.scan_parquet(src).with_columns(
        pl.col(src_column).rank("dense").alias(dst_column)
    ).with_columns(pl.col(dst_column) - 1).sink_parquet(dst, mkdir=True)


class Cat2CodeCreator:
    """
    Creates encoder from tabts for categorical columns.
    """

    def __init__(self, tabts_path: Path):
        self._lf_tabts = pl.scan_parquet(tabts_path)

    def __call__(self, cat_columns: list[str]) -> Cat2Code:
        cat2code = {}
        for col_name in cat_columns:
            q = (
                self._lf_tabts.select(col_name)
                .unique()
                .with_columns(pl.col(col_name).rank("dense").alias("rank"))
            )
            pf_cat2code = q.collect()
            print(f"{pf_cat2code.estimated_size(unit='kb')} KB")
            cat2code[col_name] = dict(zip(pf_cat2code[col_name], pf_cat2code["rank"]))
        return cat2code


def make_cat2code(tabts_path: Path, columns: list[str]) -> Cat2Code:
    create_cat2code = Cat2CodeCreator(tabts_path=tabts_path)
    return create_cat2code(columns)


def make_catnum(cat2code: Cat2Code) -> CatNum:
    # num_embeddings=cat_nums + 1, category: Vocabulary Size
    return {cat: len(cat2code) for cat, cat2code in cat2code.items()}


def cat2code_size(cat2code: Cat2Code) -> dict[str, int]:
    enc_size: dict[str, int] = {
        col: int(sys.getsizeof(enc)) for col, enc in cat2code.items()
    }

    print(f"total: {sum(enc_size.values()) / 1024} KB")
    return enc_size


def filter_part(lf: pl.LazyFrame, part_col: str, num_parts: int, part: int):
    return lf.filter(pl.col(part_col).hash() % num_parts == part)


def _rm_if_exist(encoded_data_dir):
    if os.path.exists(encoded_data_dir) and os.path.isdir(encoded_data_dir):
        shutil.rmtree(encoded_data_dir)


class Encoder:
    def __init__(
        self,
        src_tb_path: Path,
        dc: DataCatalog,
        num_parts: int,
        gcol: str,
    ):
        self._src_tb_path = src_tb_path
        self._dc = dc
        self._num_parts: int = num_parts
        self._pf_encoded_part = pl.DataFrame()
        self._gcol = gcol
        self._chunk_size = 0
        self._part_sizes = []
        self._cat2code = {}

    def _get_chunk(self, part: int, cat_column: str = "") -> pl.DataFrame:
        q: pl.LazyFrame = pl.scan_parquet(self._src_tb_path)
        columns = [self._gcol]
        if cat_column:
            columns.append(cat_column)
        q = q.select(columns)
        q = filter_part(lf=q, part_col=self._gcol, num_parts=self._num_parts, part=part)
        if cat_column:
            q = q.select(cat_column)
        return q.collect()

    def estimated_size(self, setname: str, unit: SizeUnit = "gb"):
        self._cat2code: Cat2Code = read_json(self._dc.cat2code_json(setname))
        columns = list(self._cat2code.keys())
        self._chunk_size = self.encode_chunk(
            part=0, cat_column=columns[0]
        ).estimated_size(unit=unit)
        part_size = self._chunk_size * (len(columns) + 1)
        print(f"part size = {part_size} {unit}")

    def encode_chunk(
        self, part: int, cat_column: str, unit: SizeUnit = "gb"
    ) -> pl.DataFrame:
        chunk = self._get_chunk(part=part, cat_column=cat_column).with_columns(
            pl.col(cat_column).replace_strict(
                self._cat2code[cat_column], default=0, return_dtype=pl.Int32
            )
        )
        self._chunk_size = chunk.estimated_size(unit=unit)
        print(f"chunk size: {self._chunk_size} {unit}")
        return chunk

    def encode_part(self, part: int) -> pl.DataFrame:
        encoded_chunks = self._get_chunk(part=part)
        for cat_column in self._cat2code:
            encoded_chunk = self.encode_chunk(part=part, cat_column=cat_column)
            encoded_chunks = pl.concat(
                [encoded_chunks, encoded_chunk], how="horizontal"
            )
            del encoded_chunk
        part_size = encoded_chunks.estimated_size(unit="gb")
        print(f"part size: {part_size} GB")
        self._part_sizes.append(encoded_chunks.estimated_size(unit="gb"))
        return encoded_chunks

    def encode_catset(self, setname: str) -> None:
        cat_path = self._dc.cat(colset_name=setname)
        _rm_if_exist(cat_path)
        self._cat2code: Cat2Code = read_json(self._dc.cat2code_json(setname))
        for part in range(self._num_parts):
            df_part = self.encode_part(part)
            part_path = cat_path / f"part-{part}.parquet"
            print(f"writing: {part_path}")
            df_part.write_parquet(part_path, mkdir=True)
            del df_part


def encode_cat(
    data_path: Path,
    dc: DataCatalog,
    columns: list[str],
    groupby: str,
    setname: str,
):
    cat2code: Cat2Code = make_cat2code(data_path, columns)
    write_json(data=cat2code, path=dc.cat2code_json(colset_name=setname))
    catnum: CatNum = make_catnum(cat2code)
    write_json(data=catnum, path=dc.catnum_json(colset_name=setname))
    encoder = Encoder(
        dc=dc,
        num_parts=10,
        gcol=groupby,
    )
    encoder.estimated_size(setname=setname, unit="gb")
    encoder.encode_catset(setname=setname)


@dataclass(frozen=True)
class CatColEmbeddingParams:
    num_embeddings: int
    embedding_dim: int


def cat_embedding_params(
    catnum: CatNum,
    col_embedding_dims,
) -> dict[str, CatColEmbeddingParams]:
    return {
        str(column_name): CatColEmbeddingParams(
            # The dict keys start from 1, and 0 is a default value for padded elements, so:
            num_embeddings=cat_nums + 1,
            embedding_dim=col_embedding_dims[column_name],
        )
        for column_name, cat_nums in catnum.items()
        if column_name in col_embedding_dims
    }
