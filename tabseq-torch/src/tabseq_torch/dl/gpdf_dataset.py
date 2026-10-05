from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from torch.utils.data import Dataset

from tabseq_torch.dl.group_data import GroupData

from .encoder import CatNum
from .tb_dataset import TbDataset
from .types import ColumnType, TbGroupItems


def encode_categorical_cl(
    tbts_cl: pd.Series, start=0
) -> tuple[pd.Series, dict[int, str], int]:
    """
    :return: codes, cat2code, numcat
    """
    codes: npt.NDArray[np.int32]
    codes, uniqs = tbts_cl.factorize()
    code2cat: dict[int, str] = dict(enumerate(uniqs, start=start))
    return pd.Series(codes, dtype=int) + start, code2cat, pd.Series(codes).nunique()


class PdDataset(TbDataset, Dataset):
    """
    Dataset backed by grouped Pandas df
    the tbts_path data has not encoded, the encoding will be done on on fly
    """

    def __init__(
        self,
        tbts_path: Path,
        tbts_groupby_column: str,
        cat_columns: list[str],
        num_columns: list[str] = [],
        targets_path: Path | None = None,
        target_column="labels",
    ) -> None:
        TbDataset.__init__(self)
        self._src = tbts_path
        self._groupby = tbts_groupby_column
        tbts = pd.read_parquet(self._src, engine="pyarrow")

        df = pd.DataFrame({tbts_groupby_column: tbts[tbts_groupby_column]})
        column_decoder: dict[str, dict[int, str]] = {}

        catnum: CatNum = {}
        for column in cat_columns:
            # The dict starts from 1 because 0 is used for the padded element
            encoded_col, code2cat, catnum[column] = encode_categorical_cl(
                tbts[column], start=1
            )
            column_decoder[column] = code2cat
            df[column] = encoded_col
        self.catnum = catnum
        df[tbts_groupby_column], self._code2group, _ = encode_categorical_cl(
            tbts[tbts_groupby_column], start=0
        )

        self.update_type_columns(ColumnType.CAT, cat_columns)
        if num_columns:
            self.update_type_columns(ColumnType.NUM, num_columns)

        for column in self.get_columns_by_type(ColumnType.NUM):
            df[column] = tbts[column]

        self._loc_columns_filter = ~df.columns.isin([tbts_groupby_column])
        self._grouped_df_tdts: pd.api.typing.DataFrameGroupBy = df.groupby(
            tbts_groupby_column
        )
        self._has_targets = False
        if targets_path:
            self.target_column = target_column
            self._has_targets = True
            self._targets: pd.DataFrame = pd.read_parquet(targets_path)
            encoded_col, code2cat, _ = encode_categorical_cl(
                self._targets[target_column]
            )
            self._targets[target_column] = encoded_col
            self._target_decoder = code2cat

    def __len__(self) -> int:
        return self._grouped_df_tdts.ngroups

    def __getitem__(self, idx) -> GroupData:
        df_g = self._grouped_df_tdts.get_group(idx).loc[:, self._loc_columns_filter]
        group_items: TbGroupItems = df_g.to_dict(orient="list")
        group_items[self.target_column] = self._targets.iloc[idx][self.target_column]
        return self.create_group_data(group_items=group_items, gid=idx)

    @property
    def num_classes(self):
        return len(self._target_decoder)
