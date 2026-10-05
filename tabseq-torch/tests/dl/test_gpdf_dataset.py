from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import torch
from tabseq_torch.dl import cat_embedding_params
from tabseq_torch.dl.gpdf_dataset import PdDataset
from tabseq_torch.dl.group_data import GroupData
from tabseq_torch.dl.padded_batch import (
    PaddedTabBatch,
    collate_padded_batch_fn,
    dummy_collate_fn,
)
from tabseq_torch.lit.lstm import LitMulticlassTabLSTM
from torch import LongTensor


@pytest.fixture
def decoded_tbts_pf() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name_chars": [
                "K",
                "h",
                "o",
                "u",
                "r",
                "y",
                "A",
                "n",
                "g",
                "R",
                "o",
                "m",
                "i",
                "j",
                "n",
            ],
            "G": [
                0,
                0,
                0,
                0,
                0,
                0,
                2000,
                2000,
                2000,
                3000,
                3000,
                3000,
                3000,
                3000,
                3000,
            ],
            "vc": [
                "c",
                "c",
                "v",
                "v",
                "c",
                "v",
                "v",
                "c",
                "c",
                "c",
                "v",
                "c",
                "v",
                "c",
                "c",
            ],
            "F": [
                1028,
                6547,
                10761,
                4578,
                7511,
                3185,
                1787,
                9341,
                2167,
                751,
                10761,
                2805,
                10187,
                556,
                9341,
            ],
        }
    )


@pytest.fixture
def decoded_targets_pf() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "G": [0, 2000, 3000],
            "labels": ["Arabic.txt", "Chinese.txt", "Dutch.txt"],
        }
    )


@pytest.fixture
@patch("tabseq_torch.dl.gpdf_dataset.pd.read_parquet")
def pd_dataset(
    mock_read_parquet, decoded_tbts_pf: pd.DataFrame, decoded_targets_pf
) -> PdDataset:
    mock_read_parquet.side_effect = [decoded_tbts_pf, decoded_targets_pf]
    return PdDataset(
        tbts_path=Path(),
        tbts_groupby_column="G",
        cat_columns=["name_chars", "vc"],
        num_columns=["F"],
        targets_path=Path(),
        target_column="labels",
    )


def test_pd_dataset_getitm(pd_dataset: PdDataset):
    dg: GroupData = pd_dataset[0]
    assert dg.target == 0
    assert isinstance(dg["vc"], LongTensor)
    expected_tensor: torch.Tensor = LongTensor([1, 1, 2, 2, 1, 2])
    assert torch.equal(dg["vc"], expected_tensor)
    expected_tensor: torch.Tensor = LongTensor(
        pd_dataset._grouped_df_tdts.get_group(0)["vc"]
    )
    assert dg["vc"].dtype == torch.int64
    assert torch.equal(dg["vc"], expected_tensor)


def test_compatibility_with_dataloader(pd_dataset: PdDataset, capsys):
    torch.manual_seed(0)
    loader = torch.utils.data.DataLoader(
        dataset=pd_dataset, shuffle=True, batch_size=4, collate_fn=dummy_collate_fn
    )
    first_batch_of_sampels: list[GroupData] = next(iter(loader))
    first_sample: GroupData = first_batch_of_sampels[0]
    with capsys.disabled():
        print(f"\nfirst data point in batch:\n{first_sample}\n")
    s = first_sample["name_chars"]
    assert torch.equal(LongTensor([1, 2, 3, 4, 5, 6]), s)
    assert first_sample.target == 0
    assert len(first_sample) == 6


def test_collate_padded_batch_fn(pd_dataset: PdDataset, capsys):
    torch.manual_seed(0)
    loader = torch.utils.data.DataLoader(
        dataset=pd_dataset,
        shuffle=True,
        batch_size=2,
        collate_fn=collate_padded_batch_fn,
    )
    first_batch: PaddedTabBatch = next(iter(loader))
    first_col_name = next(first_batch.get_columns_iter())
    with capsys.disabled():
        print(f"Padded batch of cl={first_col_name}: \n{first_batch[first_col_name]}\n")
        print(f"Labels: {first_batch.targets}")
    assert True


def test_loader_with_rnn(pd_dataset: PdDataset, capsys):
    # TODO: Need to break down into unit tests
    torch.manual_seed(0)
    loader = torch.utils.data.DataLoader(
        dataset=pd_dataset,
        shuffle=True,
        batch_size=4,
        collate_fn=collate_padded_batch_fn,
    )

    embedding_dims = pd_dataset.embedding_dims_template(embedding_dim=4)
    embedding_params = cat_embedding_params(
        catnum=pd_dataset.catnum, col_embedding_dims=embedding_dims
    )
    lit_lstm = LitMulticlassTabLSTM(
        cat_enc_params=embedding_params,
        h_size=8,
        num_classes=pd_dataset.num_classes,
        columns_by_type=pd_dataset.type_columns,
        num_layers=2,
    )
    with capsys.disabled():
        print(lit_lstm.tab2sec.output_size)

    first_batch: PaddedTabBatch = next(iter(loader))
    lit_lstm._forward(first_batch)
