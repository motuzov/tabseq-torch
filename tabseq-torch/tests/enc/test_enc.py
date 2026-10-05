import pytest
import torch
from tabseq_torch.dl import CatNum, ColumnType, PaddedTabBatch, cat_embedding_params
from tabseq_torch.dl.padded_batch import GroupData
from tabseq_torch.dl.tb_dataset import TbDataset
from tabseq_torch.tab2sec.cat import CatColumnsEncoder
from tabseq_torch.tab2sec.num import NumColumnsEncoder


@pytest.fixture
def dataset() -> TbDataset:
    dataset = TbDataset()

    dataset.update_type_columns(
        column_type=ColumnType.CAT, columns=["name_chars", "vc"]
    )
    dataset.update_type_columns(column_type=ColumnType.NUM, columns=["F", "F_Rank"])
    # set target
    dataset.target_column = "target"
    return dataset


@pytest.fixture
def catnum() -> CatNum:
    return {
        "vc": 2,
        "name_chars": 3,
    }


@pytest.fixture
def padded_batch(dataset) -> PaddedTabBatch:
    g1: GroupData = dataset.create_group_data(
        group_items={
            "name_chars": [3, 2, 1, 1, 3],
            "vc": [1, 2, 1, 2, 2],
            "F": [751, 14729, 6547, 5950, 10292],
            "F_Rank": [32.0, 66.0, 60.0, 58.0, 64.0],
            "target": 14,
        },
        gid=1,
    )
    g2: GroupData = dataset.create_group_data(
        group_items={
            "name_chars": [3, 3, 2, 1, 2],
            "vc": [1, 2, 2, 1, 2],
            "F": [1028, 7511, 6547, 10761, 5567],
            "F_Rank": [38.0, 61.0, 60.0, 65.0, 56.0],
            "target": 2,
        },
        gid=2,
    )

    return PaddedTabBatch([g1, g2])


def test_encoders(
    padded_batch: PaddedTabBatch, dataset: TbDataset, catnum: CatNum, capsys
):
    num_input_dim = len(padded_batch._type_columns[ColumnType.NUM])
    NumColumnsEncoder(num_dim=num_input_dim)
    embedding_dims = dataset.embedding_dims_template(embedding_dim=4)
    cat_enc = CatColumnsEncoder(params=cat_embedding_params(embedding_dims, catnum))
    num_enc = NumColumnsEncoder(len(dataset.get_columns_by_type(ColumnType.NUM)))

    cat_inputs = cat_enc(padded_batch)
    num_imputs = num_enc(padded_batch)
    inmputs = torch.cat([cat_inputs, num_imputs], 2)
    with capsys.disabled():
        print("cat inputs")
        print(cat_inputs)
        print("num inputs")
        print(num_imputs)
        print("rnn inputs")
        print(inmputs)
    assert True
