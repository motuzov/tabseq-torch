import multiprocessing
from pathlib import Path
from typing import Any
from unittest.mock import patch

import polars as pl
import pytest
from tabseq_torch.dl.chunk_reader import ChunkIdGenerator, DDChunkReader
from tabseq_torch.dl.encoder import Cat2Code, CatNum, Encoder, make_cat2code, write_json
from tabseq_torch.dl.lazy_dataset import (
    CatMetadata,
    LazyDataset,
    random_split,
)
from tabseq_torch.dl.lazy_dataset_catalog import DataCatalog
from tabseq_torch.dl.padded_batch import (
    GroupData,
)


@pytest.fixture
def tx_sample() -> dict[str, list[Any]]:
    return {
        "app_id": [
            323,
            323,
            323,
            323,
            323,
            323,
            526,
            526,
            526,
            526,
            526,
            526,
            668,
            668,
            668,
            668,
            668,
            668,
            3917,
            3917,
            3917,
            3917,
            3917,
            3917,
            35271,
            35271,
            35271,
            35271,
            35271,
            35271,
            36181,
            36181,
            36181,
            36181,
            36181,
            36181,
        ],
        "amnt": [
            0.47203313927248175,
            0.46948264837684905,
            0.47077238320444165,
            0.47077238320444165,
            0.47077238320444165,
            0.46948264837684905,
            0.47311915769548796,
            0.5206298797312711,
            0.5168336566977262,
            0.5139556856183587,
            0.6182525209382966,
            0.5871270548274807,
            0.4065418680066658,
            0.41041065630473633,
            0.4265442702066148,
            0.4265442702066148,
            0.41041065630473633,
            0.32024816944507883,
            0.4779440811646806,
            0.4779440811646806,
            0.4492871471021893,
            0.47203313927248175,
            0.43906011449663346,
            0.43906011449663346,
            0.47565386769027984,
            0.47565386769027984,
            0.4779440811646806,
            0.47565386769027984,
            0.46254790483886155,
            0.3876771200456198,
            0.5200409629519158,
            0.4779440811646806,
            0.5291810391278274,
            0.3429382186967854,
            0.5528479185694344,
            0.5264629150731294,
        ],
        "mcc": [
            2,
            2,
            2,
            2,
            2,
            2,
            53,
            79,
            2,
            79,
            79,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            9,
            9,
            9,
            9,
            9,
            9,
        ],
    }


@pytest.fixture
def tx_target_sample() -> dict[str, list[Any]]:
    return {"app_id": [323, 526, 668, 3917, 35271, 36181], "flag": [0, 0, 0, 1, 1, 1]}


@pytest.fixture
def expected_num_grous(tx_target_sample) -> int:
    return len(tx_target_sample["app_id"])


@pytest.fixture
def catnum(tx_sample) -> CatNum:
    return {"mcc": len(set(tx_sample["mcc"]))}


class TestDataCatalog:
    tx: str = "tx"
    target: str = "target"
    colset_name: str = "mcc"


@pytest.fixture(scope="session")
def tmp_data_dir(tmp_path_factory) -> Path:
    directory = tmp_path_factory.mktemp("lazy_ds_data")
    return Path(directory)


@pytest.fixture
def src_tb_path(tmp_data_dir) -> Path:
    return tmp_data_dir / TestDataCatalog.tx


@pytest.fixture
def dc(tmp_data_dir) -> DataCatalog:
    return DataCatalog(
        cat=tmp_data_dir / "cat",
        targets=tmp_data_dir / TestDataCatalog.target,
    )


@pytest.fixture
def create_test_data(
    tx_sample, tx_target_sample, catnum: CatNum, src_tb_path, dc: DataCatalog
):
    write_json(
        data=catnum, path=dc.catnum_json(colset_name=TestDataCatalog.colset_name)
    )
    # pl.DataFrame(tx_sample).write_parquet(dc.src / "1.parquet", mkdir=True)
    pl.DataFrame(tx_sample).write_parquet(src_tb_path)
    pl.DataFrame(tx_target_sample).write_parquet(dc.targets)
    test_columns = ["mcc"]
    cat2code: Cat2Code = make_cat2code(src_tb_path, test_columns)
    write_json(cat2code, dc.cat2code_json(TestDataCatalog.colset_name))
    encoder = Encoder(
        src_tb_path=src_tb_path,
        dc=dc,
        num_parts=10,
        gcol="app_id",
    )
    encoder.encode_catset(setname=TestDataCatalog.colset_name)


@pytest.fixture
@patch("tabseq_torch.dl.lazy_dataset.read_json")
def catmeta(mock_read_json, catnum: CatNum) -> CatMetadata:
    mock_read_json.return_value = catnum
    return CatMetadata(
        data_path=Path(),
        json_path=Path(),
        groupby_column="app_id",
        columns_to_select=["mcc"],
    )


@pytest.fixture
def dd_chunk_reader(expected_num_grous) -> DDChunkReader:
    chunk_reader = DDChunkReader(
        groupby="app_id", num_chunk_per_worker=1, num_groups=expected_num_grous
    )
    return chunk_reader


@pytest.fixture
def lazy_dataset(dd_chunk_reader) -> LazyDataset:
    ds: LazyDataset = LazyDataset(
        chunk_reader=dd_chunk_reader,
    )
    return ds


def test_num_calasses(create_test_data, dc, lazy_dataset):
    lazy_dataset.set_targetmeta(data_path=dc.targets, target_column="flag")
    assert lazy_dataset.num_classes == 2


def test_lazy_random_split(create_test_data, lazy_dataset, src_tb_path, dc, tx_sample):
    lazy_dataset.set_catdata(
        data_path=src_tb_path, columns_to_select=["mcc"], catnum_json=Path(".")
    )
    train_ds, test_ds = random_split(lazy_dataset, test_size=0.3)
    assert len(train_ds.chunk_reader) + len(test_ds.chunk_reader) == len(lazy_dataset)
    assert len(train_ds) + len(test_ds) == len(lazy_dataset)
    materialaized_len = 0
    materialaized_gnum = 0
    gd: GroupData
    for gd in iter(lazy_dataset):
        materialaized_len += len(gd)
        materialaized_gnum += 1
    assert materialaized_len == len(tx_sample["mcc"])
    assert materialaized_gnum == len(lazy_dataset)


class DummyWorkerInfo:
    id: int
    num_workers: int


@patch("tabseq_torch.dl.chunk_reader.get_worker_info")
def test_chunk_gen(mock_get_worker_info, capsys):
    DummyWorkerInfo.num_workers = 3
    mock_get_worker_info.return_value = DummyWorkerInfo
    num_chunks = 4
    # cg = ChunkIdGenerator(num_partitions)
    # assert cg.num_chunks == num_partitions * DummyWorkerInfo.num_workers
    chunk_inds = set()
    with capsys.disabled():
        for worker in range(DummyWorkerInfo.num_workers):
            DummyWorkerInfo.id = worker
            cg = ChunkIdGenerator(num_chunks)
            print(f"worker:= {worker}")
            for chanck, worker_chunck_id in enumerate(cg):
                chunk_inds.add(worker_chunck_id)
                print(f"worker_chunk_id: {worker_chunck_id} in chunk {chanck}")
    assert len(chunk_inds) == DummyWorkerInfo.num_workers * num_chunks


def test_multi_proc_prqtscan(tmp_data_dir, tx_sample, capsys):
    src_parquet_path = Path(tmp_data_dir) / TestDataCatalog.tx
    pl.DataFrame(tx_sample).write_parquet(src_parquet_path)

    def test_sub_process(src_parquet_path, w):
        import duckdb as dk

        # work around the problem  "polars hangs on loading parquet files within processes from multiprocessing.Process"
        # https://github.com/pola-rs/polars/issues/14219
        test_df = dk.read_parquet(
            str(src_parquet_path),
        ).df()
        pf = pl.from_pandas(test_df)
        print(w)
        print(pf)

    proc1 = multiprocessing.Process(target=test_sub_process, args=(src_parquet_path, 1))
    with capsys.disabled():
        proc1.start()
    proc2 = multiprocessing.Process(target=test_sub_process, args=(src_parquet_path, 2))
    with capsys.disabled():
        proc2.start()
    proc1.join()
    proc1.join()
    print("Executed sub process")


def test_lazy_dataset_init(
    create_test_data, lazy_dataset, src_tb_path, dc, tx_sample, tx_target_sample
):
    lazy_dataset.set_numdata(data_path=src_tb_path, columns_to_select=["amnt"])
    lazy_dataset.set_targetmeta(data_path=dc.targets, target_column="flag")
    size = 0
    gd: GroupData
    app_id = tx_target_sample["app_id"][0]
    expected_target = tx_target_sample["flag"][0]
    target = -1
    for gd in iter(lazy_dataset):
        size += len(gd)
        if gd.gid == app_id:
            target = gd.target
    assert expected_target == target
    assert len(tx_sample["mcc"]) == sum(len(gd) for gd in iter(lazy_dataset))
