# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

- `tabseq-torch/` — the Python package (Poetry build, `src/` layout, Python >=3.13; `pyproject.toml` declares no dependencies, they're installed separately).
- `images/dev/jupyter-docker-stack/` — the dev environment: a CUDA PyTorch Jupyter image (`create.sh` builds and creates the `torch` container, mounting `~/dsml/work` → `/home/jovyan/work` and `~/dsml/data` → `/home/jovyan/data`). `requirements.txt` there is the real dependency list (ruff, duckdb, polars, pyarrow, lightning, torchmetrics, load-dotenv, pytest, optuna, …).
- `integratoin_notebooks/`, `experiments/`, `data/datasets/*.ipynb` — notebooks for data prep, integration checks and training runs.
- `data/for_tests/README.md` — how to fetch the alfa transactions dataset (`git clone --depth 1 https://huggingface.co/datasets/dllllb/alfa-scoring-trx`).

The host has no `python` on PATH; code is run inside the Docker/Jupyter container (or a conda env, per `.vscode/settings.json`).

## Commands

From `tabseq-torch/`:

```bash
python -m pip install -e .
pytest                                   # all tests
pytest tests/dl/test_lazy_frame.py       # one file
pytest tests/dl/test_lazy_frame.py::test_name   # one test
ruff check . && ruff format .            # lint/format (ruff is in the dev image)
```

`tests/dl/tx_data.py` imports a module that no longer exists (`tabseq_torch.dl.tab_dataset`) and is stale. Modules named `integration_tests.py` / `datasets/alfa/tests.py` under `src/` are helper functions called from notebooks, not pytest tests.

`tabseq_torch.datasets.*` reads the `TABSEQ_HOME` env var (via `load_dotenv`) at import time and raises if it's unset; the Docker image sets it to `/home/jovyan/data/tabseq/`.

## Architecture

The pipeline turns a grouped event table (one row per event, one `groupby` column identifying the entity) into padded sequence batches for an LSTM classifier:

1. **Datasets (`dl/`)** — `TbDataset` (`tb_dataset.py`) is the base mixin: it tracks columns by `ColumnType` (CAT / NUM / TARGET, `dl/types.py`) and converts one group's raw items (`TbGroupItems`) into a `GroupTensors` (`gdata.py`) — a dict of per-column 1D tensors for a single entity plus `gid` and `target`. CAT columns become `LongTensor`, NUM become float tensors.
   - `PdDataset` (`gpdf_dataset.py`): map-style, in-memory pandas; encodes categoricals and factorizes targets itself.
   - `LazyDataset` (`lazy_dataset.py`): `IterableDataset` for large parquet. Configured via `set_catdata` / `set_numdata` / `set_targetmeta`, each producing a `*Metadata` object that registers a `ChunkInfo` with the reader. Categorical parquet must be **pre-encoded** (see below).
   - `ChunkReader` / `DDChunkReader` (`chunk_reader.py`): DuckDB-backed reader that partitions groups into chunks, materializes one chunk at a time, and shards chunks across DataLoader workers via `ChunkIdGenerator` (uses `get_worker_info`). Train/test split is done at the reader level (`split`, `random_split`).
   - `lf_dataset.py` is an older, unexported `LazyDataset` implementation; the public one is `lazy_dataset.py` (see `dl/__init__.py`).
2. **Encoding (`dl/encoder.py`)** — `encode_cat` writes encoded parquet plus `cat2code.json` and `catnum.json` (column cardinalities, `CatNum`). Code `0` is reserved for padding everywhere. `lazy_dataset_catalog.DataCatalog` (exported as `LzDatasetCatalog`) defines the on-disk layout: `<cat>/<colset>/tb` for payload, `<cat>/<colset>/meta/{cat2code,catnum}.json`. `cat_embedding_params(catnum, dims)` builds per-column embedding configs.
3. **Batching (`dl/padded_batch.py`)** — `collate_padded_batch_fn` builds a `PaddedTabBatch`: per-column `pad_sequence(batch_first=True, padding_value=0)`, plus `lengths` and `targets`. Access columns by name (`batch["mcc"]`). It has a custom `.to(device)`, which is why Lightning modules override `transfer_batch_to_device`.
4. **Tab → sequence (`tab2sec/`)** — `Encoder` subclasses (`CatColumnsEncoder` = per-column embeddings, `NumColumnsEncoder` = stacked numeric columns) each map a `PaddedTabBatch` to `[B, T, d]`; `Tab2Seq` concatenates them on dim 2 and exposes `output_size`.
5. **Model (`nn/lstm.py`)** — `TabLSTM` with `PoolingType` (LAST / AVG), optional attention, stacked/bidirectional. `forward(batch, lengths)` packs by `lengths` and masks padded steps in pooling/attention; `lengths=None` treats all steps as real. Lightning modules pass `batch.lengths`.
6. **Training (`lit/`)** — `LitMulticlassTabLSTM` (multiclass, CE loss) and `LitImbBinTabLSTM` (`ib2lstm.py`, imbalanced binary with AUC/PR logging) wire Tab2Seq + TabLSTM.

`datasets/` holds dataset-specific ETL (`alfa` transactions, `names`) built on `datasets/data_catalog.DC` paths under `TABSEQ_HOME`.

## Conventions

- Naming: "group" refers to the table side, "seq" / tensors to the torch side; recent refactors renamed `SeqData` → `GroupTensors`. Keep `tabseq-torch/README.md` examples in sync when public signatures change.
- Uses Python 3.12+ `type X = ...` aliases; public API is re-exported explicitly with `import X as X` in each subpackage `__init__.py`.
- Keep event rows in temporal order in parquet; datasets preserve source row order within each group.
