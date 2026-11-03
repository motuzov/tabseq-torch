# tabseq-torch

`tabseq-torch` turns grouped tabular data into variable-length sequences for
PyTorch models. It is designed for data where each entity—such as a customer,
session, account, or device—has a sequence of tabular events. Categorical
features are embedded, numeric features are concatenated, and the resulting
event vectors are classified with an LSTM.

## What it provides

- `PdDataset` for small-to-medium parquet datasets that fit in memory.
- `LazyDataset` and `DDChunkReader` for reading grouped parquet data in chunks
  with DuckDB, plus `random_split` for a train/test split.
- `GroupTensors`, the per-entity item: one 1D tensor per column, plus the
  group id and target.
- `PaddedTabBatch` and the DataLoader collate function
  `collate_padded_batch_fn` for batches with different sequence lengths.
- `CatColumnsEncoder`, `NumColumnsEncoder`, and `Tab2Seq` to transform a
  padded tabular batch into an LSTM-ready tensor.
- `TabLSTM` and the Lightning modules `LitMulticlassTabLSTM` (multiclass) and
  `LitImbBinTabLSTM` (imbalanced binary classification).
- `encode_cat` and `encode_targets` to prepare disk-backed encoded data for
  `LazyDataset`.

## Installation

This project targets Python 3.13 or newer. `pyproject.toml` does not declare
runtime dependencies yet, so install them alongside the package:

```bash
cd tabseq-torch
python -m pip install -e .
python -m pip install torch lightning torchmetrics tensorboard pandas pyarrow polars duckdb numpy load-dotenv
```

For development and test runs, also install pytest:

```bash
python -m pip install pytest
pytest
```

The CUDA Jupyter image in `images/dev/jupyter-docker-stack/` installs the
same dependencies and is the reference development environment.

## Data model

The input event table is a parquet file. Every row is one event and one column
identifies the entity (group) that owns that event. All events for an entity
form one sequence. A separate target parquet file supplies one label per
entity.

For example, an event table grouped by `customer_id` might look like this:

| customer_id | merchant | channel | amount |
| --- | --- | --- | --- |
| 101 | groceries | card | 12.50 |
| 101 | transit | card | 2.40 |
| 202 | books | web | 18.00 |

Categorical feature values are encoded to positive integers; `0` is reserved
for padding (and, in `encode_cat`, for unknown values). Numeric values are
converted to float tensors. Keep event rows in the desired temporal order
before writing parquet: the datasets preserve source row order within each
group.

## Quick start: in-memory data

`PdDataset` is the simplest route for experimenting with data that fits in
memory. It reads the event and target parquet files, encodes categorical
columns and the target column on the fly, and exposes one `GroupTensors` item
per entity.

Groups are indexed in order of first appearance in the event table, and the
target for group `i` is row `i` of the target table, so write the target rows
in the same order. A target file is currently required.

```python
from pathlib import Path

from torch.utils.data import DataLoader
from tabseq_torch.dl import PdDataset, cat_embedding_params, collate_padded_batch_fn
from tabseq_torch.lit import LitMulticlassTabLSTM

events = Path("data/events.parquet")
targets = Path("data/targets.parquet")

dataset = PdDataset(
    tbts_path=events,
    tbts_groupby_column="customer_id",
    cat_columns=["merchant", "channel"],
    num_columns=["amount"],
    targets_path=targets,
    target_column="label",
)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_padded_batch_fn,
)

# A single embedding width for all categorical columns is a useful baseline.
embedding_dims = dataset.embedding_dims_template(embedding_dim=16)
embedding_params = cat_embedding_params(dataset.catnum, embedding_dims)

model = LitMulticlassTabLSTM(
    cat_enc_params=embedding_params,
    h_size=64,
    num_classes=dataset.num_classes,
    columns_by_type=dataset.type_columns,
    num_layers=1,
    lr=1e-3,
)
```

`LitMulticlassTabLSTM` uses attention pooling by default (`attention=True`)
and AdamW with `weight_decay_rate=0.001`. It always includes a categorical
encoder and adds a numeric encoder when the dataset has numeric columns.

Train the Lightning module with your preferred trainer:

```python
import lightning as L

trainer = L.Trainer(max_epochs=10)
trainer.fit(model, train_dataloaders=loader)
```

Each collated `PaddedTabBatch` holds tensors shaped
`[batch, longest_sequence]` per column:

- `batch[column]` is the padded tensor for that feature (padding value `0`).
- `batch.lengths` is a list of the original sequence lengths.
- `batch.targets` is a `long` tensor with one label per sequence.
- `batch.to(device)` moves all tensors in place; the Lightning modules call it
  from `transfer_batch_to_device`.

## Model building blocks

Use the lower-level components when a Lightning training loop is not needed:

```python
from tabseq_torch.tab2sec import CatColumnsEncoder, NumColumnsEncoder, Tab2Seq
from tabseq_torch.nn import TabLSTM, PoolingType

tab_to_sequence = Tab2Seq(
    [
        CatColumnsEncoder(embedding_params),
        NumColumnsEncoder(num_dim=len(dataset.num_columns)),
    ]
)

classifier = TabLSTM(
    input_size=tab_to_sequence.output_size,
    h_size=64,
    num_classes=dataset.num_classes,
    pooling_type=PoolingType.LAST,
)

batch = next(iter(loader))
logits = classifier(tab_to_sequence(batch), batch.lengths)
```

`CatColumnsEncoder` embeds each column in `embedding_params` (with
`padding_idx=0`) and concatenates the embeddings; `NumColumnsEncoder` stacks
the batch's numeric columns as float features. `Tab2Seq` concatenates the
encoder outputs along the feature dimension, giving
`[batch, longest_sequence, output_size]`.

`TabLSTM` options:

- `pooling_type`: `PoolingType.LAST` (last hidden state) or `PoolingType.AVG`
  (the default; mean of all hidden states concatenated with the last one).
- `attention=True` replaces pooling with a learned attention-weighted sum over
  time steps.
- `num_layers` and `bidirectional` are passed to `torch.nn.LSTM`.

Pass `lengths` (e.g. `batch.lengths`) to `TabLSTM.forward` so padding is ignored:
the LSTM runs on packed sequences, `LAST` takes each sequence's last real step,
and `AVG`/attention only cover real steps. A sequence then gets the same output
regardless of what else is in its batch. The Lightning modules always pass it.
Without `lengths`, every padded step is treated as real. With
`bidirectional=True`, `LAST` uses the backward direction's state at step 0.

## Imbalanced binary classification

`LitImbBinTabLSTM` trains a single-logit `TabLSTM` with `BCEWithLogitsLoss`
(optional `pos_weight`), logs `train_bce_loss`, `train_auc`, `val_bce_loss`,
and `val_auc`, and writes a validation precision-recall curve to TensorBoard
at the end of each validation epoch, so it needs a `TensorBoardLogger`.

```python
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from tabseq_torch.lit import LitImbBinTabLSTM, LogPRMethod
from tabseq_torch.nn import PoolingType

model = LitImbBinTabLSTM(
    cat_enc_params=embedding_params,
    h_size=64,
    columns_by_type=dataset.type_columns,
    bidirectional=False,
    pooling_type=PoolingType.AVG,
    attention=False,
    pos_weight=10.0,
    lr=1e-3,
    num_layers=1,
    log_pr_with=LogPRMethod.add_pr_curve,  # or LogPRMethod.add_figure
)

trainer = L.Trainer(max_epochs=10, logger=TensorBoardLogger("tb_logs"))
```

Targets must be `0`/`1`. The optimizer is Adam by default; with
`opt_type=OptType.ADAMW` (from `tabseq_torch.lit.ib2lstm`) it uses AdamW with
a `ReduceLROnPlateau` scheduler.

## Large parquet datasets

For datasets that should not be held in memory, use `DDChunkReader` with
`LazyDataset`. The reader hash-partitions groups into chunks, materializes only
the current chunk with DuckDB, and shards chunks across DataLoader workers.
Inside a chunk, groups are yielded in random order.

Inputs:

- Categorical data must already be encoded (see `encode_cat` below). Pass the
  encoded payload directory and its `catnum.json`.
- Numeric data can be read straight from the event parquet.
- The target parquet has one row per group with the group column and an
  integer-encoded target column (`encode_targets` can produce it).

```python
from pathlib import Path

from torch.utils.data import DataLoader
from tabseq_torch.dl import (
    DDChunkReader,
    LazyDataset,
    LzDatasetCatalog,
    cat_embedding_params,
    collate_padded_batch_fn,
    random_split,
    read_num_grous,
)

catalog = LzDatasetCatalog(cat=Path("data/encoded"))
targets = Path("data/targets.parquet")

reader = DDChunkReader(
    groupby="customer_id",
    num_chunk_per_worker=8,
    num_groups=read_num_grous(targets, groupby="customer_id"),
    memlimit_gb=1.5,  # DuckDB memory limit per chunk read
)
dataset = LazyDataset(reader)

dataset.set_catdata(
    data_path=catalog.cat("default"),
    columns_to_select=["merchant", "channel"],
    catnum_json=catalog.catnum_json("default"),
)
dataset.set_numdata(
    data_path=Path("data/events.parquet"),
    columns_to_select=["amount"],
)
dataset.set_targetmeta(
    data_path=targets,
    target_column="label",
)

train_ds, test_ds = random_split(dataset, test_size=0.2)
train_loader = DataLoader(
    train_ds, batch_size=32, num_workers=4, collate_fn=collate_padded_batch_fn
)

embedding_params = cat_embedding_params(
    dataset.catnum, dataset.embedding_dims_template(embedding_dim=16)
)
```

`LazyDataset` is an `IterableDataset`, so do not pass `shuffle=True` to the
`DataLoader`. Choose `num_chunk_per_worker` and `memlimit_gb` according to
available memory and worker count; `ds_memory_usage(dataset,
num_chunk_per_worker, memlimit_gb)` materializes one chunk (by default) and
prints its memory usage to help tune them.

## Categorical encoding utility

`encode_cat` prepares a disk-backed categorical feature set named `setname`.
It maps each raw value to a dense rank starting at `1`, splits the groups into
`num_parts` hash partitions (default `10`), and writes, relative to the
catalog's `cat` directory:

- `<setname>/tb/part-<n>.parquet`: the group column and encoded columns
  (`DataCatalog.cat(setname)`);
- `<setname>/meta/cat2code.json`: raw category value → code
  (`DataCatalog.cat2code_json(setname)`);
- `<setname>/meta/catnum.json`: the cardinality of each column
  (`DataCatalog.catnum_json(setname)`).

```python
from pathlib import Path
from tabseq_torch.dl import LzDatasetCatalog
from tabseq_torch.dl.encoder import encode_cat, encode_targets

catalog = LzDatasetCatalog(cat=Path("data/encoded"))
encode_cat(
    data_path=Path("data/events.parquet"),
    dc=catalog,
    columns=["merchant", "channel"],
    groupby="customer_id",
    setname="default",
)

# Dense-rank the raw target column into 0-based integer classes.
encode_targets(
    src=Path("data/raw_targets.parquet"),
    dst=Path("data/targets.parquet"),
    src_column="label",
)
```

`LzDatasetCatalog` is `tabseq_torch.dl.lazy_dataset_catalog.DataCatalog`; the
default set name is `"all"`, and `catalog.show()` lists the available sets.

## Dataset and training paths

`tabseq_torch.datasets` (dataset-specific ETL for the `alfa` and `names`
datasets) and `tabseq_torch.lit.TrainingCatalog` resolve paths from the
`TABSEQ_HOME` environment variable (also read from a `.env` file) and raise
`ValueError` if it is not set. `TrainingCatalog(name).tb_logs` points to
`$TABSEQ_HOME/training/<name>/tb_logs`.

## Limitations and conventions

- Targets must be integer class indices by the time they reach the loss.
  `PdDataset` factorizes its target column automatically; for `LazyDataset`
  use `encode_targets` or encode them yourself.
- Padding is applied to every feature with the value `0`. Do not use `0` as a
  meaningful categorical code.
- Sequences must have at least one event (`pack_padded_sequence` rejects
  zero lengths).
- The package is early-stage. Tests in `tests/` are the most precise reference
  for current behavior and expected tensor shapes.
