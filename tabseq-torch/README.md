# tabseq-torch

`tabseq-torch` turns grouped tabular data into variable-length sequences for
PyTorch models. It is designed for data where each entity—such as a customer,
session, account, or device—has a sequence of tabular events. Categorical
features are embedded, numeric features are concatenated, and the resulting
event vectors are classified with an LSTM.

## What it provides

- `PdDataset` for small-to-medium parquet datasets that fit in memory.
- `LazyDataset` and `DDChunkReader` for reading grouped parquet data in chunks.
- `PaddedTabBatch` and a DataLoader collate function for batches with different
  sequence lengths.
- `CatColumnsEncoder`, `NumColumnsEncoder`, and `Tab2Seq` to transform a
  padded tabular batch into an LSTM-ready tensor.
- `TabLSTM` and the Lightning module `LitMulticlassTabLSTM` for multiclass
  sequence classification.

## Installation

This project currently targets Python 3.13 or newer. From the package
directory, install it in editable mode together with the libraries used by the
data and training modules:

```bash
cd tabseq-torch
python -m pip install -e .
python -m pip install torch lightning torchmetrics pandas pyarrow polars duckdb numpy
```

For development and test runs, also install pytest:

```bash
python -m pip install pytest
pytest
```

## Data model

The input event table is a parquet file. Every row is one event and one column
identifies the entity that owns that event. All events for an entity form one
sequence. A separate target parquet file supplies one label per entity.

For example, an event table grouped by `customer_id` might look like this:

| customer_id | merchant | channel | amount |
| --- | --- | --- | --- |
| 101 | groceries | card | 12.50 |
| 101 | transit | card | 2.40 |
| 202 | books | web | 18.00 |

Its target table contains a target column, such as `label`, aligned with the
dataset's group order. Categorical feature values are encoded to positive
integers; `0` is reserved for padding. Numeric values are converted to float
tensors. Keep event rows in the desired temporal order before writing parquet:
the datasets preserve source row order within each group.

## Quick start: in-memory data

`PdDataset` is the simplest route for experimenting with data that fits in
memory. It reads the event and target parquet files, encodes categorical
columns, and exposes one `GroupData` item per entity.

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

Train the Lightning module with your preferred trainer:

```python
import lightning as L

trainer = L.Trainer(max_epochs=10)
trainer.fit(model, train_dataloaders=loader)
```

Each collated batch contains tensors shaped `[batch, longest_sequence, ...]`:

- `batch[column]` is the padded tensor for that feature.
- `batch.lengths` contains original sequence lengths.
- `batch.targets` contains one integer class label for each sequence.

## Model building blocks

Use the lower-level components when a Lightning training loop is not needed:

```python
from tabseq_torch.tab2sec import CatColumnsEncoder, NumColumnsEncoder, Tab2Seq
from tabseq_torch.nn import TabLSTM, PoolingType

tab_to_sequence = Tab2Seq([
    CatColumnsEncoder(embedding_params),
    NumColumnsEncoder(num_dim=len(dataset.num_columns)),
])

classifier = TabLSTM(
    input_size=tab_to_sequence.output_size,
    h_size=64,
    num_classes=dataset.num_classes,
    pooling_type=PoolingType.LAST,
)

batch = next(iter(loader))
logits = classifier(tab_to_sequence(batch))
```

`TabLSTM` supports last-state pooling (`PoolingType.LAST`), combined average
and last-state pooling (`PoolingType.AVG`), optional attention, stacked LSTM
layers, and bidirectional LSTMs.

## Large parquet datasets

For datasets that should not be held in memory, use `DDChunkReader` with
`LazyDataset`. The reader partitions entities across chunks and materializes
only the current chunk. Categorical parquet data must already contain encoded
integer values, with `0` reserved for padding. The helper `encode_cat` builds
that encoded categorical dataset and its metadata from a source parquet file.

```python
from pathlib import Path
from tabseq_torch.dl import DDChunkReader, LazyDataset

reader = DDChunkReader(
    groupby="customer_id",
    num_chunk_per_worker=8,
    num_groups=1_000_000,
)
dataset = LazyDataset(reader)

dataset.set_catdata(
    data_path=Path("data/encoded/categories"),
    columns_to_select=["merchant", "channel"],
    catnum_json=Path("data/meta/default/catnum.json"),
)
dataset.set_numdata(
    data_path=Path("data/events.parquet"),
    columns_to_select=["amount"],
)
dataset.set_targetmeta(
    data_path=Path("data/targets.parquet"),
    target_column="label",
)
```

The chunk reader is an `IterableDataset` backend, so use a `DataLoader` with
`collate_padded_batch_fn` as above. Choose `num_chunk_per_worker` and DuckDB's
`memlimit_gb` according to available memory and worker count.

## Categorical encoding utility

`encode_cat` prepares a disk-backed categorical feature set. It writes:

- encoded parquet partitions under `DataCatalog.cat(setname)`;
- `cat2code.json`, which maps raw category values to codes; and
- `catnum.json`, which records the cardinality of each categorical column.

```python
from pathlib import Path
from tabseq_torch.dl import DataCatalog
from tabseq_torch.dl.encoder import encode_cat

catalog = DataCatalog(
    src_tabts_path=Path("data/events.parquet"),
    cat_path=Path("data/encoded"),
    meta_path=Path("data/meta"),
)
encode_cat(
    data_path=Path("data/events.parquet"),
    dc=catalog,
    columns=["merchant", "channel"],
    groupby="customer_id",
    setname="default",
)
```

## Limitations and conventions

- Targets must be integer class indices by the time they reach
  `CrossEntropyLoss`. `PdDataset` factorizes its target column automatically.
- Padding is applied to every feature with the value `0`. Do not use `0` as a
  meaningful categorical code.
- The current LSTM receives padded time steps directly; it does not yet pack
  sequences using `batch.lengths`. Consider this when padding is substantial.
- The package is early-stage. Tests in `tests/` are the most precise reference
  for current behavior and expected tensor shapes.
