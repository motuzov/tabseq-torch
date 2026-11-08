from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from tabseq_torch.datasets.names.ds import create_gpddataset
from tabseq_torch.datasets.names.etl import meta
from tabseq_torch.dl import cat_embedding_params, collate_padded_batch_fn
from tabseq_torch.dl.types import ColumnType
from tabseq_torch.lit.lstm import LitMulticlassTabLSTM
from tabseq_torch.lit.training_catalog import TrainingCatalog
from torch.utils import data

from lightning.pytorch.callbacks import EarlyStopping
import optuna
from optuna.integration import PyTorchLightningPruningCallback


def objective(trial: optuna.Trial):
    ds = create_gpddataset(num_columns=["F", "T"])
    ds.num_classes
    ds.catnum

    embedding_dim = 8
    embedding_dims = dict.fromkeys(
        ds.get_columns_by_type(column_type=ColumnType.CAT), embedding_dim
    )
    embedding_dims["vc"] = 4
    embedding_params = cat_embedding_params(
        catnum=ds.catnum, col_embedding_dims=embedding_dims
    )

    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)

    b_lstm = LitMulticlassTabLSTM(
        cat_enc_params=embedding_params,
        h_size=64,
        num_classes=ds.num_classes,
        columns_by_type=ds.type_columns,
        num_layers=4,
        lr=lr,
    )

    train_set_size = int(len(ds) * 0.9)
    valid_set_size = len(ds) - train_set_size

    seed = torch.Generator().manual_seed(42)
    train_set, valid_set = data.random_split(
        ds, [train_set_size, valid_set_size], generator=seed
    )

    train_loader = data.DataLoader(
        dataset=train_set,
        shuffle=True,
        batch_size=20,
        collate_fn=collate_padded_batch_fn,
        num_workers=8,
    )
    valid_loader = data.DataLoader(
        dataset=valid_set,
        shuffle=True,
        batch_size=20,
        collate_fn=collate_padded_batch_fn,
        num_workers=0,
    )

    tc = TrainingCatalog(dataset_name=meta.name)

    logger = TensorBoardLogger(
        save_dir=tc.tb_logs,
        name="names_study",
        version=f"trial_{trial.number:03d}",
        default_hp_metric=False,
    )

    pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val_acc")
    trainer = L.Trainer(
        max_epochs=500, accelerator="auto", logger=logger, callbacks=[pruning_callback]
    )
    trainer.fit(
        model=b_lstm, train_dataloaders=train_loader, val_dataloaders=valid_loader
    )
    return trainer.callback_metrics["val_acc"].item()


def study():
    study = optuna.create_study(
        study_name="names_study",
        storage="sqlite:///optuna_study.db",  # persistent storage = observable from outside
        load_if_exists=True,  # rerun the script to resume
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(objective, n_trials=100)


if __name__ == "__main__":
    study()
