import torch
from typing import cast, Any
from enum import Enum, auto

import lightning as L
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryPrecisionRecallCurve,
)
import matplotlib.pyplot as plt

from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.tensorboard.writer import SummaryWriter

from tabseq_torch.dl import (
    CatColEmbeddingParams,
    ColumnType,
    ColumnsByType,
    PaddedTabBatch,
)

from tabseq_torch.nn.lstm import PoolingType, TabLSTM, Attention
from tabseq_torch.tab2sec import CatColumnsEncoder, NumColumnsEncoder, Tab2Seq


class LogPRMethod(Enum):
    add_figure = auto()
    add_pr_curve = auto()


class OptType(Enum):
    ADAM = auto()
    ADAMW = auto()


class LitImbBinTabLSTM(L.LightningModule):
    def __init__(
        self,
        cat_enc_params: dict[str, CatColEmbeddingParams],
        h_size: int,
        columns_by_type: ColumnsByType,
        bidirectional: bool,
        pooling_type: PoolingType,
        attention: bool = False,
        pos_weight: float | None = None,
        lr: float = 1e-3,
        num_layers: int = 1,
        log_pr_with: LogPRMethod = LogPRMethod.add_pr_curve,
        opt_type: OptType = OptType.ADAM,
    ):
        super().__init__()
        self.opt_type = opt_type
        self.tab2sec = Tab2Seq(
            encoders=torch.nn.ModuleList([CatColumnsEncoder(cat_enc_params)])
        )
        self.save_hyperparameters()
        if ColumnType.NUM in columns_by_type:
            self.tab2sec.encoders.append(
                NumColumnsEncoder(num_dim=len(columns_by_type[ColumnType.NUM]))
            )

        self.b_lstm = TabLSTM(
            input_size=self.tab2sec.output_size,
            h_size=h_size,
            num_classes=1,
            attention=attention,
            pooling_type=pooling_type,
            bidirectional=bidirectional,
            num_layers=num_layers,
        )

        self.train_auc = BinaryAUROC()
        self.val_auc = BinaryAUROC()
        self.val_pr_curve = BinaryPrecisionRecallCurve()

        self.loss = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight) if pos_weight else None
        )

        self.lr = lr

        self._validation_step_num_batchs_checksum = 0
        self.log_pr_with: LogPRMethod = log_pr_with

    def _forward(self, batch: PaddedTabBatch) -> torch.Tensor:
        e_batch: torch.Tensor = self.tab2sec(batch)
        output = self.b_lstm(e_batch).squeeze(-1)
        return output

    def _log_auc(self, probs: torch.Tensor, targets: torch.Tensor, batch_idx: int):
        self.train_auc.update(probs.cpu(), targets.cpu())
        if hasattr(self.trainer, "log_every_n_steps"):
            log_every_n_steps = self.trainer.log_every_n_steps
        else:
            log_every_n_steps = 200
        if batch_idx % log_every_n_steps == 0:
            if batch_idx == 0:
                self.train_auc.reset()
            else:
                auc = self.train_auc.compute()
                self.log(
                    "train_auc",
                    auc,
                    on_step=True,
                    prog_bar=True,
                )
            self.train_auc.reset()

    def training_step(self, batch: PaddedTabBatch, batch_idx):
        logits = self._forward(batch)
        loss = self.loss(logits, batch.targets.float())
        probs = torch.sigmoid(logits.detach())
        batch_size = len(batch)
        self.log(
            "train_bce_loss",
            loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        # manage memory with log_every_n_steps
        # Lightning handle the entire lifecycle (update, compute, reset) wiht self.trainer.log_every_n_steps but
        # Bug: automatic logging doesn't log metric on steps if .update is used https://github.com/Lightning-AI/pytorch-lightning/issues/20160 so
        self._log_auc(probs=probs, targets=batch.targets, batch_idx=batch_idx)

        return loss

    def on_train_epoch_end(self): ...

    def validation_step(self, batch, batch_idx):
        # manage memory with self.trainer.val_check_interval
        logits = self._forward(batch)
        self._validation_step_num_batchs_checksum += 1
        targets = batch.targets
        loss = self.loss(logits, batch.targets.float())

        probs = torch.sigmoid(logits)
        batch_size = len(batch)
        self.log(
            "val_bce_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.val_auc(probs, targets)
        self.log(
            "val_auc",
            self.val_auc,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        # the noautomated life cycle(1. update -> 2. compute -> 3. reset) of val_pr_curve: 1. update
        # UserWarning: No positive samples, so use warnings.filterwarnings("ignore")
        self.val_pr_curve.update(probs.cpu(), targets.cpu())

    def on_validation_epoch_end(self):
        # manage memory with self.trainer.limit_val_batches

        loger: TensorBoardLogger = cast(TensorBoardLogger, self.logger)
        # the noautomated life val_pr_curve: 2. compute

        labels: torch.Tensor = torch.cat(self.val_pr_curve.target).bool()
        preds = torch.cat(self.val_pr_curve.preds)
        writer: SummaryWriter = loger.experiment
        match self.log_pr_with:
            case LogPRMethod.add_pr_curve:
                precision, recall, thresholds = self.val_pr_curve.compute()
                writer.add_pr_curve(
                    tag="Precision-Recall",
                    labels=labels,
                    predictions=preds,
                    global_step=self.trainer.global_step,
                )

            case LogPRMethod.add_figure:
                fig, ax_ = self.val_pr_curve.plot(score=True)
                writer.add_figure(
                    "Precision-Recall", fig, global_step=self.trainer.global_step
                )

        writer.close()
        self.val_pr_curve.reset()

    def configure_optimizers(self) -> Any:

        match self.opt_type:
            case OptType.ADAM:
                return torch.optim.Adam(self.parameters(), lr=self.lr)
            case OptType.ADAMW:
                optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
                scheduler = {
                    "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, mode="min", factor=0.5, patience=2
                    ),
                    "monitor": "train: bce_loss",
                    "interval": "epoch",
                }
                return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def transfer_batch_to_device(
        self, batch, device: torch.device, dataloader_idx
    ) -> PaddedTabBatch:
        if isinstance(batch, PaddedTabBatch):
            batch.to(device)
        return batch
