import torch
import lightning as L
from tabseq_torch.nn.lstm import TabLSTM, PoolingType
from tabseq_torch.tab2sec import CatColumnsEncoder, NumColumnsEncoder, Encoder, Tab2Seq
from tabseq_torch.dl import (
    PaddedTabBatch,
    CatColEmbeddingParams,
    ColumnsByType,
    ColumnType,
)
from torchmetrics.classification import MulticlassAccuracy


class LitMulticlassTabLSTM(L.LightningModule):
    def __init__(
        self,
        cat_enc_params: dict[str, CatColEmbeddingParams],
        h_size,
        num_classes,
        columns_by_type: ColumnsByType,
        num_layers: int,
        lr=1e-3,
        weight_decay_rate=0.001,
    ):
        super().__init__()
        self._weight_decay_rate = weight_decay_rate
        encoders: list[Encoder] = [CatColumnsEncoder(cat_enc_params)]
        if ColumnType.NUM in columns_by_type:
            encoders.append(
                NumColumnsEncoder(num_dim=len(columns_by_type[ColumnType.NUM]))
            )
        self.tab2sec = Tab2Seq(encoders)

        self.b_lstm = TabLSTM(
            input_size=self.tab2sec.output_size,
            h_size=h_size,
            num_classes=num_classes,
            pooling_type=PoolingType.LAST,
            num_layers=num_layers,
        )
        self.loss = torch.nn.CrossEntropyLoss()
        self.lr = lr
        self.accuracy = MulticlassAccuracy(num_classes=num_classes, average="micro")

    def _forward(self, batch: PaddedTabBatch):
        rnn_batch: torch.Tensor = self.tab2sec(batch)
        output = self.b_lstm(rnn_batch)
        loss = self.loss(output, batch.targets)
        acc = self.accuracy(output, batch.targets)
        return loss, acc

    def training_step(self, batch: PaddedTabBatch, batch_idx):
        # training_step defines the train loop.
        loss3, acc4 = self._forward(batch)
        metrics = {"acc11": acc4, "cross_entr111": loss3}

        self.log_dict(
            dictionary=metrics,
            on_epoch=True,
            on_step=True,
            batch_size=len(batch),
            prog_bar=True,
        )

        return loss3

    def validation_step(self, batch, batch_idx):
        loss, acc = self._forward(batch)
        metrics = {"val_acc": acc, "val_cross_entr": loss}
        self.log_dict(
            dictionary=metrics, on_epoch=True, batch_size=len(batch), prog_bar=True
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self._weight_decay_rate
        )
        return optimizer

    def transfer_batch_to_device(
        self, batch, device: torch.device, dataloader_idx
    ) -> PaddedTabBatch:
        if isinstance(batch, PaddedTabBatch):
            # move all tensors in your custom data structure to the device
            batch.to(device)
        return batch
