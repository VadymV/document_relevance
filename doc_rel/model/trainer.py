"""
Trainer.
"""

import logging
import os.path
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection

from doc_rel.data_operations.loader import (
    BatchDataLoader as DataLoaderCustom,
)

_EVALUATE_OUTPUT = List[Dict[str, float]]  # 1 dict per DataLoader
T = TypeVar("T")
torch.set_float32_matmul_precision('high')
log = logging.getLogger("torcheeg")

torch.backends.mkldnn.enabled = False


def classification_metrics(metric_list: List[str]):
    # Copied from https://github.com/torcheeg/torcheeg/blob/v1.1.2/torcheeg/trainers/classifier.py
    allowed_metrics = [
        "precision",
        "recall",
        "f1score",
        "accuracy",
        "matthews",
        "auroc",
        "kappa",
    ]

    for metric in metric_list:
        if metric not in allowed_metrics:
            raise ValueError(
                f"{metric} is not allowed. Please choose 'precision', 'recall', 'f1score', 'accuracy', 'matthews', 'auroc', 'kappa'."
            )
    metric_dict = {
        "accuracy": torchmetrics.Accuracy(task="binary"),
        "precision": torchmetrics.Precision(task="binary"),
        "recall": torchmetrics.Recall(task="binary"),
        "f1score": torchmetrics.F1Score(task="binary"),
        "matthews": torchmetrics.MatthewsCorrCoef(task="binary"),
        "auroc": torchmetrics.AUROC(task="binary"),
        "kappa": torchmetrics.CohenKappa(task="binary"),
    }
    metrics = [metric_dict[name] for name in metric_list]
    return MetricCollection(metrics)


class ClassifierTrainer(pl.LightningModule):
    # Copied from here https://github.com/torcheeg/torcheeg/blob/v1.1.2/torcheeg/trainers/classifier.py
    # and modified

    def __init__(
        self,
        model: nn.Module,
        root_dir: str,
        lr: float = 1e-3,
        metrics: List[str] = ["auroc"],
        shuffle_targets: bool = False,
    ):

        super().__init__()
        self.model = model
        self.lr = lr
        self.metrics = metrics
        self.ce_fn = nn.BCEWithLogitsLoss()
        self.shuffle_targets = shuffle_targets
        self.root_dir = root_dir
        self.init_metrics(metrics)

    def init_metrics(self, metrics: List[str]) -> None:
        self.train_loss = torchmetrics.MeanMetric()
        self.val_loss = torchmetrics.MeanMetric()
        self.test_loss = torchmetrics.MeanMetric()

        self.train_metrics = classification_metrics(metrics)
        self.val_metrics = classification_metrics(metrics)
        self.test_metrics = classification_metrics(metrics)

    def fit(
        self,
        train_loader: Union[DataLoader, DataLoaderCustom],
        val_loader: Union[Optional[DataLoader], Optional[DataLoaderCustom]],
        max_epochs: int = 100,
        *args,
        **kwargs,
    ) -> Any:
        trainer = pl.Trainer(
            default_root_dir=os.path.join(
                self.root_dir, "lightning_logs", self.model.name
            ),
            max_epochs=max_epochs,
            num_sanity_val_steps=0,
            enable_checkpointing=False,
            enable_model_summary=False,
            val_check_interval=1.0,
            logger=False,
            benchmark=True,
            deterministic=False,
            enable_progress_bar=False,
            *args,
            **kwargs,
        )
        return trainer.fit(self, train_loader, val_loader)

    def test(
        self, test_loader: DataLoader, *args, **kwargs
    ) -> _EVALUATE_OUTPUT:
        trainer = pl.Trainer(
            enable_progress_bar=False,
            inference_mode=True,
            logger=False,
            *args,
            **kwargs,
        )
        return trainer.test(self, test_loader)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        (x, _, y, _, _) = batch
        if self.shuffle_targets:
            idx = torch.randperm(y.nelement())
            y = y.view(-1)[idx].view(y.size())
        y_hat = self(x)
        loss = self.ce_fn(y_hat, y)

        self.train_loss.update(loss)
        self.train_metrics.update(y_hat, y)

        return loss

    def on_train_epoch_end(self) -> None:
        self.log(
            "train_loss",
            self.train_loss.compute(),
            on_epoch=True,
        )
        for i, metric_value in enumerate(self.train_metrics.values()):
            self.log(
                f"train_{self.metrics[i]}",
                metric_value.compute(),
                on_epoch=True,
            )

        # print the metrics
        log_str = "\n[Train] "
        for key, value in self.trainer.logged_metrics.items():
            if key.startswith("train_"):
                log_str += f"{key}: {value:.3f} "
        log.info("%s\n", log_str)

        # reset the metrics
        self.train_loss.reset()
        self.train_metrics.reset()

    def validation_step(
        self, batch: Tuple[torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        (x, _, y, _, _) = batch
        if self.shuffle_targets:
            idx = torch.randperm(y.nelement())
            y = y.view(-1)[idx].view(y.size())
        y_hat = self(x)
        loss = self.ce_fn(y_hat, y)

        self.val_loss.update(loss)
        self.val_metrics.update(y_hat, y)
        return loss

    def on_validation_epoch_end(self) -> None:
        self.log(
            "val_loss",
            self.val_loss.compute(),
            on_epoch=True,
        )
        for i, metric_value in enumerate(self.val_metrics.values()):
            self.log(
                f"val_{self.metrics[i]}",
                metric_value.compute(),
                on_epoch=True,
            )

        # print the metrics
        log_str = "\n[Val] "
        for key, value in self.trainer.logged_metrics.items():
            if key.startswith("val_"):
                log_str += f"{key}: {value:.3f} "
        log.info("%s\n", log_str)

        self.val_loss.reset()
        self.val_metrics.reset()

    def test_step(
        self, batch: Tuple[torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        (x, _, y, _, _) = batch
        y_hat = self(x)
        loss = self.ce_fn(y_hat, y)

        self.test_loss.update(loss)
        self.test_metrics.update(y_hat, y)
        return loss

    def on_test_epoch_end(self) -> None:
        self.log(
            "test_loss",
            self.test_loss.compute(),
            on_epoch=True,
        )
        for i, metric_value in enumerate(self.test_metrics.values()):
            self.log(
                f"test_{self.metrics[i]}",
                metric_value.compute(),
                on_epoch=True,
            )

        # print the metrics
        log_str = "\n[Test] "
        for key, value in self.trainer.logged_metrics.items():
            if key.startswith("test_"):
                log_str += f"{key}: {value:.3f} "
        log.info("%s\n", log_str)

        self.test_loss.reset()
        self.test_metrics.reset()

    def configure_optimizers(self):
        parameters = list(self.model.parameters())
        trainable_parameters = list(
            filter(lambda p: p.requires_grad, parameters)
        )
        optimizer = torch.optim.AdamW(trainable_parameters, lr=self.lr)
        return optimizer

    def predict_step(
        self,
        batch: Tuple[torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        (x, _, _, _, _) = batch
        y_hat = self(x)
        return y_hat

    def predict(self, test_loader: DataLoader, *args, **kwargs):
        trainer = pl.Trainer(
            enable_progress_bar=False,
            inference_mode=True,
            logger=False,
            *args,
            **kwargs,
        )
        return trainer.predict(self, test_loader, return_predictions=True)
