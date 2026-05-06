from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from src.engine.callbacks import Callback

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    max_epochs: int
    device: str = "cuda"
    grad_clip: float | None = 1.0
    log_every_n_steps: int = 50
    grad_accum_steps: int = 1
    # None to disable, "float16" for P100 (needs GradScaler), "bfloat16" for Ampere+
    amp_dtype: str | None = "float16"
    checkpoint_dir: str = "checkpoints"
    resume_from: str | None = None


class Trainer:
    def __init__(
        self,
        feature_extractor: nn.Module,
        decoder: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: TrainerConfig,
        *,
        callbacks: list[Callback] | None = None,
        layer_keys: tuple[int, int, int] = (16, 19, 22),
        num_classes: int = 15,
        ignore_index: int = 255,
    ) -> None:
        self.feature_extractor = feature_extractor.to(cfg.device)
        self.decoder = decoder.to(cfg.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.callbacks: list[Callback] = callbacks or []
        self.layer_keys = layer_keys
        self.num_classes = num_classes
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.global_step = 0
        self.epoch = 0

        self._amp_dtype = _resolve_amp_dtype(cfg.amp_dtype, cfg.device)
        self._device_type = "cuda" if cfg.device.startswith("cuda") else "cpu"
        self.scaler: GradScaler | None = (
            GradScaler() if self._amp_dtype == torch.float16 else None
        )

        for p in self.feature_extractor.parameters():
            p.requires_grad_(False)

        if cfg.resume_from is not None:
            self._load_checkpoint(cfg.resume_from)

    def fit(self) -> None:
        self._emit("on_fit_start")
        for epoch in range(self.epoch, self.cfg.max_epochs):
            self.epoch = epoch
            self._run_epoch("train")
            val_metrics = self._run_epoch("val")
            self._emit("on_validation_end", val_metrics)
        self._emit("on_fit_end")

    def _run_epoch(self, stage: str) -> dict:
        is_train = stage == "train"

        self.feature_extractor.eval()
        self.decoder.train(is_train)
        loader = self.train_loader if is_train else self.val_loader
        self._emit("on_epoch_start", stage)

        total_loss = 0.0
        conf = torch.zeros(self.num_classes, self.num_classes, dtype=torch.long)
        n_batches = 0
        n_total = len(loader)

        with torch.set_grad_enabled(is_train):
            for i, (rgb, seg) in enumerate(loader):
                rgb = rgb.to(self.cfg.device, non_blocking=True)
                seg = seg.to(self.cfg.device, non_blocking=True)

                if is_train and i % self.cfg.grad_accum_steps == 0:
                    self.optimizer.zero_grad(set_to_none=True)

                loss, logits = self._forward(rgb, seg)

                if is_train:
                    self._backward(loss, batch_idx=i, total_batches=n_total)
                    self.global_step += 1
                    if self.global_step % self.cfg.log_every_n_steps == 0:
                        logger.debug("[step %d] loss=%.4f", self.global_step, loss.item())
                    self._emit("on_train_step_end", loss.item())

                total_loss += loss.item()
                n_batches += 1

                preds = logits.argmax(dim=1)       
                targets = seg[:, 0, :, :].long()   
                conf += _confusion_matrix(preds.cpu(), targets.cpu(), self.num_classes)

        if is_train and self.scheduler is not None:
            self.scheduler.step()

        metrics = _iou_metrics(conf, avg_loss=total_loss / max(n_batches, 1))
        self._emit("on_epoch_end", stage, metrics)
        return metrics

    def _forward(
        self, rgb: torch.Tensor, seg: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Dataset returns int32; normalise to float32 [0, 1] for the backbone
        rgb_float = rgb.float().div_(255.0)

        with torch.no_grad():
            features = self.feature_extractor(rgb_float)

        p3_k, p4_k, p5_k = self.layer_keys
        with autocast(
            device_type=self._device_type,
            dtype=self._amp_dtype,
            enabled=self._amp_dtype is not None,
        ):
            logits = self.decoder(features[p3_k], features[p4_k], features[p5_k])
            targets = seg[:, 0, :, :].long()
            loss = self.loss_fn(logits, targets)

        # Detach and upcast before returning so AMP internals don't leak out
        return loss, logits.detach().float()

    def _backward(
        self, loss: torch.Tensor, batch_idx: int, total_batches: int
    ) -> None:
        scaled = loss / self.cfg.grad_accum_steps

        if self.scaler is not None:
            self.scaler.scale(scaled).backward()
        else:
            scaled.backward()

        is_update = (
            (batch_idx + 1) % self.cfg.grad_accum_steps == 0
            or batch_idx + 1 == total_batches  # flush remainder at epoch end
        )
        if not is_update:
            return

        if self.cfg.grad_clip is not None:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.decoder.parameters(), self.cfg.grad_clip)

        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

    def _load_checkpoint(self, path: str) -> None:
        if not os.path.exists(path):
            logger.warning("Checkpoint %s not found", path)
            return
            
        state = torch.load(path, map_location=self.cfg.device)
        self.decoder.load_state_dict(state["decoder"])
        self.optimizer.load_state_dict(state["optimizer"])
        if state.get("scheduler") is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        if state.get("scaler") is not None and self.scaler is not None:
            self.scaler.load_state_dict(state["scaler"])
        # Resume from the *next* epoch so we don't repeat the saved one
        self.epoch = state["epoch"] + 1
        self.global_step = state["global_step"]
        logger.info(
            "Resumed from %s (epoch=%d, step=%d)",
            path,
            self.epoch,
            self.global_step,
        )

    def _emit(self, hook: str, *args) -> None:
        for cb in self.callbacks:
            getattr(cb, hook)(self, *args)


def _resolve_amp_dtype(dtype_str: str | None, device: str) -> torch.dtype | None:
    if dtype_str is None or not device.startswith("cuda"):
        return None
    mapping = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    if dtype_str not in mapping:
        raise ValueError(f"amp_dtype must be one of {list(mapping)!r}, got {dtype_str!r}")
    return mapping[dtype_str]


def _confusion_matrix(
    preds: torch.Tensor, targets: torch.Tensor, num_classes: int
) -> torch.Tensor:
    mask = (targets >= 0) & (targets < num_classes)
    combined = num_classes * targets[mask].long() + preds[mask].long()
    return torch.bincount(combined, minlength=num_classes * num_classes).reshape(
        num_classes, num_classes
    )


def _iou_metrics(conf: torch.Tensor, avg_loss: float) -> dict:
    tp = conf.diag().float()
    fn = (conf.sum(dim=1) - conf.diag()).float()
    fp = (conf.sum(dim=0) - conf.diag()).float()
    iou = tp / (tp + fn + fp).clamp(min=1e-6)
    return {"loss": avg_loss, "mIoU": float(iou.mean())}
