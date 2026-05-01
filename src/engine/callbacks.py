from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)

@dataclass
class CheckpointState:
    epoch: int
    global_step: int
    decoder_state: dict
    optimizer_state: dict
    scheduler_state: dict | None
    scaler_state: dict | None

class Callback:
    def on_fit_start(self, c: CheckpointState) -> None: ...
    def on_fit_end(self, c: CheckpointState) -> None: ...
    def on_epoch_start(self, c: CheckpointState, stage: str) -> None: ...
    def on_epoch_end(self, c: CheckpointState, stage: str, metrics: dict) -> None: ...
    def on_train_step_end(self, c: CheckpointState, step_loss: float) -> None: ...
    def on_validation_end(self, c: CheckpointState, metrics: dict) -> None: ...


class CheckpointCallback(Callback):
    def __init__(self, checkpoint_dir: str) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.best_miou = -1.0
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger.info("Checkpoint callback initialized")

    def set_best_miou(self, best_miou: float) -> None:
        self.best_miou = best_miou

    def on_validation_end(self, checkpoint_state: CheckpointState, metrics: dict) -> None:
        self._atomic_save(checkpoint_state, "last.pt")
        miou = metrics.get("mIoU", -1.0)
        if miou > self.best_miou:
            self.best_miou = miou
            self._atomic_save(checkpoint_state, "best.pt")
            logger.info("New best mIoU=%.4f — checkpoint saved", miou)

    def _atomic_save(self, c: CheckpointState, filename: str) -> None:
        state: dict = {
            "epoch": c.epoch,
            "global_step": c.global_step,
            "decoder": c.decoder_state,
            "optimizer": c.optimizer_state,
            "scheduler": c.scheduler_state,
            "best_miou": self.best_miou,
        }
        if c.scaler_state is not None:
            state["scaler"] = c.scaler_state
        path = os.path.join(self.checkpoint_dir, filename)
        tmp = path + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, path)  # POSIX-atomic: no partial reads on failure

        


class LoggingCallback(Callback):
    def __init__(self, max_epochs: int) -> None:
        self.max_epochs = max_epochs
        logger.info("Logging callback initialized")

    def on_epoch_end(self, c: CheckpointState, stage: str, metrics: dict) -> None:
        parts = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        logger.info(
            "[epoch %d/%d][%s] %s",
            c.epoch + 1,
            self.max_epochs,
            stage,
            parts,
        )
