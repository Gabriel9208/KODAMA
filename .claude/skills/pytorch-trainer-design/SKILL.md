---
name: pytorch-trainer-design
description: Use whenever the user is writing, refactoring, reviewing, or designing a PyTorch Trainer class or training loop — the orchestration layer that owns the epoch loop, optimization step, validation, checkpointing, and callbacks. Trigger on phrases like "write me a trainer," "my train.py is messy," "refactor this training code," "what should a Trainer class look like," "how do I structure training," or when the user is choosing between rolling their own Trainer vs PyTorch Lightning vs HuggingFace Trainer. Also trigger when reviewing existing training code for design issues. Provides the decision framework, anatomy of a clean Trainer, a minimal skeleton, the callbacks pattern, checkpoint strategy, and a full reference implementation. Activate even when the user hasn't explicitly said "Trainer" — if they're working on the structural orchestration of training, this is the right skill. Do NOT activate for distributed-specific concerns (DDP/FSDP) — that has its own skill.
---

# PyTorch Trainer Design

This skill is for the **Trainer class**: the orchestration layer that owns the training loop. It complements `pytorch-training-overview` (high-level mental model) and `pytorch-distributed-training` (DDP/FSDP specifics). If the user is at the planning/architecture stage, route to overview first; if they're working on multi-GPU coordination, route to distributed.

## Decide First: Three Paths

Before writing a Trainer, decide whether to write one at all.

**Use PyTorch Lightning's Trainer** when: standard supervised learning, you want DDP/FSDP/AMP/checkpointing without writing it yourself, the team prefers convention over control. Implementation surface: `LightningModule.training_step()` + `pl.Trainer.fit()`. **Stop here** — don't write a custom Trainer, just point the user at Lightning.

**Use HuggingFace `Trainer`** when: working with HF models and datasets, want a middle path between Lightning and full custom. Subclass and override hooks like `compute_loss`, `evaluation_loop`. **Stop here.**

**Write a custom Trainer** when: research with non-standard training dynamics (alternating optimization, RL, curriculum schedules, two-stage training), the codebase will live long and be deeply customized, or the user explicitly wants control over every step. The rest of this skill is for this case.

If unsure, default to Lightning. It is much easier to migrate from Lightning to custom later than the reverse — Lightning forces you into a clean structure that translates directly to a custom Trainer.

## What a Trainer Owns (and What It Doesn't)

A Trainer **owns**:

- The epoch and step loop.
- Calling `model(batch)`, computing loss, calling `loss.backward()`, calling `optimizer.step()`.
- Switching `model.train()` / `model.eval()` and `torch.set_grad_enabled` correctly.
- Coordinating logging, metric updates, and callback invocation.
- Coordinating checkpoint save/load and resume.
- Mechanical concerns: AMP scaler, gradient accumulation, gradient clipping, device placement.

A Trainer **does not own**:

- Model architecture (lives in `src/models/`, passed in).
- Loss function (lives in `src/losses/`, passed in).
- Optimizer or scheduler construction (built outside, passed in).
- Data loading (DataLoader or DataModule passed in).
- Config (a `cfg` object passed in, treated as read-only).

The test: can you swap any one of these without touching the Trainer? If yes, the boundaries are right.

## Minimal Trainer Skeleton

The smallest viable Trainer that demonstrates clean structure. Use as a starting point and grow it.

```python
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from torch.utils.data import DataLoader

@dataclass
class TrainerConfig:
    max_epochs: int
    grad_clip: float | None = None
    log_every_n_steps: int = 50
    device: str = "cuda"

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        loss_fn,
        optimizer: torch.optim.Optimizer,
        scheduler,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: TrainerConfig,
    ):
        # Store dependencies. Do NOT construct any of them here.
        self.model = model.to(cfg.device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.global_step = 0
        self.epoch = 0

    def fit(self) -> None:
        for epoch in range(self.cfg.max_epochs):
            self.epoch = epoch
            self._run_epoch(stage="train")
            self._run_epoch(stage="val")

    def _run_epoch(self, stage: str) -> None:
        is_train = stage == "train"
        self.model.train(is_train)
        loader = self.train_loader if is_train else self.val_loader

        with torch.set_grad_enabled(is_train):
            for batch in loader:
                batch = self._move_to_device(batch)
                loss = self._step(batch, is_train=is_train)

                if is_train:
                    self.global_step += 1
                    if self.global_step % self.cfg.log_every_n_steps == 0:
                        print(f"[step {self.global_step}] loss={loss.item():.4f}")

        if is_train and self.scheduler is not None:
            self.scheduler.step()

    def _step(self, batch, is_train: bool) -> torch.Tensor:
        inputs, targets = batch
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, targets)

        if is_train:
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.grad_clip
                )
            self.optimizer.step()

        return loss

    def _move_to_device(self, batch):
        if isinstance(batch, (list, tuple)):
            return [b.to(self.cfg.device, non_blocking=True) for b in batch]
        return batch.to(self.cfg.device, non_blocking=True)
```

What to notice:

- `__init__` only stores dependencies. **Never construct model / optimizer / data inside Trainer**.
- Train and val use the same `_run_epoch` with a stage flag — no duplication.
- `set_to_none=True` on `zero_grad` — faster, and surfaces bugs where gradients aren't computed.
- `non_blocking=True` on `.to(device)` — lets transfer overlap with compute (requires `pin_memory=True` in the DataLoader; without that, `non_blocking` silently does nothing).
- `torch.set_grad_enabled(is_train)` wraps the whole epoch — cleaner than per-step `with torch.no_grad()`.

## The Callbacks Pattern

Once the Trainer needs to do more than print loss — logging to W&B, early stopping, periodic checkpointing, custom visualizations — **don't add these directly to the Trainer**. Use callbacks.

A callback is an object with optional methods like `on_train_start`, `on_epoch_end`, `on_step_end`. The Trainer calls these at known points. Concrete benefits:

- Trainer code stays focused on the loop.
- Cross-cutting concerns become composable — use any subset.
- Testing is easier: mock callbacks instead of mocking W&B.

A complete reference implementation with callbacks, AMP, gradient accumulation, and checkpointing is in **`assets/trainer_with_callbacks.py`**. Read it when the minimal skeleton above isn't enough.

The hook surface to recommend:

- `on_fit_start(trainer)` / `on_fit_end(trainer)` — once per `fit()` call.
- `on_epoch_start(trainer)` / `on_epoch_end(trainer, metrics)` — per epoch, both stages.
- `on_train_step_end(trainer, loss)` — after each train step (use sparingly; fires often).
- `on_validation_end(trainer, metrics)` — after val epoch, with computed metrics.

Resist adding more hooks. Each hook is a coupling point and has to be maintained forever.

## State Management & Checkpointing

What a checkpoint must contain:

- `model.state_dict()` — always.
- `optimizer.state_dict()` — for resume (Adam moments, momentum buffers, etc.).
- `scheduler.state_dict()` — for resume.
- `epoch`, `global_step` — for resume.
- `torch.get_rng_state()`, `torch.cuda.get_rng_state_all()`, NumPy and Python RNG states — for true bit-exact resume.
- `cfg` (or a config hash + a separately-saved config artifact) — for traceability.

What NOT to save:

- The model class itself. Use `state_dict`, never `pickle.dump(model)`.
- DataLoader state — DataLoaders are stateless across runs unless you use a stateful sampler.
- Loss / metric histories — those go to your experiment tracker.

**Atomic save** pattern (avoids corrupted checkpoints if the process is killed mid-write):

```python
import os, torch

def atomic_save(state, path):
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)  # atomic on POSIX
```

A sensible default policy is to keep "best" (highest val metric) and "last" (most recent) only. Avoid saving every epoch unless storage is cheap.

For models you'll share, prefer **safetensors** over PyTorch's pickle format — it's safer (no arbitrary code execution on load) and faster.

## Common Pitfalls

These come up over and over:

1. **Forgetting `model.eval()` for validation.** BatchNorm and Dropout behave differently in train vs eval. The `_run_epoch` pattern handles this; if you have a separate `validate()` method, it must call `self.model.eval()` first.

2. **Wrong `zero_grad` placement.** Use `optimizer.zero_grad(set_to_none=True)` _before_ `loss.backward()`, not after `optimizer.step()`. (Either order works, but before-backward is conventional and avoids a subtle bug where you accumulate gradients across `.backward()` calls if you change loop structure later.)

3. **Computing metrics on training batches as if they were the final metric.** Train metrics are noisy and biased by augmentation. Always do a separate val pass with the model in `eval()` mode.

4. **Wrong order when resuming.** Optimizer's `state_dict` must be loaded _after_ the optimizer is constructed (which requires the model parameters to exist). The correct order is: build model → load model state_dict → build optimizer → load optimizer state_dict → build scheduler → load scheduler state_dict.

5. **`scheduler.step()` in the wrong place.** Most schedulers step per epoch (at end of train epoch). Some (`OneCycleLR`, warmup schedulers, `CosineAnnealingWarmRestarts` in some configs) step per iteration. Check the scheduler's docs. Wrong placement is a silent bug — training "works" but the schedule is wrong.

6. **`non_blocking=True` without `pin_memory=True`.** It silently does nothing. Both must be set.

7. **AMP without `GradScaler` for fp16.** With `torch.amp.autocast(dtype=torch.float16)`, gradients can underflow. `GradScaler` is required. With `bfloat16` (Ampere+ GPUs), `GradScaler` is unnecessary — bf16 has the same dynamic range as fp32.

8. **Logging from inside the model's `forward()`.** The model shouldn't know it's being trained. All logging belongs in the Trainer or callbacks.

9. **Mixing `model.parameters()` order across saves.** If the model architecture is deterministic but the parameter iteration order changes (e.g., a `dict` in some Python version), state_dict keys may shift. PyTorch usually handles this correctly, but anything that relies on parameter index (rare) breaks. Prefer named state_dicts.

## Reference Implementations

- **`assets/trainer_with_callbacks.py`** — full single-GPU Trainer with callbacks, AMP, gradient accumulation, atomic checkpointing, resume support, and a small set of built-in callbacks (logging, checkpointing, early stopping). ~280 lines. Copy and adapt.

For DDP/FSDP extensions, route to the **`pytorch-distributed-training`** skill — distributed-specific Trainer modifications are out of scope here.

## When to Switch from Custom Trainer to Lightning

Honest exit criteria. Recommend porting to Lightning if any of these hit:

- The Trainer is over ~500 lines and growing.
- You're reimplementing things Lightning already has (auto-resume, gradient accumulation, multiple precision modes, distributed strategies).
- Multiple people are contributing and disagreeing about the loop's structure.
- You need DDP/FSDP and don't want to learn the failure modes from scratch.

The clean separation maintained above (model, data, optim built externally) means the port is mostly mechanical: rename `Trainer` → `LightningModule`, move `_step` → `training_step`/`validation_step`, delete checkpoint code (Lightning does it), delete the `_move_to_device` helper.

## Self-Check Before Returning Code

Before showing the user a Trainer, verify:

- `__init__` only stores dependencies; doesn't build them.
- Train and val share an epoch function with a stage flag.
- `optimizer.zero_grad(set_to_none=True)` is used.
- `non_blocking=True` is paired with `pin_memory=True` in the DataLoader.
- Scheduler stepping is at the documented frequency for that specific scheduler.
- A callback or equivalent extension point exists if logging/checkpointing is needed.
- Checkpointing uses `state_dict`, not `pickle`; saves are atomic.
- No model-internal knowledge inside the Trainer.
- Resume order is correct: build → load state_dict, in dependency order.

## Response Style for This Skill

- Lead with the **decide-first** check. If Lightning or HF Trainer fits, recommend it and stop — don't reflexively write a custom Trainer.
- When showing code, start with the minimal skeleton and add complexity only as the user's needs justify it.
- For pitfall-style questions ("why is my X broken"), check the Common Pitfalls list first before diving into a full review.
- Reference the asset file rather than inlining 300 lines of code in your response.
