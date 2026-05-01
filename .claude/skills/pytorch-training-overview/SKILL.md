---
name: pytorch-training-overview
description: Use whenever the user is starting, designing, reviewing, or refactoring a PyTorch model training pipeline. Trigger on broad questions about ML training architecture — "how should I structure this," "what's best practice for X," "set up a training repo," "review my training code," "choose between Lightning and a custom Trainer" — or any time the user is making high-level decisions about how training code is organized. This skill establishes the core mental model (separation of concerns, three-track framework choice, standard layout) and routes to more specific skills (pytorch-trainer-design, hydra-config-setup, ml-experiment-rigor, etc.) when the work narrows in scope. Activate even when the request seems vague — provide the thinking framework before diving into details. Do NOT activate for narrow tactical questions like "fix this NaN loss" or "why is my GPU not being used" — those have dedicated skills.
---

# PyTorch Training Overview

This skill is the **entry point** for non-trivial PyTorch training work. Its job is to establish the mental model and route to specific skills when the conversation narrows. If the user's question fits squarely under another skill's scope (designing a Trainer, debugging reproducibility, setting up Hydra), use that skill instead.

When activating, briefly orient: identify which **track** (A/B/C below) the user appears to be on, and whether a **sibling skill** is a better fit. Don't recite the entire mental model unless they're new — be efficient.

## Core Principle: Separation of Concerns

The single most important design rule in ML training code is that **each component should know nothing about the internals of the others**. Violating this is the root cause of nearly all unmaintainable training code.

Four hard boundaries:

1. **Model knows nothing about training.** A `nn.Module` only defines architecture and `forward()`. No optimizer, no loss, no gradient logic, no checkpoint code inside the model class. This makes the model exportable to ONNX/TorchScript without surgery and reusable for inference.

2. **Trainer knows nothing about model internals.** The Trainer interacts with the model only via `model(batch)` and `loss_fn(outputs, targets)`. It should be possible to swap the model entirely without touching the Trainer.

3. **Data knows nothing about the model.** A `Dataset` returns tensors. A `DataModule` builds DataLoaders. Neither should hardcode model-specific input shapes — those go in config.

4. **Config knows nothing about runtime.** Config files are pure data: hyperparameters, paths, flags. They should not import torch or instantiate objects — use a separate instantiation layer (Hydra's `instantiate`, or a factory function).

The diagnostic question: _"If I swap component X for a different one, how many files need to change?"_ The answer should be **one**.

## Three-Track Framework Choice

There are three reasonable ways to organize training code in 2026. Pick one early; switching is expensive.

### Track A — PyTorch Lightning + Hydra

Pick when: standard supervised learning, single-machine or DDP, the team values consistency over flexibility, you want experiment tracking and distributed training "for free."

Tradeoff: Lightning hides parts of the loop, which frustrates when debugging exotic behaviors. Debugging a `LightningModule` sometimes means reading Lightning source.

### Track B — Custom Trainer + Hydra

Pick when: research with non-standard training (RL, custom optimization schedules, two-phase training, GAN-like alternating updates), you want explicit control over every step, the codebase will live a long time and be read by many people, or you're building infrastructure for others to use.

Tradeoff: you write more code. You're responsible for AMP, distributed, and checkpointing details. Reference codebases: nanoGPT, Meta lingua, torchtitan.

### Track C — HuggingFace `Trainer` + `accelerate`

Pick when: working with HuggingFace models or datasets, NLP or multimodal where the HF ecosystem provides existing utilities, you want a middle ground between Lightning's abstraction and full-custom code.

Tradeoff: strongly opinionated for the HF model API; awkward for non-HF models. `accelerate` alone (without `Trainer`) is a great option for "I want to keep my training loop but not deal with distributed."

### Decision Heuristic

- Standard supervised learning, fresh project → **Track A**.
- Research with custom training dynamics → **Track B**.
- Inside the HF ecosystem → **Track C**.

When designing the Trainer class itself (Track B, or extending Lightning for non-trivial behavior), use the **`pytorch-trainer-design`** skill.

## Standard Project Layout

The convention most modern PyTorch projects converge on:

```
project/
├── configs/                  # Hydra YAML, layered (model/, data/, trainer/, experiment/)
├── src/<package>/
│   ├── data/                 # Dataset, DataModule, transforms
│   ├── models/               # nn.Module only, no training logic
│   ├── losses/
│   ├── metrics/
│   ├── engine/               # Trainer / training loop
│   ├── utils/                # seed, logging, checkpoint helpers
│   └── __init__.py
├── scripts/
│   ├── train.py              # entry point, ~10–30 lines
│   ├── eval.py
│   └── predict.py
├── tests/                    # pytest unit tests
├── notebooks/                # exploration only, never source of truth
├── outputs/                  # Hydra runs, gitignored
├── checkpoints/              # gitignored
├── pyproject.toml            # ruff, mypy, pytest config
└── README.md
```

Two non-obvious rules:

- **`scripts/train.py` should be very thin** (10–30 lines). It only loads config and calls into `src/`. All real logic lives in importable modules so `tests/` can test them.
- **`src/` is a real Python package**, installable via `pip install -e .`. This makes imports clean and tests reliable.

## Sibling Skill Router

When the user's request narrows to a specific concern, route to the right skill. This skill should not duplicate their content.

| User intent                                             | Skill                          |
| ------------------------------------------------------- | ------------------------------ |
| Designing/refactoring the Trainer class itself          | `pytorch-trainer-design`       |
| Setting up Hydra config layout, structured configs      | `hydra-config-setup`           |
| Reproducibility, seeds, NaN debugging, sanity checks    | `ml-experiment-rigor`          |
| W&B / MLflow setup, sweep configuration                 | `ml-experiment-tracking`       |
| Multi-GPU, DDP, FSDP, `torchrun`, launch scripts        | `pytorch-distributed-training` |
| Slow training, OOM, profiling, AMP, `torch.compile`     | `pytorch-performance-tuning`   |
| `Dataset` / `DataModule` design, WebDataset, large data | `pytorch-data-pipeline`        |
| Testing ML code, CI for training repos                  | `ml-testing-strategy`          |
| Initial scaffold, `pyproject.toml`, pre-commit setup    | `pytorch-project-scaffold`     |

If none match, stay in this skill.

## Hard Rules: Refuse These Patterns

If the user's existing code or proposal contains these, name them and propose a fix before continuing:

- **Hardcoded paths or hyperparameters anywhere outside config.** Replace with config keys.
- **Notebook as the source of truth.** Notebooks are for exploration. Move training logic into `src/` and import from notebooks if needed.
- **God-class Trainer** that owns model definition, optimizer construction, data loaders, and loss — all instantiated inside `__init__`. Use dependency injection: pass these in.
- **`pickle.dump(model)` instead of `state_dict`.** When the model class moves, the pickle becomes unloadable. Save `state_dict` only; loading code reconstructs the model from config.
- **No seed setting at all.** Even research code needs reproducibility for debugging.
- **Validation interleaved into the train loop with shared mutable state.** Train and val should be separate functions; sharing batch-level state causes subtle bugs.
- **Logging from inside `model.forward()`.** The model shouldn't know it's being trained. Logging belongs in Trainer or callbacks.

## Out of Scope

This skill does **not** cover:

- Specific model architectures (transformers, vision models, diffusion) — domain-specific.
- Inference, serving, or deployment — different skill.
- Pure data engineering (DVC, lakehouse setup) — different domain.
- Hyperparameter search algorithms (Optuna, Ray Tune) — assume the user knows what to sweep.

## References for Deeper Reading

- **Reference codebases by track:**
  - Track B minimalist: [Karpathy nanoGPT](https://github.com/karpathy/nanoGPT)
  - Track B production: [Meta lingua](https://github.com/facebookresearch/lingua), [torchtitan](https://github.com/pytorch/torchtitan)
  - Canonical PyTorch examples: [pytorch/examples](https://github.com/pytorch/examples) (especially `imagenet/`)
  - Track A template: [ashleve/lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)
- **Lightning docs:** https://lightning.ai/docs/pytorch/stable/
- **HF Trainer docs:** https://huggingface.co/docs/transformers/main_classes/trainer
- **Hydra docs:** https://hydra.cc/

## Response Style for This Skill

When this skill is active:

- Lead with **which track** the user is on and **which sibling skill** (if any) you're routing to.
- Don't recite the full mental model unless asked or unless the user is clearly new.
- When proposing a layout, show the directory tree once and explain only the non-obvious parts.
- Prefer concrete recommendations over enumerating all options — the user can ask for alternatives.
