# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

KODAMA (Kinetic Obstacle & Distance Awareness for Mobility Assistance) — panoptic segmentation on the SANPO dataset for accessibility-focused street scene understanding. Uses a frozen YOLO backbone with custom semantic decoder, panoptic fusion, and depth-aware risk scoring.

## Development Environment

- Python 3.11, PyTorch 2.2.0, CUDA 12.1 (Miniconda)
- Conda env name: `KODAMA`
- Run commands via: `conda run -n KODAMA <command>`

## Common Commands

```bash
# Tests (must use KODAMA conda env — pytest not in base)
conda run -n KODAMA python -m pytest tests/ -v
conda run -n KODAMA python -m pytest tests/scripts/test_pipeline.py -v  # single file
conda run -n KODAMA python -m pytest tests/scripts/test_pipeline.py::TestValidation::test_3_1_all_present_counts_match -v  # single test

# Linting
ruff check .

# Data pipeline (Windows — downloads from GCS, preprocesses, packs, uploads to GDrive)
conda run -n KODAMA python scripts/run_pipeline.py
conda run -n KODAMA python scripts/run_pipeline.py --config configs/pipeline_config_test.yaml

# Streaming (Ubuntu P100 — mounts GDrive, streams WebDataset shards)
python scripts/run_stream.py
```

## Architecture

### Data Pipeline (`scripts/data_pipeline/`)

Modular package handling the SANPO dataset processing pipeline. Each module is named by function:

- `config.py` — loads `configs/pipeline_config.yaml`, resolves relative paths, caches result
- `progress.py` — atomic JSON progress file read/write, decimation config validation
- `download.py` — GCS download via `gcloud storage cp`
- `validate.py` — per-camera data integrity checks (file counts, frame ID alignment)
- `decimate.py` — deterministic frame sampling (`_decimate_indices` is a pure function)
- `preprocess.py` — crop/resize via `SANPO_data_processor` (lazy imports cv2 to keep tests fast)
- `pack.py` — WebDataset shard packing (lazy imports webdataset)
- `upload.py` — rclone upload, verification, cleanup
- `display.py` — CLI progress formatting (pure functions)

Entry points: `scripts/run_pipeline.py` (orchestrator) and `scripts/run_stream.py` (training).

### Source (`src/`)

- `datasets/SANPO__dataset.py` — PyTorch Dataset for RGB + segmentation + depth
- `metrics/panoptic_quality.py` — PQ/SQ/RQ metric with dataclasses
- `utils/SANPO_data_processor.py` — `image_crop()` (1242x2208 → 3 overlapping 640x640 patches), `process_and_patch_sanpo_depth()`

### Config (`configs/`)

- `pipeline_config.yaml` — all pipeline paths, remote storage, batch sizes, preprocessing toggles
- `decimation_config.yaml` — frame sampling interval/offset
- `sanpo_class_mapping.yaml` — 30-class semantic/panoptic type labels

## Key Design Decisions

- **SDD (Spec-Driven Development)**: specs live in `docs/` — write spec first, implement after review
- **Lazy imports**: cv2, webdataset, SANPO_data_processor are imported inside functions so unit tests run without heavy deps
- **Per-camera validation**: each camera (chest/head) validated independently; session skipped only if ALL cameras fail
- **Batch-level cleanup**: shards may span sessions, so cleanup happens after entire batch is verified
- **Config-driven extensibility**: `enable_image_crop` / `enable_depth_process` toggles in pipeline_config.yaml let test-set processing skip cropping
- **Session state machine**: `pending → downloaded → validated → decimated → processed → packed → uploaded → verified → cleaned` (plus `skipped` and `error`)

## CI

GitHub Actions on `dev` branch: ruff lint + pytest. CI uses CPU-only PyTorch. Dependencies in `.github/workflows/requirements.txt`.
