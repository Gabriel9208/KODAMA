# Phase 6: Modular Refactoring SDD

> **Goal**: Refactor the monolithic `pipeline_pack_upload.py` (~1137 lines) into a modular package, externalize hardcoded config, and rename `pipeline_train_stream.py` for multi-split usage.

---

## 1. Motivation

| Problem | Impact |
|---------|--------|
| Single file contains all phases (0-3, 5) + utilities + orchestrator | Hard to navigate, review, and test independently |
| Hardcoded paths, constants, GCS bucket, GDrive remote | Cannot reuse for test set or different dataset without editing source |
| `_preprocess_camera` always calls `image_crop` + `process_and_patch_sanpo_depth` | Test set may require different preprocessing (e.g., no cropping) |
| `pipeline_train_stream.py` name implies train-only | Will also serve test set streaming |
| `load_session_ids()` reads a hardcoded train split file path | Cannot switch to test split |

---

## 2. Target Structure

```
scripts/
├── data_pipeline/               # Package: data processing pipeline
│   ├── __init__.py
│   ├── config.py                # Config loader (reads pipeline_config.yaml)
│   ├── progress.py              # load_progress(), save_progress(), state machine helpers
│   ├── download.py              # download_session(), GCS helpers
│   ├── validate.py              # _validate_camera(), validate_session(), _discover_cameras()
│   ├── decimate.py              # _decimate_indices(), decimate_session()
│   ├── preprocess.py            # _preprocess_camera(), preprocess_session() (lazy imports)
│   ├── pack.py                  # pack_shards(), shard helpers
│   ├── upload.py                # upload_shards(), verify_shards(), cleanup_batch()
│   └── display.py               # format_batch_range(), format_resume_summary()
├── run_pipeline.py              # Thin entry point: parse args → load config → orchestrate
└── run_stream.py                # Renamed from pipeline_train_stream.py
```

### 2.1 Module Responsibilities

| Module | Content | Depends On |
|--------|---------|------------|
| `config.py` | `load_pipeline_config()` — reads `configs/pipeline_config.yaml`, returns typed dict | `yaml` |
| `progress.py` | `load_progress()`, `save_progress()`, `_now_iso()`, `_available_disk_gb()`, `load_session_ids()` | `config.py` |
| `download.py` | `_gcloud_ls()`, `_gcloud_cp()`, `_integrity_check_post_download()`, `_delete_raw_session()`, `download_session()`, `download_batch()` | `config.py`, `progress.py` |
| `validate.py` | `_extract_frame_id()`, `_discover_cameras()`, `_validate_camera()`, `_delete_camera_data()`, `validate_session()` | `config.py`, `progress.py` |
| `decimate.py` | `_decimate_indices()`, `decimate_session()` | `config.py`, `progress.py`, `validate.py` (for `_extract_frame_id`) |
| `preprocess.py` | `_preprocess_camera()`, `_integrity_check_post_preprocess()`, `preprocess_session()`, `validate_decimate_preprocess()` | `config.py`, `progress.py`, `validate.py`, `decimate.py` |
| `pack.py` | `_parse_processed_filename()`, `_get_batch_index()`, `_increment_batch_index()`, `pack_shards()` | `config.py`, `progress.py` |
| `upload.py` | `upload_shards()`, `verify_shards()`, `cleanup_batch()` | `config.py`, `progress.py`, `pack.py` (for `_increment_batch_index`) |
| `display.py` | `format_batch_range()`, `format_resume_summary()` | None (pure functions) |
| `run_pipeline.py` | `main()` + `parse_args()` | All `data_pipeline.*` modules |
| `run_stream.py` | Mount + WebDataset streaming | `config.py` (for GDrive remote) |

### 2.2 Naming Convention

- Public functions: no underscore prefix (e.g., `validate_session`, `download_batch`)
- Module-internal helpers: single underscore prefix (e.g., `_gcloud_ls`, `_extract_frame_id`)
- Constants: defined in `config.py` via YAML, not as module-level globals

---

## 3. Config Externalization

### 3.1 New File: `configs/pipeline_config.yaml`

```yaml
# =========================================================================
# Pipeline Configuration
# =========================================================================

# --- Paths (relative to project root) ---
paths:
  data_dir: "data"
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  shards_dir: "data/shards"
  progress_file: "data/pipeline_progress.json"
  session_ids_file: "data/sanpo_dataset_v0_sanpo-real_splits_train_session_ids.txt"
  decimation_config_file: "configs/decimation_config.yaml"

# --- Remote Storage ---
remote:
  gcs_bucket: "gs://gresearch/sanpo_dataset/v0/sanpo-real"
  gdrive_remote: "gdrive:SANPO-Dataset/shards/"

# --- Phase 0: Download ---
download:
  batch_size: 4
  disk_safety_factor: 1.5
  estimated_session_size_gb: 8

# --- Phase 1: Preprocessing ---
preprocess:
  cameras: ["camera_chest", "camera_head"]
  patch_positions: ["left", "center", "right"]
  enable_image_crop: true      # Set false to skip crop+resize (e.g., for test set)
  enable_depth_process: true   # Set false to skip depth processing

# --- Phase 2: Packing ---
packing:
  shard_max_size: 1.5e9  # bytes, ~1.5 GB per shard

# --- Phase 3: Upload ---
upload:
  rclone_transfers: 4
  rclone_chunk_size: "128M"

# --- Phase 4: Streaming ---
streaming:
  mount_point: "~/gdrive/shards"
  batch_size: 8
  shuffle_buffer: 2000
  num_workers: 4
```

### 3.2 Config Loader (`data_pipeline/config.py`)

```python
"""Pipeline configuration loader."""

from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

_config_cache: dict | None = None

def load_pipeline_config(config_path: Path | None = None) -> dict:
    """
    Load pipeline_config.yaml and resolve all relative paths to absolute.

    Caches the result — subsequent calls return the same dict.
    """
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "pipeline_config.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Resolve relative paths to absolute
    for key, value in cfg.get("paths", {}).items():
        cfg["paths"][key] = str(PROJECT_ROOT / value)

    _config_cache = cfg
    return cfg
```

**Key design decisions:**

1. **Module-level cache**: `load_pipeline_config()` is called once; all modules share the same config dict. No global constants.
2. **Path resolution**: Relative paths in YAML are resolved to absolute at load time.
3. **Testability**: `config_path` parameter allows tests to inject a custom config file.

### 3.3 What Moves to Config vs. What Stays in Code

| Currently hardcoded | Moves to YAML | Stays in code |
|---------------------|---------------|---------------|
| `PROJECT_ROOT` | — | `config.py` (computed from `__file__`, 向上三層: `config.py` → `data_pipeline/` → `scripts/` → project root) |
| `DATA_DIR`, `RAW_DIR`, `PROCESSED_DIR`, `SHARDS_DIR` | `paths.*` | — |
| `PROGRESS_FILE`, `SESSION_IDS_FILE`, `DECIMATION_CONFIG_FILE` | `paths.*` | — |
| `GCS_BUCKET`, `GDRIVE_REMOTE` | `remote.*` | — |
| `BATCH_SIZE`, `DISK_SAFETY_FACTOR`, `ESTIMATED_SESSION_SIZE_GB` | `download.*` | — |
| `SHARD_MAX_SIZE` | `packing.shard_max_size` | — |
| `CAMERAS`, `PATCH_POSITIONS` | `preprocess.*` | — |
| Logging setup | — | Each module's own logger (standard pattern) |
| State machine transitions | — | `progress.py` (logic, not config) |

---

## 4. Extensibility Design (Train vs. Test)

### 4.1 Problem

Test set processing may differ:
- Different session IDs file (`test_session_ids.txt`)
- May skip image cropping (no `image_crop`)
- May skip depth processing
- Different GDrive destination

### 4.2 Solution: Config-Driven Behavior

The `preprocess` section in `pipeline_config.yaml` controls which processing steps run:

```yaml
preprocess:
  enable_image_crop: true      # false → skip crop+resize, copy raw frames directly
  enable_depth_process: true   # false → skip depth processing
```

**Implementation in `preprocess.py`:**

```python
def _preprocess_camera(session_id, camera, raw_base, selected_frames,
                       selected_segs, selected_depths, config):
    preprocess_cfg = config["preprocess"]

    if preprocess_cfg["enable_image_crop"]:
        # Current behavior: SANPO_data_processor.image_crop(...)
        ...
    else:
        # Copy raw files directly to processed dir (no crop/resize)
        ...

    if preprocess_cfg["enable_depth_process"]:
        # Current behavior: SANPO_data_processor.process_and_patch_sanpo_depth(...)
        ...
    else:
        # Copy raw depth files directly
        ...
```

### 4.3 Multi-Split Workflow

To process a different split, the user only needs a different config file:

```bash
# Train (default)
python scripts/run_pipeline.py

# Test (custom config)
python scripts/run_pipeline.py --config configs/pipeline_config_test.yaml
```

The entry point `run_pipeline.py` adds `--config` CLI argument:

```python
parser.add_argument(
    "--config",
    type=str,
    default=None,  # → uses default configs/pipeline_config.yaml
    help="Path to pipeline config YAML",
)
```

---

## 5. Rename: `pipeline_train_stream.py` → `run_stream.py`

### 5.1 Rationale

The streaming script will be used for both train and test sets. The name `pipeline_train_stream` incorrectly implies train-only.

### 5.2 Changes

| Before | After |
|--------|-------|
| `scripts/pipeline_train_stream.py` | `scripts/run_stream.py` |
| Hardcoded `DEFAULT_MOUNT_POINT = ~/gdrive/shards` | Read from `config["streaming"]["mount_point"]` |
| Hardcoded `GDRIVE_REMOTE` | Read from `config["remote"]["gdrive_remote"]` |
| No `--config` argument | Add `--config` argument |

### 5.3 Updated CLI

```bash
# Train streaming (default config)
python scripts/run_stream.py

# Test streaming (custom config)
python scripts/run_stream.py --config configs/pipeline_config_test.yaml
```

---

## 6. Migration Strategy

### 6.1 Constraints

- All 22 existing tests must continue to pass
- No behavioral changes — pure structural refactoring
- Progress file format unchanged

### 6.2 Steps

| Step | Action | Verify |
|------|--------|--------|
| 1 | Create `configs/pipeline_config.yaml` with current hardcoded values | YAML loads correctly |
| 2 | Create `scripts/data_pipeline/__init__.py` and `config.py` | `load_pipeline_config()` returns expected dict |
| 3 | Extract `progress.py` from current helpers | Import works |
| 4 | Extract `display.py` (pure functions, no dependencies) | Existing Phase 5 tests pass |
| 5 | Extract `validate.py` + `decimate.py` | Existing validation + decimation tests pass |
| 6 | Extract `download.py` | Import works |
| 7 | Extract `preprocess.py` | Import works |
| 8 | Extract `pack.py` + `upload.py` | Import works |
| 9 | Create `scripts/run_pipeline.py` as thin entry point | Full pipeline works |
| 10 | Rename `pipeline_train_stream.py` → `run_stream.py`, integrate config | Streaming works |
| 11 | Update test imports | All 22 tests pass |
| 12 | Delete old `scripts/pipeline_pack_upload.py` and `scripts/pipeline_train_stream.py` | No leftover files |

### 6.3 Test Import Updates

Current test imports from:
```python
from pipeline_pack_upload import (
    _decimate_indices,
    _extract_frame_id,
    _validate_camera,
    format_batch_range,
    format_resume_summary,
)
```

After refactoring, imports change to:
```python
from data_pipeline.decimate import _decimate_indices
from data_pipeline.validate import _extract_frame_id, _validate_camera
from data_pipeline.display import format_batch_range, format_resume_summary
```

**Alternatively**, re-export from `data_pipeline/__init__.py` for backward compatibility:
```python
# scripts/data_pipeline/__init__.py
from .decimate import _decimate_indices
from .validate import _extract_frame_id, _validate_camera
from .display import format_batch_range, format_resume_summary
```

**Recommendation**: Use direct imports (first option). Clearer module ownership, no re-export maintenance burden.

---

## 7. Config Function Signature Changes

Functions that currently read module-level globals will receive `config` as a parameter.

### 7.1 Before → After

```python
# BEFORE (reads global)
def load_progress() -> dict:
    tmp_path = PROGRESS_FILE.with_suffix(".json.tmp")
    ...

# AFTER (receives config)
def load_progress(config: dict) -> dict:
    progress_file = Path(config["paths"]["progress_file"])
    tmp_path = progress_file.with_suffix(".json.tmp")
    ...
```

### 7.2 Affected Functions

| Function | New parameter |
|----------|--------------|
| `load_progress()` | `config` |
| `save_progress()` | `config` |
| `load_session_ids()` | `config` |
| `download_session()` | `config` |
| `download_batch()` | `config` |
| `validate_session()` | `config` |
| `decimate_session()` | `config` |
| `_preprocess_camera()` | `config` |
| `preprocess_session()` | `config` |
| `pack_shards()` | `config` |
| `upload_shards()` | `config` |
| `verify_shards()` | `config` |
| `cleanup_batch()` | `config` |

Pure functions with no config dependency remain unchanged:
- `_decimate_indices(total_frames, interval, offset)`
- `_extract_frame_id(filename)`
- `format_batch_range(batch_index, batch_size, total_sessions)`
- `format_resume_summary(progress, total_sessions)`
- `_now_iso()`
- `_available_disk_gb(path)`

---

## 8. Overview Document Update

After refactoring, update the Document Index table in `data_process_pipeline_sdd.md`:

```markdown
**Program Outputs:**

- Phases 0-3, 5 → `scripts/data_pipeline/` package + `run_pipeline.py` entry point (Windows)
- Phase 4 → `run_stream.py` (Ubuntu P100 / Windows)
```

---

## 9. What This SDD Does NOT Change

- **Progress file format**: Unchanged. Existing `pipeline_progress.json` files remain compatible.
- **State machine**: Same states, same transitions.
- **Decimation config**: `configs/decimation_config.yaml` remains separate (referenced by path in pipeline config).
- **SANPO_data_processor**: No changes to `src/utils/SANPO_data_processor.py`.
- **Phase logic**: All phase functions retain their current behavior. This is a structural refactoring only.
