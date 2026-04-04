<!-- Generated: 2026-04-04 | Files scanned: 23 | Token estimate: ~700 -->

# Pipeline & Source Modules

## Entry Points

| Script | Env | Purpose |
|---|---|---|
| `scripts/run_pipeline.py` | Windows | Orchestrate download→pack→upload cycle |
| `scripts/run_stream.py` | Ubuntu P100 | Mount GDrive + stream WebDataset for training |
| `scripts/extract_yolo_format.py` | Any | Convert packed shards → YOLO seg label txts |
| `scripts/split_dataset.py` | Any | Assign cleaned sessions to train/val shards |
| `scripts/yolo26_explore.py` | Any | YOLO backbone layer inspection (one-shot script) |

## data_pipeline Package (`scripts/data_pipeline/`)

| Module | Lines | Key exports |
|---|---|---|
| `config.py` | 39 | `load_pipeline_config(path)` — loads YAML, resolves relative paths, cached |
| `progress.py` | 86 | `load_progress`, `save_progress`, `load_session_ids`, `_validate_decimation_config` |
| `download.py` | 202 | `download_batch(progress, config)` — gcloud storage cp, disk space check |
| `validate.py` | 162 | `validate_batch(progress, config)` — per-camera file counts + frame ID alignment |
| `decimate.py` | 72 | `decimate_session()`, `_decimate_indices()` (pure function) |
| `preprocess.py` | 244 | `validate_decimate_preprocess(progress, config)` — orchestrates val+dec+preprocess |
| `pack.py` | 159 | `pack_shards(sessions, progress, config)`, `_get_batch_index()` |
| `upload.py` | 133 | `upload_shards`, `verify_shards`, `cleanup_batch` — rclone operations |
| `display.py` | 16 | `format_batch_range`, `format_resume_summary` — pure CLI formatting |

## src/ Modules

| Module | Lines | Purpose |
|---|---|---|
| `src/datasets/SANPO__dataset.py` | 45 | `SANPO__dataset(Dataset)` — loads RGB+seg+depth from preprocessed dirs |
| `src/metrics/panoptic_quality.py` | 102 | `calculate_pq(g, p)` → `PanopticQualityResult`; PQ/SQ/RQ per class |
| `src/utils/SANPO_data_processor.py` | 88 | `SANPO_data_processor.image_crop()`, `process_and_patch_sanpo_depth()` |

## model/ Module

| Module | Lines | Purpose |
|---|---|---|
| `model/feature_extractor.py` | 48 | Forward hook registration on YOLO layers 16/19/22 (P3/P4/P5) |

## YOLO Label Extraction (`scripts/extract_yolo_format.py`)

```
extract_yolo_format(mask, output_path)
  └─ sanpo_to_yolo_label(mask)     — remap class IDs via lookup_table
       └─ pixel_to_polygon(mask, output_path)
             ├─ instance_isolation(mask)     — (class_id, inst_id) → binary mask dict
             └─ contour_extraction(masks)    — cv2 contour → normalized YOLO polygon
```

## run_stream.py Key Functions

```
train(args)
  ├─ mount_gdrive(mount_point, gdrive_remote)
  ├─ create_dataset(mount_point, batch_size, shuffle_buffer) → wds.WebDataset
  │     └─ seg_decoder() — decode _seg.png as raw uint8 (bypasses float normalization)
  └─ create_dataloader(dataset, num_workers) → DataLoader

decode_seg_mask(seg) → (semantic, instance)
  semantic = seg[:,:,0]       # R channel = class ID
  instance = G*256 + B        # packed instance ID
```
