<!-- Generated: 2026-04-04 | Files scanned: 23 | Token estimate: ~600 -->

# KODAMA Architecture

## Project Type
Single-repo ML pipeline — data ingestion + model research for panoptic segmentation on SANPO dataset.

## High-Level Data Flow

```
GCS (sanpo_dataset/v0/sanpo-real)
  └─► download_batch()          [scripts/data_pipeline/download.py]
        └─► validate_decimate_preprocess()  [scripts/data_pipeline/preprocess.py]
              ├─ validate_batch()    [validate.py]  — per-camera file counts + frame alignment
              ├─ decimate_session()  [decimate.py]  — deterministic frame sampling
              └─ preprocess()        [preprocess.py] — crop 1242×2208 → 3×640×640 patches
                    └─► pack_shards()         [scripts/data_pipeline/pack.py]
                          └─► upload_shards() [scripts/data_pipeline/upload.py]
                                └─► Google Drive (rclone)
                                      └─► run_stream.py — WebDataset training loop
```

## Session State Machine

```
pending → downloaded → validated → decimated → processed → packed → uploaded → verified → cleaned
                                                                                        └─► skipped
                                                                                        └─► error
```

## System Boundaries

| Component | Purpose |
|---|---|
| `scripts/run_pipeline.py` | Batch orchestrator (Windows, GCS → GDrive) |
| `scripts/run_stream.py` | Training streamer (Ubuntu P100, mounts GDrive) |
| `scripts/data_pipeline/` | Modular pipeline package |
| `src/` | PyTorch dataset, metrics, preprocessing utils |
| `model/` | YOLO backbone exploration / feature extraction |
| `scripts/extract_yolo_format.py` | Convert SANPO masks → YOLO segmentation labels |
| `scripts/split_dataset.py` | Train/val shard split from pipeline progress |

## Key Design Constraints

- Lazy imports (cv2, webdataset, SANPO_data_processor) — keep tests fast
- Batch-level cleanup — shards may span sessions
- Per-camera validation — chest/head validated independently
- Config-driven toggles — `enable_image_crop`, `enable_depth_process`
