<!-- Generated: 2026-04-04 | Files scanned: 23 | Token estimate: ~500 -->

# Data Architecture

## SANPO Dataset Structure (raw, per session)

```
<session_id>/
  camera_chest/
    left/
      video_frames/       *.png  (1242×2208 RGB)
      segmentation_masks/ *.png  (RGB-encoded: R=class_id, G=inst_hi, B=inst_lo)
      depth_maps/         *.gz   (float16, 720×1280, 4-byte header)
  camera_head/
    left/  (same structure)
```

## WebDataset Shard Format (packed)

Each `.tar` shard contains samples with keys:
- `<key>.png`       — RGB frame
- `<key>._seg.png`  — segmentation mask (uint8 RGB)
- `<key>._depth.npy`— depth map (float16)
- `<key>.json`      — metadata

## YOLO Export Format

```
data/
  images/train/   *.png
  labels/train/   *.txt  (YOLO seg: <class_id> <x1> <y1> ... per line)
```

## Segmentation Mask Encoding

```
R channel → semantic class ID (0–30)
G channel → instance ID high byte
B channel → instance ID low byte
instance_id = G * 256 + B
```

## Config Files

| File | Purpose |
|---|---|
| `configs/pipeline_config.yaml` | All pipeline paths, remote storage, preprocessing toggles |
| `configs/decimation_config.yaml` | Frame sampling interval / offset |
| `configs/sanpo_class_mapping.yaml` | 30-class semantic/panoptic type labels |
| `configs/sanpo_yolo_label_mapping.yaml` | SANPO class → YOLO label remapping |
| `configs/split_config.yaml` | Train/val split ratios, seed, output shard lists |

## Progress State File

`data/pipeline_progress.json` — tracks per-session status + shard file assignments.
Read/written atomically via `scripts/data_pipeline/progress.py`.

## Preprocessing: Image Patch Extraction

`SANPO_data_processor.image_crop()` — 1242×2208 → 3 overlapping 640×640 patches (left/center/right).
`SANPO_data_processor.process_and_patch_sanpo_depth()` — 720×1280 float16.gz → 3×640×640 .npy.
