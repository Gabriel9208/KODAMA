<!-- Generated: 2026-04-04 | Files scanned: 23 | Token estimate: ~400 -->

# Dependencies

## Runtime (training / streaming)

| Package | Version | Purpose |
|---|---|---|
| Python | 3.11 | Language runtime |
| PyTorch | 2.2.0 | Tensors, DataLoader |
| CUDA | 12.1 | GPU compute (Ubuntu P100) |
| ultralytics | latest | YOLO backbone (yolo26n-seg.pt) |
| webdataset | latest | Shard streaming |
| opencv-python (cv2) | latest | Image I/O, patch extraction |
| numpy | latest | Array ops, mask decoding |
| Pillow | latest | WebDataset image decode |
| scikit-learn | latest | `train_test_split` in split_dataset.py |
| PyYAML | latest | Config loading |

## Pipeline Infrastructure

| Tool | Purpose |
|---|---|
| `gcloud storage cp` | Download from GCS bucket |
| `rclone` | Upload to Google Drive, mount for streaming |

## Dev / CI

| Tool | Purpose |
|---|---|
| ruff | Linting |
| pytest | Test runner |
| conda (KODAMA env) | Environment isolation |

## External Services

| Service | Role |
|---|---|
| GCS `gs://gresearch/sanpo_dataset/v0/sanpo-real` | Raw data source (read-only) |
| Google Drive `gdrive:SANPO-Dataset/shards/` | Packed shard storage + streaming source |

## Model Checkpoint

`model/yolo26n-seg.pt` — frozen YOLO backbone. Feature extraction hooks registered at layers 16 (P3), 19 (P4), 22 (P5).
