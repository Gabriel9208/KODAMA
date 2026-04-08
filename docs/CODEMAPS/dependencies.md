<!-- Generated: 2026-04-08 | Files scanned: 28 | Token estimate: ~450 -->

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

## Model Architecture

| Component | File | Purpose |
|-----------|------|---------|
| YOLO Backbone (Frozen) | `model/yolo26n-seg.pt` | Input feature extraction (layers 16/19/22 → P3/P4/P5) |
| Feature Extractor | `src/models/feature_extractor.py` | Hook-based capture of P3, P4, P5 tensors |
| Semantic Decoder | `model/semantic_decoder.py` | FPN fusion + semantic segmentation head |

**YOLO Layers:**
- Layer 16 (P3): 80×60, 64 channels
- Layer 19 (P4): 40×30, 128 channels  
- Layer 22 (P5): 20×15, 256 channels

**Decoder output:** (batch, 30 classes, 640, 480)
