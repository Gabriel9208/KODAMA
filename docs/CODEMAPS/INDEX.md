<!-- Generated: 2026-04-08 | Updated from codemaps/ | Token budget: ~150 -->

# KODAMA Codemaps Index

**Last Updated:** 2026-04-08  
**Scope:** Single-repo ML pipeline for panoptic segmentation (SANPO dataset)  
**Python Version:** 3.11 | **PyTorch:** 2.2.0 | **CUDA:** 12.1

## Quick Navigation

| Codemap | Purpose | Coverage |
|---------|---------|----------|
| [architecture.md](./architecture.md) | System overview, data flow, state machine | High-level design |
| [pipeline.md](./pipeline.md) | Data pipeline modules, entry points, functions | `scripts/data_pipeline/`, entry points |
| [model.md](./model.md) | Model architecture, feature extraction, semantic decoder | `src/models/`, `model/` |
| [data.md](./data.md) | SANPO dataset format, WebDataset shards, preprocessing | Data structures, configs |
| [dependencies.md](./dependencies.md) | External services (GCS, GDrive), libraries | Infrastructure |

## Repository Structure

```
KODAMA/
├── scripts/
│   ├── run_pipeline.py          # Windows: Download→Pack→Upload orchestrator
│   ├── run_stream.py            # Ubuntu P100: Mount GDrive + WebDataset stream
│   ├── extract_yolo_format.py   # Convert SANPO masks → YOLO seg labels
│   ├── split_dataset.py         # Assign cleaned sessions to train/val shards
│   ├── yolo26_explore.py        # YOLO backbone layer inspection
│   └── data_pipeline/           # Modular pipeline package (11 modules)
│       ├── config.py, progress.py, download.py, validate.py
│       ├── decimate.py, preprocess.py, pack.py, upload.py
│       └── display.py
├── src/
│   ├── datasets/SANPO__dataset.py    # PyTorch Dataset (RGB+seg+depth)
│   ├── metrics/panoptic_quality.py   # PQ/SQ/RQ calculation
│   ├── models/feature_extractor.py   # Hook-based feature extraction
│   └── utils/SANPO_data_processor.py # Image crop, depth processing
├── model/
│   ├── yolo26n-seg.pt           # Frozen YOLO backbone checkpoint
│   ├── yolo26-seg.yaml          # YOLO config
│   ├── feature_extractor.py     # Forward hook registration (layers 16/19/22)
│   └── semantic_decoder.py      # FPN-based semantic decoder (NEW)
├── configs/
│   ├── pipeline_config.yaml           # All paths, remote storage, batch sizes
│   ├── decimation_config.yaml         # Frame sampling interval/offset
│   ├── sanpo_class_mapping.yaml       # 30 semantic/panoptic class labels
│   ├── sanpo_yolo_label_mapping.yaml  # SANPO→YOLO class remapping
│   └── split_config.yaml              # Train/val split ratios
├── data/
│   ├── pipeline_progress.json         # Session status + shard assignments
│   ├── sanpo_dataset_v0_labelmap.json # Label lookup
│   ├── sanpo_dataset_v0_labeltype.json# Label types
│   └── *_session_ids.txt              # Train/test session IDs
├── tests/
│   ├── scripts/test_pipeline.py       # Pipeline validation tests
│   ├── metrics/test_panoptic_quality.py
│   └── datasets/test_SANPO__dataset.py
└── docs/
    ├── CODEMAPS/          # Architecture documentation (this folder)
    ├── data_process_pipeline_sdd.md
    ├── TECH.md, plan.md, REF.md
    └── notes/
```

## Key Metrics

| Metric | Value |
|--------|-------|
| Total Python LOC | ~1650 |
| Largest module | `download.py` (252 lines) |
| Data pipeline modules | 11 |
| External integrations | GCS, Google Drive (rclone) |
| Tests | 3 files |
| Session states | 9 (`pending` → `cleaned`, `skipped`, `error`) |

## Recent Changes (2026-04-04 → 2026-04-08)

- **Added:** `model/semantic_decoder.py` (71 lines) — FPN-based decoder with UpsampleBlocks
- **Added:** `data/` folder with `pipeline_progress.json`, label mappings, session ID lists
- **Fixed:** Data pipeline logging and pre-download logic (commit 144f989)
- **Fixed:** Remove `shell=True` in download.py (OS-specific change, commit d63e4ee)

## Related Documentation

- **Spec:** `docs/data_process_pipeline_sdd.md` — Spec-Driven Development document
- **Tech:** `docs/TECH.md` — Technical decisions
- **Plan:** `docs/plan.md` — Development roadmap
- **Notes:** `docs/notes/` — Research and exploration

## Revision History

| Date | Changes | Commit |
|------|---------|--------|
| 2026-04-08 | Added INDEX.md, updated pipeline.md + model.md with semantic_decoder.py | Latest |
| 2026-04-04 | Initial codemap generation (architecture, pipeline, data, dependencies) | — |
