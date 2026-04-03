# SDD Overview: SANPO Dataset Streaming Training Pipeline v2

> This document provides the architectural overview. For implementation details of each phase, refer to the corresponding standalone documents.

---

## Document Index

| File                                 | Content                                                                        | Execution Environment |
| ------------------------------------ | ------------------------------------------------------------------------------ | --------------------- |
| `00_overview.md` (this file)         | Architecture overview, Decimation design, Resume capability, Safety mechanisms | —                     |
| `01_phase0_download.md`              | Phase 0: Download session data from GCS                                        | Windows               |
| `02_phase1_decimation_preprocess.md` | Phase 1: Decimation + Data preprocessing                                       | Windows               |
| `03_phase2_3_pack_upload.md`         | Phase 2-3: Pack shards + Upload + Verify + Clean                               | Windows               |
| `04_phase4_train_stream.md`          | Phase 4: Mount Google Drive + Streaming training                               | Ubuntu P100           |
| `05_phase5_cli_progress.md`          | Phase 5: CLI progress display                                                  | Windows               |
| `06_phase6_refactoring.md`           | Phase 6: Modular refactoring + config externalization + rename                 | —                     |

**Program Outputs:**

- Phases 0-3, 5 → `scripts/data_pipeline/` package + `run_pipeline.py` entry point (Windows)
- Phase 4 → `scripts/run_stream.py` (Ubuntu P100 / Windows)

---

## 1. Background and Goals

The training dataset is SANPO-Real (560 sessions), stored in a public GCS bucket. Local and training machine storage is insufficient for the full dataset. The pipeline must:

1. Download required sessions from GCS (left side only)
2. Decimation: Sample frames by configurable parameters (reduce to ~1/10)
3. Preprocessing: Crop and resize to 640×640 (using existing `SANPO_data_processor`)
4. Pack into WebDataset shard format
5. Upload to Google Drive for long-term storage
6. Delete local data after verification passes
7. Stream from Google Drive during training without full download

---

## 2. Environment

| Role             | System                    | Purpose                                               |
| ---------------- | ------------------------- | ----------------------------------------------------- |
| Packing machine  | Windows (~97GB available) | Download, Decimation, Preprocess, Pack, Upload, Clean |
| Cloud storage    | Google Drive              | Long-term shard storage (~250GB)                      |
| Training machine | Ubuntu (2×P100 16GB)      | Mount cloud storage, Execute training                 |

---

## 3. Core Technology Choices

| Technology                    | Purpose                     | Rationale                                                                                                                                                         |
| ----------------------------- | --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| WebDataset (tar-based shards) | Packing format              | Merges many small files into large tars; native PyTorch streaming support; shard design enables parallel reads and shuffle. Target shard size: **1-2 GB** per tar |
| gcloud CLI                    | Download from GCS           | Official tool, supports public buckets                                                                                                                            |
| rclone                        | Google Drive upload / mount | Supports upload, check, and mount in one tool                                                                                                                     |

---

## 4. Full Pipeline Overview

```
GCS (gs://gresearch/sanpo_dataset/v0/sanpo-real/)
        ↓  Phase 0: gcloud storage cp (left side only)
Windows local (data\raw\)
        ↓  Phase 1 Step 0: Pre-Decimation Validation
        ↓    → FAIL: delete raw data, mark "skipped", free disk space
        ↓    → PASS: continue
        ↓  Phase 1a: Decimation (sample every N frames, configurable interval + offset)
        ↓  Phase 1b: Preprocessing (crop + resize to 640×640, using SANPO_data_processor)
Windows local (data\processed\)
        ↓  Phase 2: Pack into WebDataset shards
Windows local (data\shards\)
        ↓  Phase 3a: rclone copy upload
        ↓  Phase 3b: rclone check verification
        ↓  Phase 3c: Delete raw + processed + shards after verification passes
Google Drive (SANPO-Dataset/shards/)
        ↓  Phase 4a: rclone mount (Ubuntu P100)
        ↓  Phase 4b: WebDataset streaming training
~/gdrive/shards/ → GPU training
```

---

## 5. Decimation Design

### 5.1 Parameter Definition

```yaml
# configs/decimation_config.yaml
decimation:
  interval: 10 # Sample 1 frame every N frames
  offset: 0 # Start from which frame (0-indexed)
  # Sampling result: offset, offset+interval, offset+2*interval, ...
  # Default interval=10, offset=0 → frames 0, 10, 20, 30...
```

### 5.2 Sampling Logic

```
Input: Sorted frame list (natural sort by filename)
Sample: frames[offset], frames[offset + interval], frames[offset + 2*interval], ...

Example (interval=10, offset=0, 527 frames total):
  Selected frame indices: 0, 10, 20, 30, ... 520 → 53 frames
```

### 5.3 Key Rules

1. **Deterministic sampling**: No randomness. Same parameters always produce same results
2. **Three data types stay synchronized**: video_frames, segmentation_masks, depth_maps use identical frame indices
3. **Parameters recorded**: Decimation parameters written into each sample's metadata for traceability
4. **Left side only**: Each session processes only `camera_chest/left` and/or `camera_head/left`

### 5.4 Data Volume Estimate

> **Important:** Not all 560 train sessions have segmentation annotations. SANPO documentation states ~20% of real frames have panoptic masks. Sessions without complete seg + depth annotations will be filtered out in Phase 1 Step 0 (Pre-Decimation Validation). Actual valid session count may be significantly less than 560.

| Item                                            | Before Decimation                   | After Decimation (interval=10)              |
| ----------------------------------------------- | ----------------------------------- | ------------------------------------------- |
| Per session (~527 frames)                       | ~8 GB                               | ~800 MB                                     |
| 560 sessions (if all valid)                     | ~4.5 TB                             | ~450 GB (raw)                               |
| **Estimated valid sessions**                    | **~100-200 (TBD after validation)** | —                                           |
| After preprocessing (crop + resize)             | —                                   | TBD                                         |
| Google Drive requirement                        | ~4 TB                               | **Estimated ~50-120 GB**                    |
| Total training samples (if ~150 valid sessions) | —                                   | 150 × ~53 × 3 patches ≈ **~24,000 samples** |

> These estimates will be refined after the first batch of sessions is validated. Update this section with actual numbers.

---

## 6. Resume Capability Design

### 6.1 Progress Tracking File

`data\pipeline_progress.json` tracks the processing status of each session:

```json
{
  "decimation_config": {
    "interval": 10,
    "offset": 0
  },
  "sessions": {
    "session_abc123": {
      "status": "cleaned",
      "downloaded_at": "2026-03-28T10:00:00",
      "validated_at": "2026-03-28T10:05:00",
      "valid_cameras": ["camera_chest"],
      "total_frames": { "camera_chest": 527 },
      "decimated_at": "2026-03-28T10:15:00",
      "processed_at": "2026-03-28T10:30:00",
      "packed_at": "2026-03-28T10:45:00",
      "uploaded_at": "2026-03-28T11:00:00",
      "verified_at": "2026-03-28T11:05:00",
      "cleaned_at": "2026-03-28T11:06:00",
      "shard_files": ["shard-000003.tar", "shard-000004.tar"],
      "frame_count": 53,
      "patch_count": 159
    },
    "session_def456": {
      "status": "downloaded",
      "downloaded_at": "2026-03-28T12:00:00"
    },
    "session_ghi789": {
      "status": "skipped",
      "skip_reason": "all cameras failed validation (chest: seg missing, head: depth count mismatch)",
      "skipped_at": "2026-03-28T12:05:00"
    }
  }
}
```

### 6.2 Session State Machine

```
pending → downloaded → validated → decimated → processed → packed → uploaded → verified → cleaned
           ↘ error                ↘ skipped
```

- **`skipped`**: All cameras failed validation (missing seg/depth, count mismatch). Permanently excluded, never retried. Raw data is immediately deleted to free disk space. Note: validation is per-camera — individual cameras that fail are deleted, but the session is only `skipped` if no camera passes.
- **`error`**: Environment failure (network interruption, download failure, gcloud error). Retryable — on resume, treated the same as `pending` and re-downloaded.

Each state transition is written to the progress file immediately upon success.

### 6.3 Resume Logic

On script startup:

1. Read `pipeline_progress.json` (initialize if not found)
2. For all sessions, resume from the next step based on current status:
   - `cleaned` → Skip
   - `skipped` → Skip (permanently excluded — missing/misaligned seg or depth)
   - `verified` → Clean
   - `uploaded` → Verify
   - `packed` → Upload
   - `processed` → Pack
   - `decimated` → Preprocess
   - `validated` → Decimate
   - `downloaded` → Validate (Pre-Decimation Validation)
   - `error` → Download (retry)
   - `pending` → Download

### 6.4 Decimation Parameter Change Protection

If the `decimation_config` in the progress file differs from the current yaml:

- Script raises an error and aborts
- User must manually delete the progress file to proceed
- Prevents mixing different decimation parameters within the same shard set

---

## 7. Safety Mechanisms

### 7.1 Automatic Directory Creation

Before all I/O operations: `os.makedirs(target_dir, exist_ok=True)`. Applies to:

- `data\raw\{session_id}\...`
- `data\processed\{session_id}\...`
- `data\shards\`

### 7.2 Pre-Deletion Verification

**Unverified deletion is strictly forbidden.** Prerequisites:

- Corresponding shards have been uploaded to Google Drive
- `rclone check` passed (checksum match confirmed)
- Session status in `pipeline_progress.json` is `"verified"`

### 7.3 File Integrity Checks

| Timing                        | Check Content                                                                                                                                                                                                                                 |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| After download                | `video_frames/` exists and is non-empty                                                                                                                                                                                                       |
| **Pre-Decimation Validation** | **Per-camera check: seg_masks and depth_maps directories exist and non-empty; file counts match across all three types; frame IDs are aligned. FAIL → delete that camera's raw data only. Session marked `skipped` only if all cameras fail** |
| After decimation              | Sampled counts match across all three data types, indices match parameters                                                                                                                                                                    |
| After preprocessing           | Each frame produced 9 files (3 patches × 3 data types)                                                                                                                                                                                        |
| After packing                 | Each shard sample contains `.png` + `_seg.png` + `_depth.npy` + `.json`                                                                                                                                                                       |
| After upload                  | `rclone check` checksum comparison                                                                                                                                                                                                            |

### 7.4 Progress File Protection

- Write to `.tmp` first, then rename to overwrite (atomic write)
- On startup, if `.tmp` exists, the previous write was incomplete — use the last good version

### 7.5 Disk Space Check

Before downloading each new batch, check available space >= estimated requirement × 1.5. If insufficient, pause and prompt the user.

---

## 8. Storage Requirements

| Location      | Requirement                       | Notes                                                                          |
| ------------- | --------------------------------- | ------------------------------------------------------------------------------ |
| Windows local | Varies by batch                   | Peak: raw + processed + shard coexist; deleted after verification              |
| Google Drive  | ~50-120 GB (TBD after validation) | Long-term storage for all shards; much less than 560 sessions due to filtering |
| Ubuntu P100   | ~50 GB                            | rclone cache ceiling, does not grow                                            |

---

## 9. Prerequisites

**Windows packing machine:** gcloud CLI, rclone (gdrive remote configured), Python (webdataset, opencv-python, numpy, pyyaml), existing `SANPO_data_processor` code

**Ubuntu training machine:** rclone (gdrive remote configured), FUSE, Python (webdataset, torch, ultralytics)

---

## 10. Open Items

- [ ] Confirm camera_head vs camera_chest distribution across 560 sessions
- [ ] Confirm whether some sessions only have chest or only head
- [x] ~~Determine batch download size~~ → **3-4 sessions per batch** (~24-32 GB raw, peak ~40-50 GB with processed + shards)
- [ ] Confirm whether GCS download incurs egress fees
- [ ] Choose Google Drive storage plan
- [ ] Check Ubuntu P100 actual available disk space (determines rclone cache ceiling)
- [ ] Confirm decimation interval=10 is sufficient (can adjust later, but requires repacking all data)
- [ ] Confirm SANPO_data_processor's image_crop works on all session resolutions (currently hardcoded 2208×1242 and 720×1280)

---

The following are the step by step SDD

# Phase 0: Download Session Data from GCS

> Execution environment: Windows
> Prerequisite: `00_overview.md`
> Next step: `02_phase1_decimation_preprocess.md`

---

## Goal

Download SANPO-Real train sessions from Google Cloud Storage. Only download the left side of each camera for three data types.

---

## Input

| Item            | Path / Source                                                                                   |
| --------------- | ----------------------------------------------------------------------------------------------- |
| Session ID list | `data\sanpo_dataset_v0_sanpo-real_splits_train_session_ids.txt` (560 sessions, one ID per line) |
| GCS bucket      | `gs://gresearch/sanpo_dataset/v0/sanpo-real/`                                                   |

---

## Download Scope

**Left side only.** Segmentation masks and depth maps are only annotated for the left lens.

For each session, download three directories:

- `{session_id}/camera_chest/left/video_frames/` (`.png`)
- `{session_id}/camera_chest/left/segmentation_masks/` (`.png`)
- `{session_id}/camera_chest/left/depth_maps/` (`.float16.gz`)

> If a session contains `camera_head`, also download `camera_head/left/` with the same three directories.

**Do NOT download:** right side, zed_depth_maps, camera poses, or other metadata.

---

## Download Commands

```bash
# Download chest camera (left side)
gcloud storage cp -r gs://gresearch/sanpo_dataset/v0/sanpo-real/{session_id}/camera_chest/left/video_frames data\raw\{session_id}\camera_chest\left\
gcloud storage cp -r gs://gresearch/sanpo_dataset/v0/sanpo-real/{session_id}/camera_chest/left/segmentation_masks data\raw\{session_id}\camera_chest\left\
gcloud storage cp -r gs://gresearch/sanpo_dataset/v0/sanpo-real/{session_id}/camera_chest/left/depth_maps data\raw\{session_id}\camera_chest\left\

# If head camera exists (left side)
gcloud storage cp -r gs://gresearch/sanpo_dataset/v0/sanpo-real/{session_id}/camera_head/left/video_frames data\raw\{session_id}\camera_head\left\
gcloud storage cp -r gs://gresearch/sanpo_dataset/v0/sanpo-real/{session_id}/camera_head/left/segmentation_masks data\raw\{session_id}\camera_head\left\
gcloud storage cp -r gs://gresearch/sanpo_dataset/v0/sanpo-real/{session_id}/camera_head/left/depth_maps data\raw\{session_id}\camera_head\left\
```

---

## Download Strategy

Local available space is ~97GB. Each session's raw data is ~8GB. **Download in batches of 3-4 sessions** (~24-32 GB raw, peak ~40-50 GB with processed + shards):

```
Batch loop:
  1. Take the next 3-4 session IDs (skip sessions already "cleaned", "skipped", or "error" in progress)
  2. Check disk space >= estimated requirement × 1.5
  3. For each session in the batch:
     a. Check camera_head existence via: gcloud storage ls gs://.../{session_id}/camera_head/left/
     b. Download camera_chest/left/ (always)
     c. Download camera_head/left/ (only if step a confirmed it exists)
  4. → Proceed to Phase 1 (Decimation + Preprocessing)
  5. → Proceed to Phase 2-3 (Pack + Upload + Clean)
  6. Repeat until all sessions are processed
```

---

## Output Directory Structure

```
data\raw\{session_id}\
  ├── camera_chest\left\
  │   ├── video_frames\
  │   │   ├── 000000.png
  │   │   ├── 000001.png
  │   │   └── ... (~527 files)
  │   ├── segmentation_masks\
  │   │   ├── 000000.png
  │   │   └── ...
  │   └── depth_maps\
  │       ├── 000000.float16.gz
  │       └── ...
  └── camera_head\left\ (if available)
      └── ... (same structure)
```

---

## Integrity Check (run immediately after download)

**Note:** Not all SANPO-Real sessions have segmentation masks — only a subset has panoptic annotations. The download phase performs a basic check only. The strict validation (seg/depth existence, frame alignment) is performed in Phase 1 before decimation.

1. Confirm `video_frames/` directory exists and is non-empty
2. If `video_frames/` is missing or empty, mark session as `skipped` (confirmed missing data — never retry) and delete the raw data immediately

> Segmentation and depth validation is deferred to Phase 1 Step 0 (Pre-Decimation Validation). Sessions missing segmentation masks will be immediately deleted there to free disk space.

---

## Status Update

Download successful and basic check passed → update session in `pipeline_progress.json`:

```json
{
  "status": "downloaded",
  "downloaded_at": "2026-03-28T10:00:00"
}
```

---

## Prerequisites

- gcloud CLI installed
- `gcloud auth login` completed (SANPO is a public bucket, no project permissions needed)

---

## Error Handling

| Scenario                                        | Action                                                                                                |
| ----------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| Network interruption mid-download               | Mark as `error` (retryable). On resume, re-download from scratch. gcloud CLI handles partial transfer |
| gcloud CLI crash or unexpected exit code        | Mark as `error` (retryable). Log the exit code for diagnostics                                        |
| Session ID not found in GCS                     | Mark as `skipped` (permanent). Delete any partial download                                            |
| Insufficient disk space                         | Pause, prompt user to free space or wait for previous batch upload to complete                        |
| `video_frames/` missing or empty after download | Mark as `skipped` (permanent). Delete raw data immediately                                            |

---

# Phase 1: Decimation + Data Preprocessing

> Execution environment: Windows
> Prerequisite: `01_phase0_download.md`
> Next step: `03_phase2_3_pack_upload.md`

---

## Goal

Process downloaded raw session data in three steps:

1. **Pre-Decimation Validation**: Verify segmentation masks and depth maps exist, are complete, and align with video frames **per camera**. Cameras that fail validation are individually deleted to free disk space. A session is marked `skipped` only if **all** cameras fail.
2. **Decimation**: Sample frames by configurable parameters, reducing ~527 frames to ~53 frames
3. **Preprocessing**: Crop wide images into three 640×640 patches, using the existing `SANPO_data_processor`

---

## Input

| Item               | Source                                                      |
| ------------------ | ----------------------------------------------------------- |
| Raw data           | `data\raw\{session_id}\camera_chest\left\` (Phase 0 output) |
| Decimation config  | `configs\decimation_config.yaml`                            |
| Preprocessing tool | `SANPO_data_processor` (existing code)                      |

---

## Step 0: Pre-Decimation Validation

**Background:** Not all SANPO-Real sessions have segmentation annotations — only a subset (~20%) has panoptic masks. Cameras without complete annotations are useless for panoptic segmentation training and must be filtered out before any processing.

**Validation is per-camera, not per-session.** Each camera (chest, head) is validated independently. A camera that fails is deleted individually. The session is only marked `skipped` if all cameras fail.

### Validation Checks

For each camera within a downloaded session (`camera_chest/left`, `camera_head/left` if present):

```python
video_frames_dir = f"data/raw/{session_id}/{camera}/left/video_frames"
seg_masks_dir    = f"data/raw/{session_id}/{camera}/left/segmentation_masks"
depth_maps_dir   = f"data/raw/{session_id}/{camera}/left/depth_maps"

# --- Check 1: All three directories exist and are non-empty ---
if not os.path.isdir(seg_masks_dir) or len(os.listdir(seg_masks_dir)) == 0:
    # FAIL: No segmentation masks
    → delete and skip

if not os.path.isdir(depth_maps_dir) or len(os.listdir(depth_maps_dir)) == 0:
    # FAIL: No depth maps
    → delete and skip

# --- Check 2: File counts match ---
frame_files = sorted([f for f in os.listdir(video_frames_dir) if f.endswith(".png")])
seg_files   = sorted([f for f in os.listdir(seg_masks_dir) if f.endswith(".png")])
depth_files = sorted([f for f in os.listdir(depth_maps_dir) if f.endswith(".float16.gz")])

if not (len(frame_files) == len(seg_files) == len(depth_files)):
    # FAIL: File count mismatch
    → delete and skip

# --- Check 3: Frame alignment (filenames correspond) ---
# Extract the numeric prefix from each filename and compare
frame_ids = [extract_id(f) for f in frame_files]    # e.g. ["000000", "000001", ...]
seg_ids   = [extract_id(f) for f in seg_files]
depth_ids = [extract_id(f) for f in depth_files]

if frame_ids != seg_ids or frame_ids != depth_ids:
    # FAIL: Frames are not aligned
    → delete and skip
```

### On Camera Validation Failure

If any check fails for a camera:

1. **Log the reason** (which camera, which check failed, file counts, mismatched IDs)
2. **Delete only that camera's raw data**:
   ```python
   shutil.rmtree(f"data/raw/{session_id}/{camera}")
   ```
3. **Continue** to validate the next camera in the same session

### After All Cameras Validated

If **all cameras failed** → mark the session as `skipped`:

```json
{
  "status": "skipped",
  "skip_reason": "all cameras failed validation (chest: seg missing, head: depth count mismatch)",
  "skipped_at": "2026-03-28T10:05:00"
}
```

Delete any remaining raw data for the session: `shutil.rmtree(f"data/raw/{session_id}")`

If **at least one camera passed** → mark the session as `validated`:

```json
{
  "status": "validated",
  "validated_at": "2026-03-28T10:05:00",
  "valid_cameras": ["camera_chest"],
  "total_frames": { "camera_chest": 527 }
}
```

### Why Delete Immediately

- Raw data is ~8GB per session
- Local disk space is limited (~97GB)
- Invalid sessions waste space that could be used for the next batch download
- The original data remains on GCS and can be re-downloaded if needed

---

## Step 1a: Decimation

### Processing Logic

```python
# Read decimation parameters
interval = config["decimation"]["interval"]  # default 10
offset = config["decimation"]["offset"]      # default 0

# List and sort video_frames
all_frames = sorted(os.listdir(video_frames_dir))

# Sample
selected_indices = list(range(offset, len(all_frames), interval))
selected_frames = [all_frames[i] for i in selected_indices]

# Use the same indices for segmentation_masks and depth_maps
all_segs = sorted(os.listdir(seg_masks_dir))
all_depths = sorted(os.listdir(depth_maps_dir))
selected_segs = [all_segs[i] for i in selected_indices]
selected_depths = [all_depths[i] for i in selected_indices]
```

### Key Rules

- Sort by filename (natural sort) to ensure all three data types are aligned
- `selected_indices` is deterministic — no randomness
- Decimation only determines "which frames to process" — no files are moved or copied at this stage

### Integrity Check

- Confirm `len(selected_frames) == len(selected_segs) == len(selected_depths)`
- If mismatch, log error and mark session as `error`

### Status Update

Decimation complete → status updated to `"decimated"`

---

## Step 1b: Data Preprocessing

### Existing Code Dependencies

The `SANPO_data_processor` class provides two static methods:

**`image_crop(image_path, save_dir, file_prefix, interpolate)`**

- Input: SANPO wide image (2208×1242)
- Process: Crop into three overlapping square patches: left / center / right
- Each resized to 640×640
- Output: `{file_prefix}_left.png`, `{file_prefix}_center.png`, `{file_prefix}_right.png`

**`process_and_patch_sanpo_depth(gz_file_path, save_dir, file_prefix)`**

- Input: `.float16.gz` depth map (720×1280)
- Process: Decompress → float16 → float32 → nan_to_num → crop three patches → resize to 640×640
- Output: `{file_prefix}_left_float16.npy`, etc.

### Processing Logic

```python
for i, frame_filename in enumerate(selected_frames):
    frame_index = extract_index(frame_filename)  # e.g. "000000"

    # RGB: crop + resize (bilinear interpolation)
    SANPO_data_processor.image_crop(
        image_path=os.path.join(video_frames_dir, frame_filename),
        save_dir=processed_video_dir,
        file_prefix=frame_index,
        interpolate=cv2.INTER_LINEAR
    )

    # Segmentation: crop + resize (nearest interpolation to preserve labels)
    SANPO_data_processor.image_crop(
        image_path=os.path.join(seg_masks_dir, selected_segs[i]),
        save_dir=processed_seg_dir,
        file_prefix=frame_index,
        interpolate=cv2.INTER_NEAREST
    )

    # Depth: decompress + crop + resize
    SANPO_data_processor.process_and_patch_sanpo_depth(
        gz_file_path=os.path.join(depth_maps_dir, selected_depths[i]),
        save_dir=processed_depth_dir,
        file_prefix=frame_index
    )
```

### Output Per Frame

Each sampled frame produces **9 files** (3 patches × 3 data types):

| Patch  | RGB                | Segmentation       | Depth                      |
| ------ | ------------------ | ------------------ | -------------------------- |
| left   | `{idx}_left.png`   | `{idx}_left.png`   | `{idx}_left_float16.npy`   |
| center | `{idx}_center.png` | `{idx}_center.png` | `{idx}_center_float16.npy` |
| right  | `{idx}_right.png`  | `{idx}_right.png`  | `{idx}_right_float16.npy`  |

---

## Output Directory Structure

```
data\processed\{session_id}\camera_chest\left\
  ├── video_frames\
  │   ├── 000000_left.png      (640×640 RGB)
  │   ├── 000000_center.png
  │   ├── 000000_right.png
  │   ├── 000010_left.png
  │   ├── 000010_center.png
  │   ├── 000010_right.png
  │   └── ...
  ├── segmentation_masks\
  │   ├── 000000_left.png      (640×640 semantic+instance mask)
  │   ├── 000000_center.png
  │   └── ...
  └── depth_maps\
      ├── 000000_left_float16.npy   (640×640 float16)
      ├── 000000_center_float16.npy
      └── ...
```

---

## Integrity Check (after preprocessing)

1. Each sampled frame must produce 3 patch groups (left / center / right)
2. Each group must contain RGB + seg + depth (3 files)
3. Total files per sampled frame = 9
4. Total file count = `len(selected_frames) × 9`

---

## Status Update

Preprocessing complete and checks passed → status updated to `"processed"`:

```json
{
  "status": "processed",
  "processed_at": "2026-03-28T10:30:00",
  "frame_count": 53,
  "patch_count": 159
}
```

---

## Data Volume Estimate

> Actual numbers depend on how many sessions pass Pre-Decimation Validation. Not all 560 sessions have segmentation annotations.

| Item                                        | Count                           |
| ------------------------------------------- | ------------------------------- |
| Sampled frames per session                  | ~53 (527 / 10)                  |
| Patches per frame                           | 3 (left / center / right)       |
| Samples per session                         | ~159                            |
| Estimated valid sessions                    | ~100-200 (TBD after validation) |
| **Estimated total samples (if ~150 valid)** | **~24,000**                     |

> Update these numbers after running the first batch.

---

## Error Handling

| Scenario                                                | Action                                                                                                                       |
| ------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| **Validation: seg_masks directory missing or empty**    | **Log reason, delete that camera's raw data only, continue to next camera. Mark session `skipped` only if all cameras fail** |
| **Validation: depth_maps directory missing or empty**   | **Same as above — per-camera deletion, not per-session**                                                                     |
| **Validation: File count mismatch across three types**  | **Same as above — log counts, delete that camera only**                                                                      |
| **Validation: Frame IDs not aligned**                   | **Same as above — log mismatched IDs, delete that camera only**                                                              |
| `SANPO_data_processor` fails on a specific frame        | Log error, skip that frame, continue with others                                                                             |
| Image resolution doesn't match expected (not 2208×1242) | `image_crop` raises ValueError internally; log and skip                                                                      |
| Depth file corrupted                                    | `process_and_patch_sanpo_depth` raises IOError internally; log and skip                                                      |
| Post-preprocessing file count doesn't match expected    | Log warning but continue (allow minor gaps)                                                                                  |

---

# Phase 2-3: Pack Shards + Upload + Verify + Clean

> Execution environment: Windows
> Prerequisite: `02_phase1_decimation_preprocess.md`
> Next step: `04_phase4_train_stream.md`

---

## Goal

Pack preprocessed data into WebDataset shards, upload to Google Drive, verify integrity, then clean up local files. **All operations are batch-level** — a batch of sessions shares one set of shards, and cleanup occurs only after the entire batch is verified.

---

## Batch Lifecycle

```
data\shards\ is empty
    ↓  Phase 2: Pack all batch sessions into shards
data\shards\ has shard-XXX-YYYYYY.tar files
    ↓  Phase 3a: rclone copy → Google Drive
    ↓  Phase 3b: rclone check --one-way
    ↓  Phase 3c: Delete raw + processed + shards
data\shards\ is empty → ready for next batch
```

`data\shards\` must be empty before packing. If not, the pipeline aborts — previous batch cleanup is incomplete.

---

## Phase 2: Pack into WebDataset Shards

### Input

| Item              | Source                                              |
| ----------------- | --------------------------------------------------- |
| Preprocessed data | `data\processed\{session_id}\...\` (Phase 1 output) |
| Progress file     | `data\pipeline_progress.json`                       |

### Shard Specification

| Item          | Spec                                    |
| ------------- | --------------------------------------- |
| Shard size    | ~1.5 GB target (`maxsize=1.5e9`)        |
| Naming format | `shard-{batch_index:03d}-{seq:06d}.tar` |
| Output path   | `data\shards\`                          |

> Batch index prevents name collisions on Google Drive across batches.

### Sample Format (inside tar)

Each sample = four files sharing one `__key__`:

| File              | Content                      |
| ----------------- | ---------------------------- |
| `{key}.png`       | RGB (640×640)                |
| `{key}_seg.png`   | Segmentation mask (640×640)  |
| `{key}_depth.npy` | Depth map (640×640, float16) |
| `{key}.json`      | Metadata                     |

**Key format:** `{session_id}_{camera}_{frame_index}_{patch}`
**Example:** `abc123_chest_000000_left`

### Metadata `.json` Content

```json
{
  "session_id": "abc123",
  "camera": "camera_chest",
  "side": "left",
  "frame_index": 0,
  "patch": "left",
  "original_filename": "000000.png",
  "decimation_interval": 10,
  "decimation_offset": 0
}
```

### Packing Logic

```python
import webdataset as wds

assert len(os.listdir("data/shards/")) == 0, "data/shards/ must be empty before packing"

shard_pattern = f"data/shards/shard-{batch_index:03d}-%06d.tar"

with wds.ShardWriter(shard_pattern, maxsize=1.5e9) as sink:
    for session_id in batch_sessions:
        for camera in progress["sessions"][session_id]["valid_cameras"]:
            processed_dir = f"data/processed/{session_id}/{camera}/left"
            camera_short = camera.replace("camera_", "")

            for rgb_file in sorted(glob(f"{processed_dir}/video_frames/*_*.png")):
                frame_index, patch = parse_filename(rgb_file)
                key = f"{session_id}_{camera_short}_{frame_index}_{patch}"

                seg_file = f"{processed_dir}/segmentation_masks/{frame_index}_{patch}.png"
                depth_file = f"{processed_dir}/depth_maps/{frame_index}_{patch}_float16.npy"

                if not (os.path.exists(seg_file) and os.path.exists(depth_file)):
                    logger.warning(f"Missing files for {key}, skipping")
                    continue

                sample = {
                    "__key__": key,
                    "png": open(rgb_file, "rb").read(),
                    "_seg.png": open(seg_file, "rb").read(),
                    "_depth.npy": open(depth_file, "rb").read(),
                    "json": json.dumps(metadata).encode()
                }
                sink.write(sample)
```

### Integrity Check

1. Read each shard, confirm every sample contains `.png`, `_seg.png`, `_depth.npy`, `.json`
2. Confirm total sample count matches sum of `patch_count` across batch sessions
3. Record shard file list in progress for each session

### Status Update

All batch sessions → `"packed"`, with `shard_files` recorded

---

## Phase 3a: Upload

```bash
rclone copy data\shards\ gdrive:SANPO-Dataset/shards/ --transfers 4 --drive-chunk-size 128M --progress
```

Since `data\shards\` only contains the current batch (previous batch was cleaned), this uploads exactly what is needed.

All batch sessions → `"uploaded"`

---

## Phase 3b: Verify

```bash
rclone check data\shards\ gdrive:SANPO-Dataset/shards/ --one-way
```

`--one-way`: Only checks local files exist on remote with matching checksums. Previous batches' shards on remote are ignored.

- Return code 0 → pass
- Return code non-zero → re-upload failed shards, re-verify

All batch sessions → `"verified"`

---

## Phase 3c: Batch-Level Cleanup

### Safety Prerequisites

**ALL conditions must be met:**

1. Every session in the batch has status `"verified"`
2. `rclone check` passed in the current execution (not historical)
3. Paths to delete are confirmed under `data\raw\`, `data\processed\`, or `data\shards\`

### Cleanup

```python
# Per session
for session_id in batch_sessions:
    shutil.rmtree(f"data/raw/{session_id}", ignore_errors=True)
    shutil.rmtree(f"data/processed/{session_id}", ignore_errors=True)

# Batch shards (all at once)
for f in glob("data/shards/shard-*.tar"):
    os.remove(f)
```

All batch sessions → `"cleaned"`. `data\shards\` is now empty for the next batch.

---

## Error Handling

| Scenario                                | Action                                              |
| --------------------------------------- | --------------------------------------------------- |
| `data\shards\` not empty before packing | Abort — previous batch cleanup incomplete           |
| Missing file during packing             | Log warning, skip that sample, continue             |
| `rclone copy` interrupted               | Re-run; rclone skips already-uploaded files         |
| `rclone check` fails                    | Do NOT clean up; re-upload failed shards, re-verify |
| Cleanup target already missing          | Ignore, continue                                    |

---

## Prerequisites

- rclone installed, `gdrive` remote configured (OAuth authorized)
- Python package: `webdataset`
- Confirm `rclone check` works correctly before first run

---

# Phase 4: Mount Google Drive + Streaming Training

> Execution environment: Ubuntu (2×P100 16GB)
> Prerequisite: `03_phase2_3_pack_upload.md` (Phase 2-3 must be complete)
> Standalone program: `pipeline_train_stream.py`

---

## Goal

Mount Google Drive on the Ubuntu training machine and read shards via WebDataset streaming for training. No full dataset download required.

---

## Step 4a: Mount Google Drive

### Mount Command

```bash
rclone mount gdrive:SANPO-Dataset/shards/ ~/gdrive/shards/ \
  --vfs-cache-mode full \
  --vfs-cache-max-size 50G \
  --buffer-size 256M \
  --daemon
```

### Parameter Explanation

| Parameter              | Value  | Description                                                                            |
| ---------------------- | ------ | -------------------------------------------------------------------------------------- |
| `--vfs-cache-mode`     | `full` | Full caching; read shards are cached locally                                           |
| `--vfs-cache-max-size` | `50G`  | Local cache ceiling. Exceeding this triggers LRU eviction of least recently used cache |
| `--buffer-size`        | `256M` | Read buffer to improve sequential read performance                                     |
| `--daemon`             | —      | Run in background, does not occupy the terminal                                        |

### Cache Behavior

- rclone caches previously read shards on local disk
- Local storage usage stays near `--vfs-cache-max-size` ceiling and does not grow indefinitely
- Training machine needs **at least 50GB free disk space** for the cache

### Verify Mount Success

```bash
ls ~/gdrive/shards/
# Should show shard-000-000000.tar, shard-001-000000.tar, ...
```

---

## Step 4b: WebDataset Streaming Read

### Key Settings

| Item                  | Recommended Value   | Rationale                                                                                                                                         |
| --------------------- | ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| Shard shuffle         | `shardshuffle=True` | Randomly reorder shard read sequence each epoch                                                                                                   |
| Sample shuffle buffer | 2,000 samples       | ~2.3 GB RAM. Must be much larger than batch size to ensure frames in the same batch come from different sessions. Reduce to 1,000 if RAM is tight |
| Batch size            | Start at 8 per GPU  | See estimation below                                                                                                                              |

### Batch Size Estimation (2×P100 16GB)

| Setting              | Estimated per-GPU batch size | Total batch size |
| -------------------- | ---------------------------- | ---------------- |
| FP32 (conservative)  | 8                            | 16               |
| AMP FP16 (save VRAM) | 16                           | 32               |

> P100 lacks FP16 tensor cores, so AMP speed gains are limited, but it still saves VRAM.
> **Recommendation:** Start with `batch_size=8`, run a few iterations, observe VRAM usage via `nvidia-smi`, then gradually increase if stable.

### Code Example

```python
import io
import glob

import numpy as np
import webdataset as wds
import torch
from PIL import Image
from torch.utils.data import DataLoader

# --- Shard discovery (glob matches Phase 2 naming: shard-{batch}-{seq}.tar) ---
shards = sorted(glob.glob("/home/user/gdrive/shards/shard-*.tar"))

# --- Custom decoder: preserve seg mask as uint8 (decode("rgb") would convert to float) ---
def seg_decoder(key, data):
    """Decode _seg.png as raw uint8 numpy array, bypassing float normalization."""
    if key.endswith("_seg.png"):
        return np.array(Image.open(io.BytesIO(data)))  # (H, W, 3) uint8
    return None  # fallback to default decoder for other keys

# --- Dataset definition ---
dataset = (
    wds.WebDataset(shards, shardshuffle=True)
    .shuffle(2000)
    .decode(seg_decoder, "rgb")       # seg_decoder handles _seg.png; "rgb" handles .png
    .to_tuple("png", "_seg.png", "_depth.npy", "json")
    .batched(8)
)

# --- DataLoader ---
loader = DataLoader(dataset, batch_size=None, num_workers=4)

# --- Training loop ---
for batch in loader:
    rgb, seg, depth, metadata = batch
    # rgb:   (B, H, W, 3) float32 [0, 1]  — decoded by "rgb"
    # seg:   (B, H, W, 3) uint8           — decoded by seg_decoder, preserves label values
    # depth: (B, H, W) float16            — loaded from .npy
    # metadata: list of dict

    # TODO: Your training logic here
    pass
```

### Segmentation Mask Decoding

The `_seg.png` is preserved as uint8 by the custom decoder. Extract semantic and instance channels:

```python
# seg shape: (B, H, W, 3) uint8 — RGB
semantic = seg[:, :, :, 0]                          # R channel = class ID (0-30)
instance = seg[:, :, :, 1] * 256 + seg[:, :, :, 2]  # G*256 + B = instance ID
```

### Depth Decoding

```python
# depth is loaded from .npy, already float16
# If float32 is needed:
depth_f32 = depth.astype(np.float32)
```

### Corrupt Sample Handling

Shards may contain corrupt samples (truncated PNG, invalid .npy). Use WebDataset's error handler to skip them gracefully:

```python
def log_and_continue(exn):
    """Log the error and skip the corrupt sample."""
    logging.warning(f"Skipping corrupt sample: {exn}")
    return True

dataset = (
    wds.WebDataset(shards, shardshuffle=True, handler=log_and_continue)
    .shuffle(2000)
    .decode(seg_decoder, "rgb", handler=log_and_continue)
    .to_tuple("png", "_seg.png", "_depth.npy", "json", handler=log_and_continue)
    .batched(8)
)
```

If a specific shard has a high error rate (many corrupt samples), the shard should be considered unusable:

1. Log which shard and sample keys are failing
2. Remove the corrupt shard from Google Drive
3. Re-download the affected sessions' raw data from GCS
4. Re-run Phase 1-3 to repack (equivalent to discarding that camera's data and repacking)

---

## Step 4c: After Training

### Save Checkpoints

```bash
# Option 1: Write directly to mount path (simple but slow)
cp checkpoint.pt ~/gdrive/checkpoints/

# Option 2: Upload with rclone (recommended, more stable)
rclone copy ./checkpoints/ gdrive:SANPO-Dataset/checkpoints/ --progress
```

### Unmount

```bash
fusermount -u ~/gdrive/shards/
```

---

## Important Notes

### Training Speed Depends on Network Bandwidth

In streaming mode, data is pulled from the cloud during training. If the network is slow:

- GPU will idle while waiting for data
- Mitigate by increasing `num_workers` to prefetch more batches
- Ultimate bottleneck depends on rclone cache hit rate and network bandwidth

### Shuffle Limitations

WebDataset streaming cannot perform global shuffle. Only:

1. **Shard-level shuffle**: Randomly reorder shard sequence each epoch
2. **Buffer-level shuffle**: Randomly sample within a fixed-size buffer

A buffer of 2,000 samples out of ~89,000 total covers ~2.2%. This is not perfect shuffle, but is sufficient for most training tasks.

### Multi-GPU Training

If using DataParallel or DistributedDataParallel:

- WebDataset's `nodesplitter` can automatically distribute shards across different workers
- DistributedDataParallel is more efficient than DataParallel for 2×P100

---

## Prerequisites

- rclone installed, `gdrive` remote configured
- FUSE installed (`sudo apt install fuse`)
- Python packages: `webdataset`, `torch`, `ultralytics`, `opencv-python`, `numpy`
- At least 50GB free disk space (rclone cache)

---

## Error Handling

| Scenario                           | Action                                                                                                          |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| Mount fails                        | Check rclone config, FUSE installation, Google Drive authorization                                              |
| Shard read timeout                 | Network issue; increase `--buffer-size` or check connection                                                     |
| Single corrupt sample              | `handler=log_and_continue` skips it automatically; log the key for investigation                                |
| Many corrupt samples in one shard  | Shard is unusable. Remove from Google Drive, re-download affected sessions from GCS, re-run Phase 1-3 to repack |
| GPU OOM                            | Reduce batch size, enable AMP, enable gradient accumulation                                                     |
| rclone disconnects during training | rclone daemon usually auto-reconnects; if persistent, re-run mount                                              |

---

# Phase 5: CLI Progress Display

> Execution environment: Windows
> Applies to: `pipeline_pack_upload.py`

---

## Goal

Display structured progress information during pipeline execution so the user always knows where the pipeline is at.

---

## Output Format

```
Resuming pipeline: 24 / 560 sessions completed, 3 skipped
[Batch 4] Processing sessions 28~31 / 560
[Batch 4] Session abc123: downloading...
[Batch 4] Session abc123: validated (camera_chest: 527 frames)
[Batch 4] Session def456: skipped (segmentation_masks directory missing)
[Batch 4] Session ghi789: decimating...
[Batch 4] Session ghi789: preprocessing...
[Batch 4] Session ghi789: processed (53 frames, 159 patches)
...
[Batch 4] Packing 3 sessions into shards...
[Batch 4] Uploading shards...
[Batch 4] Verifying shards...
[Batch 4] Cleanup complete
[Batch 5] Processing sessions 32~35 / 560
...
```

---

## Requirements

1. **Batch header**: Show batch number and session range at the start of each batch: `[Batch N] Processing sessions X~Y / total`
2. **Per-session status**: Show session ID and current state as it progresses through the state machine
3. **Skip reason inline**: Sessions marked `skipped` display the reason (e.g. `skipped (depth_maps directory missing)`)
4. **Batch-level phases**: Show packing / upload / verify / cleanup progress
5. **Resume summary**: On startup, show how many sessions are completed and skipped before starting the next batch

---

## Formatting Functions

### `format_batch_range(batch_index, batch_size, total_sessions) -> str`

```
batch_index=0, batch_size=4, total_sessions=560  → "1~4 / 560"
batch_index=2, batch_size=4, total_sessions=560  → "9~12 / 560"
batch_index=139, batch_size=4, total_sessions=560 → "557~560 / 560"  (last batch, clamped)
```

### `format_resume_summary(progress) -> str`

Counts sessions by status:

```
"Resuming pipeline: 24 / 560 sessions completed, 3 skipped"
```

Where "completed" = status `cleaned`, "skipped" = status `skipped`.

---

## Integration

These functions are called from the existing pipeline phases — no new phase execution flow. The logging calls in Phase 0-3 functions are updated to use the batch-aware format.

---

# Test Spec: SANPO Pipeline Unit Tests

> Scope: Only test logic that, if broken, would silently corrupt training data or cause confusing failures.
> Out of scope: Directory creation, file deletion, rclone commands, gcloud download, progress JSON I/O.

---

## Test File

`tests/scripts/test_pipeline.py`

---

## 1. Decimation Sampling Logic

### Test 1.1: Basic sampling with default parameters

```
Input:  frames = ["000000.png", "000001.png", ..., "000526.png"]  (527 files)
Config: interval=10, offset=0
Expected output indices: [0, 10, 20, 30, ..., 520]
Expected count: 53
```

### Test 1.2: Sampling with non-zero offset

```
Input:  frames = ["000000.png", "000001.png", ..., "000526.png"]  (527 files)
Config: interval=10, offset=3
Expected output indices: [3, 13, 23, 33, ..., 523]
Expected count: 53
```

### Test 1.3: Sampling with interval larger than total frames

```
Input:  frames = ["000000.png", ..., "000004.png"]  (5 files)
Config: interval=10, offset=0
Expected output indices: [0]
Expected count: 1
```

### Test 1.4: Sampling with offset >= total frames

```
Input:  frames = ["000000.png", ..., "000004.png"]  (5 files)
Config: interval=10, offset=7
Expected output indices: []
Expected count: 0
→ Should handle gracefully (empty list, not crash)
```

### Test 1.5: Determinism — same input + same config = same output every time

```
Run decimation twice with identical input and config.
Assert both runs produce identical index lists.
```

### Test 1.6: Interval=1 returns all frames (edge case)

```
Input:  frames = ["000000.png", ..., "000009.png"]  (10 files)
Config: interval=1, offset=0
Expected count: 10 (all frames)
```

---

## 2. Three Data Types Stay Synchronized

### Test 2.1: Same indices applied to all three types

```
Input:
  video_frames/      → ["000000.png", "000001.png", ..., "000049.png"]
  segmentation_masks/ → ["000000.png", "000001.png", ..., "000049.png"]
  depth_maps/         → ["000000.float16.gz", "000001.float16.gz", ..., "000049.float16.gz"]
Config: interval=10, offset=0

Assert:
  selected_frames  = ["000000.png", "000010.png", "000020.png", "000030.png", "000040.png"]
  selected_segs    = ["000000.png", "000010.png", "000020.png", "000030.png", "000040.png"]
  selected_depths  = ["000000.float16.gz", "000010.float16.gz", "000020.float16.gz", "000030.float16.gz", "000040.float16.gz"]

  All three lists have identical numeric prefixes.
```

### Test 2.2: Filenames with different extensions still align by index

```
Input:
  video_frames/      → ["frame_000.png", "frame_001.png", "frame_002.png"]
  segmentation_masks/ → ["seg_000.png", "seg_001.png", "seg_002.png"]
  depth_maps/         → ["depth_000.float16.gz", "depth_001.float16.gz", "depth_002.float16.gz"]
Config: interval=2, offset=0

Assert: selected indices are [0, 2] for all three types.
The function uses positional index (sorted order), not filename matching.
```

---

## 3. Pre-Decimation Validation Logic

### Test 3.1: All three directories present, counts match → PASS

```
Input:
  video_frames/       → 50 files
  segmentation_masks/ → 50 files
  depth_maps/         → 50 files
Expected: validation passes, returns True
```

### Test 3.2: segmentation_masks directory missing → FAIL (skip)

```
Input:
  video_frames/       → 50 files
  segmentation_masks/ → does not exist
  depth_maps/         → 50 files
Expected: validation fails, returns False with reason "segmentation_masks directory missing"
```

### Test 3.3: segmentation_masks directory exists but empty → FAIL (skip)

```
Input:
  video_frames/       → 50 files
  segmentation_masks/ → 0 files (empty directory)
  depth_maps/         → 50 files
Expected: validation fails, returns False with reason "segmentation_masks directory empty"
```

### Test 3.4: depth_maps directory missing → FAIL (skip)

```
Input:
  video_frames/       → 50 files
  segmentation_masks/ → 50 files
  depth_maps/         → does not exist
Expected: validation fails, returns False with reason "depth_maps directory missing"
```

### Test 3.5: File count mismatch — seg has fewer files → FAIL (skip)

```
Input:
  video_frames/       → 50 files
  segmentation_masks/ → 30 files
  depth_maps/         → 50 files
Expected: validation fails, returns False with reason containing the actual counts
  e.g. "File count mismatch: video_frames=50, segmentation_masks=30, depth_maps=50"
```

### Test 3.6: File count mismatch — depth has fewer files → FAIL (skip)

```
Input:
  video_frames/       → 50 files
  segmentation_masks/ → 50 files
  depth_maps/         → 45 files
Expected: validation fails, returns False with reason containing actual counts
```

### Test 3.7: Frame ID misalignment → FAIL (skip)

```
Input:
  video_frames/       → ["000000.png", "000001.png", "000002.png"]
  segmentation_masks/ → ["000000.png", "000001.png", "000003.png"]  ← 000003 instead of 000002
  depth_maps/         → ["000000.float16.gz", "000001.float16.gz", "000002.float16.gz"]
Expected: validation fails, returns False with reason "Frame ID misalignment"
```

### Test 3.8: All directories present, all counts match, all IDs align → PASS

```
Input:
  video_frames/       → ["000000.png", "000005.png", "000010.png"]
  segmentation_masks/ → ["000000.png", "000005.png", "000010.png"]
  depth_maps/         → ["000000.float16.gz", "000005.float16.gz", "000010.float16.gz"]
Expected: validation passes, returns True
  (non-contiguous frame IDs are fine as long as all three types match)
```

---

## 4. Error Messages

### Test 4.1: Validation failure includes actionable detail

```
For each failing validation test (3.2 through 3.7):
  Assert the error/log message contains:
    - session_id
    - which check failed
    - actual values (e.g. file counts, mismatched IDs)

  The message should be sufficient for a human to understand what went wrong
  without looking at the source code.
```

### Test 4.2: Decimation with empty frame list produces clear message

```
Input: frames = [] (empty directory)
Expected: does not crash. Returns empty list or logs a clear warning
  e.g. "Session {session_id}: video_frames directory is empty, skipping"
```

### Test 4.3: Decimation config missing required fields

```
Input config: {"decimation": {"interval": 10}}  ← offset missing
Expected: raises clear error or uses default offset=0 with a warning log
```

---

## 5. CLI Progress Display (Phase 5)

> Feature spec is in Phase 5 SDD above. Tests below verify the formatting functions.

### Test 5.1: Progress display shows correct session range

```
Input: batch_size=8, total_sessions=560, current_batch=0
Expected output contains: "1~8 / 560"

Input: batch_size=8, total_sessions=560, current_batch=2
Expected output contains: "17~24 / 560"
```

### Test 5.2: Last batch handles remainder correctly

```
Input: batch_size=8, total_sessions=563, current_batch=70 (last batch)
Expected output contains: "561~563 / 563"  (not 561~568)
```

### Test 5.3: Resume message shows correct counts

```
Input: progress file has 24 "cleaned" + 3 "skipped" + 533 "pending"
Expected output contains: "24 / 560 sessions completed, 3 skipped"
```

---

## Implementation Notes

- Use `pytest` with `tmp_path` fixture to create temporary directory structures for validation tests
- Decimation tests are pure logic — no filesystem needed, just test the index calculation function
- Validation tests need mock directories with dummy files (empty `.png` / `.float16.gz` files are fine, content doesn't matter for validation)
- CLI progress tests can capture stdout or test the formatting function directly

---

# Phase 6: Modular Refactoring SDD

> **Goal**: Refactor the monolithic `pipeline_pack_upload.py` (~1137 lines) into a modular package, externalize hardcoded config, and rename `pipeline_train_stream.py` for multi-split usage.

---

## 1. Motivation

| Problem                                                                          | Impact                                                                |
| -------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| Single file contains all phases (0-3, 5) + utilities + orchestrator              | Hard to navigate, review, and test independently                      |
| Hardcoded paths, constants, GCS bucket, GDrive remote                            | Cannot reuse for test set or different dataset without editing source |
| `_preprocess_camera` always calls `image_crop` + `process_and_patch_sanpo_depth` | Test set may require different preprocessing (e.g., no cropping)      |
| `pipeline_train_stream.py` name implies train-only                               | Will also serve test set streaming                                    |
| `load_session_ids()` reads a hardcoded train split file path                     | Cannot switch to test split                                           |

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

| Module            | Content                                                                                                                               | Depends On                                                           |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| `config.py`       | `load_pipeline_config()` — reads `configs/pipeline_config.yaml`, returns typed dict                                                   | `yaml`                                                               |
| `progress.py`     | `load_progress()`, `save_progress()`, `_now_iso()`, `_available_disk_gb()`, `load_session_ids()`                                      | `config.py`                                                          |
| `download.py`     | `_gcloud_ls()`, `_gcloud_cp()`, `_integrity_check_post_download()`, `_delete_raw_session()`, `download_session()`, `download_batch()` | `config.py`, `progress.py`                                           |
| `validate.py`     | `_extract_frame_id()`, `_discover_cameras()`, `_validate_camera()`, `_delete_camera_data()`, `validate_session()`                     | `config.py`, `progress.py`                                           |
| `decimate.py`     | `_decimate_indices()`, `decimate_session()`                                                                                           | `config.py`, `progress.py`, `validate.py` (for `_extract_frame_id`)  |
| `preprocess.py`   | `_preprocess_camera()`, `_integrity_check_post_preprocess()`, `preprocess_session()`, `validate_decimate_preprocess()`                | `config.py`, `progress.py`, `validate.py`, `decimate.py`             |
| `pack.py`         | `_parse_processed_filename()`, `_get_batch_index()`, `_increment_batch_index()`, `pack_shards()`                                      | `config.py`, `progress.py`                                           |
| `upload.py`       | `upload_shards()`, `verify_shards()`, `cleanup_batch()`                                                                               | `config.py`, `progress.py`, `pack.py` (for `_increment_batch_index`) |
| `display.py`      | `format_batch_range()`, `format_resume_summary()`                                                                                     | None (pure functions)                                                |
| `run_pipeline.py` | `main()` + `parse_args()`                                                                                                             | All `data_pipeline.*` modules                                        |
| `run_stream.py`   | Mount + WebDataset streaming                                                                                                          | `config.py` (for GDrive remote)                                      |

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
  enable_image_crop: true # Set false to skip crop+resize (e.g., for test set)
  enable_depth_process: true # Set false to skip depth processing

# --- Phase 2: Packing ---
packing:
  shard_max_size: 1.5e9 # bytes, ~1.5 GB per shard

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

| Currently hardcoded                                             | Moves to YAML            | Stays in code                                                                                                |
| --------------------------------------------------------------- | ------------------------ | ------------------------------------------------------------------------------------------------------------ |
| `PROJECT_ROOT`                                                  | —                        | `config.py` (computed from `__file__`, 向上三層: `config.py` → `data_pipeline/` → `scripts/` → project root) |
| `DATA_DIR`, `RAW_DIR`, `PROCESSED_DIR`, `SHARDS_DIR`            | `paths.*`                | —                                                                                                            |
| `PROGRESS_FILE`, `SESSION_IDS_FILE`, `DECIMATION_CONFIG_FILE`   | `paths.*`                | —                                                                                                            |
| `GCS_BUCKET`, `GDRIVE_REMOTE`                                   | `remote.*`               | —                                                                                                            |
| `BATCH_SIZE`, `DISK_SAFETY_FACTOR`, `ESTIMATED_SESSION_SIZE_GB` | `download.*`             | —                                                                                                            |
| `SHARD_MAX_SIZE`                                                | `packing.shard_max_size` | —                                                                                                            |
| `CAMERAS`, `PATCH_POSITIONS`                                    | `preprocess.*`           | —                                                                                                            |
| Logging setup                                                   | —                        | Each module's own logger (standard pattern)                                                                  |
| State machine transitions                                       | —                        | `progress.py` (logic, not config)                                                                            |

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
  enable_image_crop: true # false → skip crop+resize, copy raw frames directly
  enable_depth_process: true # false → skip depth processing
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

| Before                                            | After                                          |
| ------------------------------------------------- | ---------------------------------------------- |
| `scripts/pipeline_train_stream.py`                | `scripts/run_stream.py`                        |
| Hardcoded `DEFAULT_MOUNT_POINT = ~/gdrive/shards` | Read from `config["streaming"]["mount_point"]` |
| Hardcoded `GDRIVE_REMOTE`                         | Read from `config["remote"]["gdrive_remote"]`  |
| No `--config` argument                            | Add `--config` argument                        |

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

| Step | Action                                                                              | Verify                                         |
| ---- | ----------------------------------------------------------------------------------- | ---------------------------------------------- |
| 1    | Create `configs/pipeline_config.yaml` with current hardcoded values                 | YAML loads correctly                           |
| 2    | Create `scripts/data_pipeline/__init__.py` and `config.py`                          | `load_pipeline_config()` returns expected dict |
| 3    | Extract `progress.py` from current helpers                                          | Import works                                   |
| 4    | Extract `display.py` (pure functions, no dependencies)                              | Existing Phase 5 tests pass                    |
| 5    | Extract `validate.py` + `decimate.py`                                               | Existing validation + decimation tests pass    |
| 6    | Extract `download.py`                                                               | Import works                                   |
| 7    | Extract `preprocess.py`                                                             | Import works                                   |
| 8    | Extract `pack.py` + `upload.py`                                                     | Import works                                   |
| 9    | Create `scripts/run_pipeline.py` as thin entry point                                | Full pipeline works                            |
| 10   | Rename `pipeline_train_stream.py` → `run_stream.py`, integrate config               | Streaming works                                |
| 11   | Update test imports                                                                 | All 22 tests pass                              |
| 12   | Delete old `scripts/pipeline_pack_upload.py` and `scripts/pipeline_train_stream.py` | No leftover files                              |

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

| Function               | New parameter |
| ---------------------- | ------------- |
| `load_progress()`      | `config`      |
| `save_progress()`      | `config`      |
| `load_session_ids()`   | `config`      |
| `download_session()`   | `config`      |
| `download_batch()`     | `config`      |
| `validate_session()`   | `config`      |
| `decimate_session()`   | `config`      |
| `_preprocess_camera()` | `config`      |
| `preprocess_session()` | `config`      |
| `pack_shards()`        | `config`      |
| `upload_shards()`      | `config`      |
| `verify_shards()`      | `config`      |
| `cleanup_batch()`      | `config`      |

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

---

## Phase 7. Phase 0 Step 0: Pre-Download Validation

> Execution environment: Windows
> Runs before any `gcloud storage cp` download

---

### Goal

Validate that a session's data on GCS is complete and aligned **before** downloading, to avoid wasting bandwidth and disk space on sessions that would fail Pre-Decimation Validation anyway.

---

### Rationale

Pre-Decimation Validation (Phase 1 Step 0) catches incomplete sessions **after** download, deleting ~8 GB of raw data per failed session. Pre-Download Validation performs the identical three checks remotely via `gcloud storage ls`, skipping the download entirely when validation fails. This saves:

- Download time and bandwidth
- Temporary disk space consumption
- Cleanup I/O

---

### Validation Checks

Identical to Pre-Decimation Validation, but executed against GCS paths instead of local paths. For each camera within a session (`camera_chest/left`, `camera_head/left` if present):

```
GCS base: gs://gresearch/sanpo_dataset/v0/sanpo-real/{session_id}/{camera}/left/

video_frames_gcs  = {GCS base}/video_frames/
seg_masks_gcs     = {GCS base}/segmentation_masks/
depth_maps_gcs    = {GCS base}/depth_maps/
```

#### Check 1: All three directories exist and are non-empty

Use `gcloud storage ls` to list each directory. If the command returns non-zero exit code or produces no output → FAIL.

```python
# Reuse existing _gcloud_ls() from download.py for existence check
# New helper: _gcloud_list_files() to list and parse filenames from GCS path
result = subprocess.run(
    ["gcloud", "storage", "ls", gcs_path],
    capture_output=True, text=True, shell=True
)
# Parse result.stdout → list of filenames
```

#### Check 2: File counts match

```python
frame_files = _gcloud_list_files(video_frames_gcs)   # list of .png filenames
seg_files   = _gcloud_list_files(seg_masks_gcs)       # list of .png filenames
depth_files = _gcloud_list_files(depth_maps_gcs)      # list of .float16.gz filenames

if not (len(frame_files) == len(seg_files) == len(depth_files)):
    # FAIL: File count mismatch
    → skip this camera
```

#### Check 3: Frame ID alignment

```python
# Reuse existing _extract_frame_id() from validate.py
frame_ids = [_extract_frame_id(f) for f in sorted(frame_files)]
seg_ids   = [_extract_frame_id(f) for f in sorted(seg_files)]
depth_ids = [_extract_frame_id(f) for f in sorted(depth_files)]

if frame_ids != seg_ids or frame_ids != depth_ids:
    # FAIL: Frames are not aligned
    → skip this camera
```

---

### On Camera Validation Failure

If any check fails for a camera:

1. **Log the reason** (which camera, which check failed, file counts if applicable)
2. **Do NOT download** that camera's data
3. **Continue** to validate the next camera in the same session

> Unlike Pre-Decimation Validation, there is no local data to delete — the download has not happened yet.

### After All Cameras Validated

If **all cameras failed** → mark the session as `skipped`:

```json
{
  "status": "skipped",
  "skip_reason": "pre-download: all cameras failed validation (chest: seg missing, head: depth count mismatch)",
  "skipped_at": "2026-04-01T10:00:00"
}
```

Skip download entirely. No raw data was ever written.

If **at least one camera passed** → proceed to download only the cameras that passed validation.

---

### Implementation: Refactor `_validate_camera()` to Accept a Listing Strategy

Instead of duplicating validation logic, refactor `_validate_camera()` to accept a **file-listing function** parameter. Local and GCS validation share the same 3-check logic — only the way file lists are obtained differs.

#### Current signature

```python
def _validate_camera(session_dir: Path, camera: str) -> tuple[bool, str, int]:
```

Hardcoded to `os.listdir()` on local paths.

#### Refactored signature

```python
def _validate_camera(
    camera: str,
    list_files_fn: Callable[[str], list[str]],
    video_path: str,
    seg_path: str,
    depth_path: str,
) -> tuple[bool, str, int]:
```

- `list_files_fn(path)` → returns sorted filenames under `path`, or empty list if path doesn't exist
- For **local**: wraps `os.listdir()` + existence check
- For **GCS**: wraps `gcloud storage ls` + output parsing

#### Two listing implementations

```python
# Local listing (validate.py) — wraps existing os.listdir logic
def _list_local_files(dir_path: str) -> list[str]:
    p = Path(dir_path)
    if not p.exists() or not any(p.iterdir()):
        return []
    return sorted(os.listdir(p))

# GCS listing (download.py) — NEW, the only new function needed
def _gcloud_list_files(gcs_path: str) -> list[str]:
    result = subprocess.run(
        ["gcloud", "storage", "ls", gcs_path],
        capture_output=True, text=True, shell=True
    )
    if result.returncode != 0:
        return []
    # Each line is a full GCS URI → extract filename only
    return sorted(
        line.rstrip("/").split("/")[-1]
        for line in result.stdout.strip().splitlines()
        if line.strip()
    )
```

#### Callers

| Caller | Listing function | Paths |
| --- | --- | --- |
| `validate_session()` (existing, Pre-Decimation) | `_list_local_files` | `data/raw/{session_id}/{camera}/left/{dtype}` |
| `download_session()` (updated, Pre-Download) | `_gcloud_list_files` | `gs://.../{session_id}/{camera}/left/{dtype}` |

Both call the **same** `_validate_camera()` — zero duplicated validation logic.

#### New code needed

| Item | Module | Description |
| --- | --- | --- |
| `_gcloud_list_files(gcs_path)` | `download.py` | `gcloud storage ls` → parse output → sorted filename list |
| `_list_local_files(dir_path)` | `validate.py` | Thin wrapper around `os.listdir` (extracted from current `_validate_camera`) |
| Refactored `_validate_camera()` | `validate.py` | Accept `list_files_fn` + 3 path strings instead of hardcoded local paths |

---

### Integration into Download Flow

Pre-Download Validation runs inside `download_session()`, **before** any `gcloud storage cp`:

```
download_session(session_id):
    1. Pre-Download Validation (NEW)
       → all cameras failed → return "skipped"
       → get list of valid_cameras
    2. Download only valid_cameras (existing logic, scoped to valid cameras)
    3. Post-download integrity check (existing _integrity_check_post_download)
    → return "downloaded"
```

---

### Comparison: Pre-Download vs Pre-Decimation Validation

| Aspect              | Pre-Download Validation (Phase 0 Step 0) | Pre-Decimation Validation (Phase 1 Step 0) |
| ------------------- | ---------------------------------------- | ------------------------------------------ |
| **When**            | Before `gcloud storage cp`               | After download, before decimation          |
| **Data source**     | GCS remote (`gcloud storage ls`)         | Local filesystem (`os.listdir`)            |
| **Check 1**         | 3 dirs exist & non-empty on GCS          | 3 dirs exist & non-empty locally           |
| **Check 2**         | File counts match (remote listing)       | File counts match (local listing)          |
| **Check 3**         | Frame ID alignment (remote listing)      | Frame ID alignment (local listing)         |
| **On camera fail**  | Skip download for that camera            | Delete that camera's raw data              |
| **On session fail** | Mark `skipped`, skip download entirely   | Mark `skipped`, delete raw data            |
| **Cost of failure** | Only `gcloud ls` API calls (~seconds)    | ~8 GB download wasted + deletion I/O       |
| **subprocess flag** | `shell=True`                             | N/A (local filesystem)                     |

---

### Error Handling

| Scenario                                  | Action                                                             |
| ----------------------------------------- | ------------------------------------------------------------------ |
| `gcloud storage ls` fails (network error) | Treat as inconclusive — proceed to download (fail-open)            |
| `gcloud storage ls` returns empty         | Camera fails Check 1 — skip that camera                            |
| GCS listing is very slow                  | No timeout override needed — `gcloud` CLI handles its own timeouts |

> **Fail-open design:** If remote validation cannot be completed (e.g., network issue), the pipeline falls through to download + Pre-Decimation Validation as before. Pre-Download Validation is an optimization, not a gate.

---

## 4. Strict Batch Control

### Problem

Current `download_batch()` always downloads `batch_size` (4) **new** sessions from the pending list, without checking how many sessions are already in-flight (status between `downloaded` and `verified`). This causes:

- **Restart after partial download**: If the pipeline downloaded 2 sessions and was interrupted, restarting downloads 4 more → 6 sessions in-flight, exceeding the intended batch size and disk budget.
- **No gate between batches**: The next batch starts downloading before the previous batch finishes uploading and cleanup.

### Goal

Enforce strict batch boundaries: exactly `batch_size` sessions go through the full pipeline (download → clean) before any new sessions are downloaded.

---

### Design: In-Flight Slot Counting

#### Definition

A session is **in-flight** if its status is any of:

```
"downloaded", "validated", "decimated", "processed", "packed", "uploaded", "verified"
```

These are sessions that have been downloaded but not yet cleaned (i.e., still occupying local disk space or awaiting completion).

Sessions with status `"pending"`, `"error"`, `"skipped"`, or `"cleaned"` are **not** in-flight.

#### Slot Calculation

```python
in_flight = [sid for sid in all_session_ids
             if progress["sessions"].get(sid, {}).get("status") in IN_FLIGHT_STATUSES]
slots_available = batch_size - len(in_flight)
```

- `slots_available > 0` → download exactly `slots_available` sessions from pending list
- `slots_available <= 0` → skip download entirely, process existing in-flight sessions

#### Behavior by Scenario

| Scenario | In-flight | Slots | Action |
| --- | --- | --- | --- |
| Fresh start | 0 | 4 | Download 4 |
| Restart after 2 downloaded | 2 | 2 | Download 2 more, then process all 4 |
| Restart after 4 downloaded, 2 validated | 4 | 0 | Skip download, continue processing |
| All 4 cleaned | 0 | 4 | Download next 4 |
| 3 cleaned, 1 still uploading | 1 | 3 | Download 3 more (fill to 4) |
| Only 2 pending left (near end) | 0 | 4 | Download 2 (all remaining) |

---

### Changes Required

#### `download_batch()` in `download.py`

**Current logic:**

```python
pending = [sid for sid in all_session_ids if status in ("pending", "error")]
batch = pending[:batch_size]  # Always takes batch_size from pending
```

**New logic:**

```python
# 1. Count in-flight sessions
in_flight = [sid for sid in all_session_ids
             if status in IN_FLIGHT_STATUSES]
slots_available = batch_size - len(in_flight)

# 2. If batch is full, skip download
if slots_available <= 0:
    return progress

# 3. Only download enough to fill remaining slots
pending = [sid for sid in all_session_ids if status in ("pending", "error")]
batch = pending[:slots_available]
```

Remove the `for batch_start in range(...)` loop and the `break` — the function now downloads exactly one partial-or-full batch per call, controlled by slot counting.

#### `run_pipeline.py` main loop

No change needed. The existing `while True` loop already:

1. Calls `download_batch()` — which now respects slots
2. Processes all in-flight sessions through validate/decimate/preprocess/pack/upload/clean
3. Loops back — `download_batch()` sees 0 in-flight after cleanup, downloads next batch

---

### Pipeline Flow with Strict Batching

```
while sessions remain:
    ┌─ download_batch()
    │    count in-flight sessions
    │    slots = batch_size - in_flight
    │    if slots > 0: download min(slots, len(pending)) sessions
    │
    ├─ validate_decimate_preprocess()   ← processes all "downloaded" sessions
    ├─ pack_shards()                    ← processes all "processed" sessions
    ├─ upload_shards()                  ← processes all "packed" sessions
    ├─ verify_shards()                  ← processes all "uploaded" sessions
    ├─ cleanup_batch()                  ← processes all "verified" sessions
    │
    └─ loop back → download_batch() sees 0 in-flight → downloads next batch
```

---

### Disk Space Guarantee

With strict batching, the maximum local disk usage is bounded:

```
max_disk = batch_size × estimated_session_size_gb × safety_factor
         = 4 × 8 GB × 1.5
         = 48 GB
```

This is within the ~97 GB available on the Windows machine, with margin for processed + shard data coexisting.
